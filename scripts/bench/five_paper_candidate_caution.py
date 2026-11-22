"""Test one added candidate-fallibility instruction on both exact Qwen inputs."""
import json
import sys
import time

import five_paper_context_pilot as q
import five_paper_context_v100 as v
import five_paper_dev_tuning as d
import five_paper_repeatability as r
import five_paper_pilot as p
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/five_paper_candidate_caution_2026-11-22'
PLAN=DEST/'plan.json'
ADDITION=('Some listed candidates may be genuine research gaps, while others may not be; '
          'assess each candidate independently and do not assume it is correct merely because it was selected.')


def plan_for(data,inputs):
    return {'purpose':'Test one candidate-fallibility instruction on the two exact previously repeated inputs',
        'baseline_system_sha256':digest(d.system('neutral')),'addition':ADDITION,
        'system_sha256':digest(d.system('neutral')+'\n\n'+ADDITION),
        'model':v.MODEL,'model_digest':v.DIGEST,'temperature':0,'seed':0,'thinking':False,
        'max_output_tokens':8192,'context_length':131072,'requests':'one per input, no model retries except invalid output',
        'inputs':{k:{'input_sha256':digest(x['user']),'candidates':len(x['valid']),
                     'baseline_result_sha256':p.sha(x['original'])} for k,x in inputs.items()},
        'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_candidate_caution.py'),
        'caveat':'Same five development papers, already examined; LLM-generated labels, exploratory prompt change.'}


def main(name):
    if name not in ('zero','fine'):raise SystemExit('Unknown input')
    data,inputs=r.frozen();q.check_tracked(PLAN);q.check_tracked(p.ROOT/'scripts/bench/five_paper_candidate_caution.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(data,inputs):raise SystemExit('Plan or input changed')
    path=DEST/f'{name}_caution.json'
    if path.exists():raise SystemExit('Preserve existing evidence')
    x=inputs[name];keys=data['keys'];valid=x['valid'];system=d.system('neutral')+'\n\n'+ADDITION;c=p.client()
    if v.MODEL not in {m.id for m in c.models.list().data}:raise SystemExit('Expected model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None
    for _ in range(2):
        started=time.time();item=None
        try:
            resp=c.chat.completions.create(model=v.MODEL,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':system},{'role':'user','content':x['user']}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-started,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives')
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in valid for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','input':name,'plan_sha256':p.sha(PLAN),'input_sha256':digest(x['user']),
                    'candidate_count':len(valid),'kept':[list(k) for k in sorted(kept)],'attempts':attempts,
                    'metrics':p.metrics(data['rows'],kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-started,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','input':name,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'input':name,'status':result['status'],'metrics':result.get('metrics'),
                      'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok':raise SystemExit('Failed; evidence preserved')


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        data,inputs=r.frozen();DEST.mkdir(parents=True,exist_ok=True)
        PLAN.write_text(json.dumps(plan_for(data,inputs),indent=2)+'\n',encoding='utf8');print('Prepared plan; commit before inference')
    elif sys.argv[1]=='run':main(sys.argv[2])
    else:raise SystemExit('Unknown command')
