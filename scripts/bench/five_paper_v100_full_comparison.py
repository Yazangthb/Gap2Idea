"""One-request full-paper control for the V100 27B local-context pilot."""
import json
import time

import five_paper_context_pilot as q
import five_paper_context_v100 as v
import five_paper_dev_tuning as d
import five_paper_pilot as p
from baseline_evidence import digest


def main():
    data,items=q.frozen();q.check_tracked(v.CONFIG)
    q.check_tracked(p.ROOT/'scripts/bench/five_paper_v100_full_comparison.py')
    q.check_tracked(q.DEST/'27b_v100_q4_neighbors.json')
    plan=json.loads(v.CONFIG.read_text(encoding='utf8'))
    if plan!=v.plan():raise SystemExit('V100 plan changed')
    path=q.DEST/'27b_v100_q4_full_papers.json'
    if path.exists():raise SystemExit('Preserve existing output')
    valid,user=d.user_text(data,'full');system=d.system('neutral');keys=data['keys'];rows=data['rows'];c=p.client()
    if v.MODEL not in {x.id for x in c.models.list().data}:raise SystemExit('V100 model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None
    for _ in range(2):
        start=time.time();item=None
        try:
            resp=c.chat.completions.create(model=v.MODEL,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-start,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives')
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in valid for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','method':'full_papers','model':v.MODEL,'runtime_plan_sha256':p.sha(v.CONFIG),
                    'input_sha256':digest(user),'system_sha256':digest(system),'sentence_count':len(valid),
                    'attempts':attempts,'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-start,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','method':'full_papers','attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'status':result['status'],'metrics':result.get('metrics'),
                      'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok':raise SystemExit('Request failed; evidence preserved')


if __name__=='__main__':main()
