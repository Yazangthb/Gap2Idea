"""Repeat the exact zero-shot and fine-tuned neighbor-context Qwen inputs.

Freeze/commit plan before 3 additional calls per input. Same 27B Q4 model,
temperature 0, seed 0, 131072 context, schema and valid-ID deduplication.
Failures are recorded rather than scored as negatives.
"""
import itertools
import json
import sys
import time

import five_paper_context_pilot as q
import five_paper_context_v100 as v
import five_paper_dev_tuning as d
import five_paper_ft_neighbors as f
import five_paper_pilot as p
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/five_paper_repeatability_2026-11-22'
PLAN=DEST/'plan.json'
ORDER=[('zero',1),('fine',1),('fine',2),('zero',2),('zero',3),('fine',3)]


def inputs():
    data,items=q.source();mapping={x['id']:x for x in items};lines=[];last=None
    for i,r in enumerate(data['rows']):
        if i not in mapping:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        x=mapping[i]
        lines.append(f'[{i}] {r["sentence"]}\nContext: Previous: {x["previous"]} Next: {x["next"]}')
    zero_user='\n'.join(lines)
    selected,ftkeys,tau,ft_user=f.build()
    if selected['keys']!=data['keys']:raise ValueError('Different paper order')
    zero_base=q.DEST/'27b_v100_q4_neighbors.json';fine_base=f.RESULT
    z=json.loads(zero_base.read_text(encoding='utf8'));ftplan=json.loads(f.PLAN.read_text(encoding='utf8'))
    if digest(zero_user)!=z['input_sha256'] or digest(ft_user)!=ftplan['input_sha256']:
        raise ValueError('Reconstructed request differs from original')
    return data,{'zero':{'user':zero_user,'valid':data['candidate_keys'],'original':zero_base},
                 'fine':{'user':ft_user,'valid':ftkeys,'original':fine_base}}


def plan_for(data,ins):
    return {'purpose':'Estimate short-run repeatability of exactly the previously compared single-request inputs',
      'repeats_per_input':3,'call_order':[list(x) for x in ORDER],'model':v.MODEL,'ollama_digest':v.DIGEST,
      'system_sha256':digest(d.system('neutral')),'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192,
      'context_length':131072,'gpu':'single V100 GPU 0',
      'inputs':{k:{'input_sha256':digest(x['user']),'candidate_count':len(x['valid']),
                   'original_result_sha256':p.sha(x['original'])} for k,x in ins.items()},
      'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_repeatability.py'),
      'scope':'five development papers with LLM labels; repeated samples do not prove long-term determinism',
      'invalid_output':'record failure; do not treat it as a negative result; duplicate valid IDs deduplicated as in original'}


def frozen():
    q.frozen();q.check_tracked(f.PLAN);q.check_tracked(f.RESULT);q.check_tracked(q.DEST/'27b_v100_q4_neighbors.json')
    data,ins=inputs();q.check_tracked(PLAN);q.check_tracked(p.ROOT/'scripts/bench/five_paper_repeatability.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(data,ins):raise SystemExit('Plan or input changed')
    return data,ins


def run(name,index):
    if (name,index) not in ORDER:raise SystemExit('Unknown repetition')
    data,ins=frozen();path=DEST/f'{name}_repeat_{index}.json'
    if path.exists():raise SystemExit('Preserve existing output')
    settings=ins[name];keys=data['keys'];wanted=settings['valid'];c=p.client();system=d.system('neutral')
    if v.MODEL not in {x.id for x in c.models.list().data}:raise SystemExit('Expected Q4 model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    start=time.time();item={};result=None
    try:
        resp=c.chat.completions.create(model=v.MODEL,temperature=0,max_tokens=8192,
            messages=[{'role':'system','content':system},{'role':'user','content':settings['user']}],
            response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
            extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
        item={'seconds':round(time.time()-start,3),'finish_reason':resp.choices[0].finish_reason,
              'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content}
        if item['finish_reason']!='stop':raise ValueError('Did not finish')
        obj=json.loads(item['response']);ids=obj.get('positives')
        if (set(obj)!={'positives'} or not isinstance(ids,list) or
            any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in wanted for i in ids)):
            raise ValueError('Invalid or non-input ids')
        item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
        result={'status':'ok','input':name,'repeat':index,'plan_sha256':p.sha(PLAN),'input_sha256':digest(settings['user']),
                'attempt':item,'kept':[list(x) for x in sorted(kept)],'metrics':p.metrics(data['rows'],kept)}
    except Exception as e:
        result={'status':'failed','input':name,'repeat':index,'plan_sha256':p.sha(PLAN),
                'error_type':type(e).__name__,'attempt':item or {'seconds':round(time.time()-start,3)}}
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'input':name,'repeat':index,'status':result['status'],'metrics':result.get('metrics')}),flush=True)


def analyze():
    data,ins=frozen();result={}
    for name in ('zero','fine'):
        original=json.loads(ins[name]['original'].read_text(encoding='utf8'));runs=[('original',original)]
        for i in range(1,4):
            path=DEST/f'{name}_repeat_{i}.json';q.check_tracked(path)
            rec=json.loads(path.read_text(encoding='utf8'))
            if rec['plan_sha256']!=p.sha(PLAN):raise SystemExit('Run plan mismatch')
            runs.append((f'repeat_{i}',rec))
        good=[(label,{tuple(x) for x in r['kept']},r) for label,r in runs if r['status']=='ok']
        pairs=[]
        for (la,a,_),(lb,b,_) in itertools.combinations(good,2):
            pairs.append({'a':la,'b':lb,'symmetric_difference':len(a^b),'jaccard':len(a&b)/len(a|b) if a|b else 1})
        union=set().union(*(s for _,s,_ in good)) if good else set()
        intersection=set.intersection(*(s for _,s,_ in good)) if good else set()
        result[name]={'input_count':len(ins[name]['valid']),'run_count':len(runs),'successful_count':len(good),
          'runs':[{'name':label,'status':r['status'],'metrics':r.get('metrics')} for label,r in runs],
          'stable_accepted':len(intersection),'ever_accepted':len(union),'variable_sentence_count':len(union-intersection),
          'pairwise':pairs}
    output=DEST/'repeatability_summary.json'
    if output.exists():raise SystemExit('Summary exists')
    output.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'inputs':result},indent=2)+'\n',encoding='utf8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);data,ins=inputs()
        PLAN.write_text(json.dumps(plan_for(data,ins),indent=2)+'\n',encoding='utf8');print('Plan prepared; commit before inference.')
    elif cmd=='run':run(sys.argv[2],int(sys.argv[3]))
    elif cmd=='analyze':analyze()
    else:raise SystemExit('Unknown command')
