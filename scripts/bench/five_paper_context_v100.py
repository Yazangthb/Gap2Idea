"""Matched Qwen3.8-27B Q4 V100 comparison with and without candidate context.

User selected the V100 rather than the previous H100 FP8 service. The Q4 model
and serving stack differ, so all three comparison conditions are rerun here.
"""
import json
import subprocess
import sys
import time

import five_paper_context_pilot as q
import five_paper_dev_tuning as d
import five_paper_pilot as p
from baseline_evidence import digest

CONFIG=q.DEST/'v100_q4_plan.json'
MODEL='qwen38-v100:27b-q4'
DIGEST='15d962ff762f7d02c565fd78afb81eb514692119fce228b6c429cbbdd28a373e'


def plan():
    return {'purpose':'Same-model V100 Q4 matched comparison: no extra context vs 4B notes vs original neighboring text',
        'reason':'User requested 27B on the second V100; the installed 27B is Q4, unlike the earlier H100 FP8 service',
        'model':MODEL,'ollama_digest':DIGEST,'quantization':'Q4_K_M','underlying_hf_revision':'not independently verified',
        'data_plan_sha256':p.sha(q.PLAN),'notes_sha256':p.sha(q.NOTES),'recovery_sha256':p.sha(q.DEST/'4b_context_recovery.jsonl'),
        'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_context_v100.py'),
        'prompt_sha256':digest(d.system('neutral')),'conditions':['none','notes','neighbors'],
        'requests':'one per condition, same 258 candidates, same IDs/order/system/schema/decoding',
        'notes':'same 4B notes as prior stage; context additions are after each candidate and are not separate target sentences',
        'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192}


def main(name):
    if name not in ('none','notes','neighbors'):raise SystemExit('Unknown condition')
    data,items=q.frozen();q.check_tracked(q.NOTES);q.check_tracked(q.RAW);q.check_tracked(q.DEST/'4b_context_recovery.jsonl')
    q.check_tracked(CONFIG);q.check_tracked(p.ROOT/'scripts/bench/five_paper_context_v100.py')
    if json.loads(CONFIG.read_text(encoding='utf8'))!=plan():raise SystemExit('V100 plan changed')
    notes=json.loads(q.NOTES.read_text(encoding='utf8'))
    if notes['status']!='ok' or notes['plan_sha256']!=p.sha(q.PLAN):raise SystemExit('Incomplete notes')
    note_by_id=notes['notes'];want={str(x['id']) for x in items}
    if set(note_by_id)!=want:raise SystemExit('Note IDs mismatch')
    path=q.DEST/f'27b_v100_q4_{name}.json'
    if path.exists():raise SystemExit('Preserve existing evidence')
    mapping={x['id']:x for x in items};lines=[];last=None;keys=data['keys'];rows=data['rows']
    for i,r in enumerate(rows):
        if i not in mapping:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        x=mapping[i];line=f'[{i}] {r["sentence"]}'
        if name=='notes':line+='\nContext: '+note_by_id[str(i)]
        elif name=='neighbors':line+='\nContext: Previous: '+x['previous']+' Next: '+x['next']
        lines.append(line)
    user='\n'.join(lines);system=d.system('neutral');c=p.client()
    if MODEL not in {x.id for x in c.models.list().data}:raise SystemExit('V100 27B model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None
    for _ in range(2):
        start=time.time();item=None
        try:
            resp=c.chat.completions.create(model=MODEL,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-start,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives')
            if (set(obj)!={'positives'} or not isinstance(ids,list) or any(type(i) is not int or str(i) not in want for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','condition':name,'model':MODEL,'runtime_plan_sha256':p.sha(CONFIG),
                    'input_sha256':digest(user),'system_sha256':digest(system),'candidate_count':len(items),
                    'attempts':attempts,'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-start,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','condition':name,'runtime_plan_sha256':p.sha(CONFIG),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'condition':name,'status':result['status'],'metrics':result.get('metrics'),
                      'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok':raise SystemExit('Request failed; evidence preserved')


if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if CONFIG.exists():raise SystemExit('Plan exists')
        CONFIG.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        print('V100 plan prepared; commit before inference')
    elif sys.argv[1]=='run':main(sys.argv[2])
    else:raise SystemExit('Unknown command')
