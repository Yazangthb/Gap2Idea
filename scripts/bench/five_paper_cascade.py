"""Explore 4B Stage C shortlists followed by 27B Stage C on five dev papers.

Usage: five_paper_cascade.py {strict,keep_biased} {neutral,keep_biased}
Input shortlists and runner must be committed before inference. Failed calls are
preserved and never treated as negative predictions.
"""
import json
import subprocess
import sys
import time

import five_paper_pilot as p
from baseline_evidence import digest
from stage_c_rubric_val import RUBRICS


def main(small, large):
    if small not in ('strict','keep_biased') or large not in ('neutral','keep_biased'):
        raise SystemExit('Unsupported cascade settings')
    p.require_frozen()
    rows, _, chosen = p.data()
    source = p.DEST / ('4b_funnel_summary.json' if small == 'strict' else '4b_keep_biased_summary.json')
    for path in (source, p.ROOT / 'scripts/bench/five_paper_cascade.py'):
        rel = path.relative_to(p.ROOT).as_posix()
        if subprocess.run(['git','status','--porcelain','--',rel],cwd=p.ROOT,capture_output=True,text=True).stdout.strip():
            raise SystemExit('Commit the runner and source before inference')
        if subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=p.ROOT,capture_output=True).returncode:
            raise SystemExit('Runner or source is not tracked in HEAD')
    source_data=json.loads(source.read_text(encoding='utf8'))
    if source_data['status']!='ok' or source_data['plan_sha256']!=p.sha(p.PLAN):
        raise SystemExit('The source shortlist is not a complete matching pilot')
    source_keys={tuple(x) for x in source_data['kept']}
    selected=[r for r in chosen if (r['id'],r['idx']) in source_keys]
    if len(selected)!=len(source_keys): raise SystemExit('Shortlist keys do not match candidates')
    out=p.DEST/f'cascade_4b_{small}_27b_{large}_batches.jsonl'
    summary=p.DEST/f'cascade_4b_{small}_27b_{large}_summary.json'
    if out.exists() or summary.exists(): raise SystemExit('Preserve existing output')
    c=p.client(); model='qwen3.8-27b'
    if model not in {x.id for x in c.models.list().data}: raise SystemExit('Expected model not available')
    schema={'type':'object','additionalProperties':False,'required':['results'],
            'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,
            'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    kept=set()
    with out.open('x',encoding='utf8') as fh:
        for start in range(0,len(selected),20):
            batch=selected[start:start+20]
            items=[{'i':start+j,'type':r['gap_type'],'context':r['paragraph_text'][:600],
                    'sentence':r['gap_sentence'][:400]} for j,r in enumerate(batch)]
            want={i['i'] for i in items}; attempts=[]; verdict=None
            for _ in range(2):
                t=time.time(); item=None
                try:
                    resp=c.chat.completions.create(model=model,temperature=0,max_tokens=256+400*len(items),
                        messages=[{'role':'system','content':RUBRICS[large]},
                                  {'role':'user','content':json.dumps(items,ensure_ascii=False)}],
                        response_format={'type':'json_schema','json_schema':{'name':'gap_verdicts','schema':schema,'strict':True}},
                        extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                    item={'seconds':round(time.time()-t,3),'finish_reason':resp.choices[0].finish_reason,
                          'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content}
                    attempts.append(item)
                    if item['finish_reason']!='stop': continue
                    parsed=json.loads(item['response']); result=parsed.get('results',[])
                    if (set(parsed)!={'results'} or len(result)!=len(want) or
                        any(set(r)!={'id','keep'} or type(r['id']) is not int or type(r['keep']) is not bool for r in result) or
                        {r['id'] for r in result}!=want):
                        item['validation_error']='incomplete, duplicate or invalid results'; continue
                    verdict=result; break
                except Exception as exc:
                    err={'error_type':type(exc).__name__,'status_code':getattr(exc,'status_code',None)}
                    if item is None: attempts.append({'seconds':round(time.time()-t,3),**err})
                    else: item['validation_error']=err
            rec={'start':start,'keys':[[r['id'],r['idx']] for r in batch],'attempts':attempts,
                 'status':'ok' if verdict is not None else 'failed','verdicts':verdict}
            fh.write(json.dumps(rec,ensure_ascii=False)+'\n'); fh.flush()
            if verdict is None: raise SystemExit('Failed batch; raw evidence retained')
            kept.update((selected[r['id']]['id'],selected[r['id']]['idx']) for r in verdict if r['keep'])
            print(f'{small} -> {large}: {min(start+20,len(selected))}/{len(selected)}',flush=True)
    result={'status':'ok','small_prompt':small,'large_prompt':large,'model':model,
            'plan_sha256':p.sha(p.PLAN),'source_shortlist_sha256':p.sha(source),'large_prompt_sha256':digest(RUBRICS[large]),
            'candidate_count':len(selected),'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)}
    summary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'cascade':small+' -> '+large,'metrics':result['metrics']}),flush=True)


if __name__=='__main__':
    main(sys.argv[1],sys.argv[2])
