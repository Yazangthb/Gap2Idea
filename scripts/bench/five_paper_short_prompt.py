"""Matched 27B comparison: one-sentence prompt vs existing neutral rubric."""
import json
import subprocess
import sys
import time

import five_paper_pilot as p
from baseline_evidence import digest
from stage_c_rubric_val import RUBRICS

SHORT = ('For each item, set keep to true if the sentence states a research limitation, '
         'a future-work direction, or an open problem in the field or prior work, '
         'and return exactly one JSON result with its id and keep value.')


def main(name):
    if name not in ('short','neutral'): raise SystemExit('Invalid prompt')
    p.require_frozen()
    rel='scripts/bench/five_paper_short_prompt.py'
    if (subprocess.run(['git','status','--porcelain','--',rel],cwd=p.ROOT,capture_output=True,text=True).stdout.strip()
        or subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=p.ROOT,capture_output=True).returncode):
        raise SystemExit('Commit the exact runner before inference')
    rows,_,chosen=p.data(); prompt=SHORT if name=='short' else RUBRICS['neutral']
    out=p.DEST/f'27b_matched_{name}_batches.jsonl'; summary=p.DEST/f'27b_matched_{name}_summary.json'
    if out.exists() or summary.exists(): raise SystemExit('Preserve existing evidence')
    c=p.client(); model='qwen3.8-27b'
    if model not in {x.id for x in c.models.list().data}: raise SystemExit('Expected model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['results'],
            'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,
            'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    kept=set()
    with out.open('x',encoding='utf8') as fh:
        for start in range(0,len(chosen),20):
            batch=chosen[start:start+20]
            items=[{'i':start+j,'type':r['gap_type'],'context':r['paragraph_text'][:600],
                    'sentence':r['gap_sentence'][:400]} for j,r in enumerate(batch)]
            want={i['i'] for i in items}; attempts=[]; verdict=None
            for _ in range(2):
                t=time.time(); item=None
                try:
                    resp=c.chat.completions.create(model=model,temperature=0,max_tokens=256+400*len(items),
                        messages=[{'role':'system','content':prompt},{'role':'user','content':json.dumps(items,ensure_ascii=False)}],
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
            if verdict is None: raise SystemExit('Failed batch; evidence preserved')
            kept.update((chosen[r['id']]['id'],chosen[r['id']]['idx']) for r in verdict if r['keep'])
            print(f'{name}: {min(start+20,len(chosen))}/{len(chosen)}',flush=True)
    result={'status':'ok','prompt':name,'prompt_text':prompt,'model':model,'prompt_sha256':digest(prompt),
            'plan_sha256':p.sha(p.PLAN),'candidate_count':len(chosen),
            'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)}
    summary.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'prompt':name,'metrics':result['metrics']}),flush=True)


if __name__=='__main__': main(sys.argv[1])
