"""Exploratory high-recall 4B shortlisting followed by neutral 27B filtering.

Run `4b {screen,permissive}` first, commit its output, then run `27b NAME`.
The prompts are defined before inference. No evaluation labels are sent.
"""
import json
import subprocess
import sys
import time

import five_paper_pilot as p
from baseline_evidence import digest
from stage_c_rubric_val import RUBRICS

FORMAT = '''Each item has i, predicted type, sentence and context. Judge the sentence;
use context only to clarify it. Return exactly one item for each input in JSON:
{"results":[{"id": <i>, "keep": true or false}, ...]}.'''
PROMPTS = {
    'screen': '''You are a high-recall first-pass screener. A second model will check
your selections, so missing a real research gap is more costly than including a
possible one. Keep a sentence if it may describe a limitation, weakness, failure,
unmet need or open problem of this work or prior work, or proposes future work.
Keep uncertain, indirect, qualified, or context-dependent cases for review.
Drop only sentences clearly unrelated to a limitation, open problem or future
research direction, such as an unqualified result, title or acknowledgement.
For example: "Existing models struggle with rare cases" -> keep;
"We evaluate our model on three datasets" -> drop;
"The approach has not been tested in hospitals" -> keep.
''' + FORMAT,
    'permissive': '''Select candidate research gaps for a careful later review.
Mark keep=true if the sentence could express a weakness, challenge, shortcoming,
failure, unresolved question, missing capability, constraint, caveat, or future
investigation. These can refer to this paper, existing methods, or the field.
Include statements about difficulty, poor performance, missing validation, or
limited applicability even if phrased as background or a motivation. If unsure,
keep the sentence. Reject only sentences clearly unrelated to any such issue.
Do not require the words limitation or future work. The later reviewer will remove
false positives. Do not use a candidate's predicted type as proof of a gap.
''' + FORMAT,
}


def check_tracked(path):
    rel=path.relative_to(p.ROOT).as_posix()
    if (subprocess.run(['git','status','--porcelain','--',rel],cwd=p.ROOT,capture_output=True,text=True).stdout.strip()
        or subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=p.ROOT,capture_output=True).returncode):
        raise SystemExit('Commit the exact runner and source shortlist before inference')


def main(stage, name):
    if stage not in ('4b','27b') or name not in PROMPTS: raise SystemExit('Invalid configuration')
    p.require_frozen(); check_tracked(p.ROOT/'scripts/bench/five_paper_recall_cascade.py')
    rows,_,chosen=p.data(); source=None
    if stage=='27b':
        source=p.DEST/f'4b_recall_{name}_summary.json'; check_tracked(source)
        s=json.loads(source.read_text(encoding='utf8'))
        if s['status']!='ok' or s['plan_sha256']!=p.sha(p.PLAN): raise SystemExit('Source is incomplete')
        keys={tuple(x) for x in s['kept']}; chosen=[r for r in chosen if (r['id'],r['idx']) in keys]
        if len(chosen)!=len(keys): raise SystemExit('Shortlist mismatch')
    model='qwen3.5-4b-bf16:latest' if stage=='4b' else 'qwen3.8-27b'
    prompt=PROMPTS[name] if stage=='4b' else RUBRICS['neutral']
    stem=f'4b_recall_{name}' if stage=='4b' else f'cascade_4b_recall_{name}_27b_neutral'
    out=p.DEST/f'{stem}_batches.jsonl'; summary=p.DEST/f'{stem}_summary.json'
    if out.exists() or summary.exists(): raise SystemExit('Preserve existing evidence')
    c=p.client()
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
            print(f'{stage} {name}: {min(start+20,len(chosen))}/{len(chosen)}',flush=True)
    result={'status':'ok','stage':stage,'prompt_name':name,'model':model,'prompt_sha256':digest(prompt),
            'plan_sha256':p.sha(p.PLAN),'source_sha256':p.sha(source) if source else None,
            'candidate_count':len(chosen),'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)}
    summary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'stage':stage,'prompt':name,'metrics':result['metrics']}),flush=True)


if __name__=='__main__': main(sys.argv[1],sys.argv[2])
