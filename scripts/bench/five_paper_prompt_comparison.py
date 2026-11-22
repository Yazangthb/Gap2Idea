"""4B pilot using the saved neutral and keep-biased Stage C rubrics."""
import json
import sys
import time

import five_paper_pilot as p
from baseline_evidence import digest
from stage_c_rubric_val import RUBRICS


def main(name):
    p.require_frozen()
    if name not in ('neutral', 'keep_biased'):
        raise SystemExit('Choose neutral or keep_biased')
    rows, _, chosen = p.data()
    out = p.DEST / f'4b_{name}_batches.jsonl'
    summary = p.DEST / f'4b_{name}_summary.json'
    if out.exists() or summary.exists():
        raise SystemExit('Existing evidence must be preserved')
    client = p.client(); model = 'qwen3.5-4b-bf16:latest'
    if model not in {x.id for x in client.models.list().data}:
        raise SystemExit('Expected model unavailable')
    schema = {'type':'object','additionalProperties':False,'required':['results'],
              'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,
              'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    kept = set()
    with out.open('x',encoding='utf8') as fh:
        for start in range(0,len(chosen),20):
            batch=chosen[start:start+20]
            items=[{'i':start+j,'type':r['gap_type'],'context':r['paragraph_text'][:600],
                    'sentence':r['gap_sentence'][:400]} for j,r in enumerate(batch)]
            want={i['i'] for i in items}; attempts=[]; verdict=None
            for _ in range(2):
                t=time.time(); item=None
                try:
                    resp=client.chat.completions.create(model=model,temperature=0,max_tokens=256+400*len(items),
                        messages=[{'role':'system','content':RUBRICS[name]},
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
                    error={'error_type':type(exc).__name__,'status_code':getattr(exc,'status_code',None)}
                    if item is None: attempts.append({'seconds':round(time.time()-t,3), **error})
                    else: item['validation_error']=error
            record={'start':start,'keys':[[r['id'],r['idx']] for r in batch], 'attempts':attempts,
                    'status':'ok' if verdict is not None else 'failed','verdicts':verdict}
            fh.write(json.dumps(record,ensure_ascii=False)+'\n'); fh.flush()
            if verdict is None: raise SystemExit(f'Batch {start} failed; evidence preserved')
            kept.update((chosen[r['id']]['id'],chosen[r['id']]['idx']) for r in verdict if r['keep'])
            print(f'{name}: {min(start+20,len(chosen))}/{len(chosen)}',flush=True)
    result={'status':'ok','model':model,'prompt':name,'prompt_sha256':digest(RUBRICS[name]),
            'plan_sha256':p.sha(p.PLAN),'candidate_count':len(chosen),
            'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)}
    summary.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'prompt':name,'metrics':result['metrics']}),flush=True)


if __name__=='__main__':
    main(sys.argv[1])
