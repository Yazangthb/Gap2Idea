"""Recover incomplete 4B context notes without guessing repeated-ID output."""
import collections
import json
import time

import five_paper_context_pilot as q
import five_paper_pilot as p


def main():
    _,items=q.frozen(); q.check_tracked(q.RAW)
    q.check_tracked(p.ROOT/'scripts/bench/five_paper_context_recover.py')
    recover=q.DEST/'4b_context_recovery.jsonl'
    if recover.exists() or q.NOTES.exists(): raise SystemExit('Preserve existing output')
    records=[json.loads(x) for x in q.RAW.read_text(encoding='utf8').splitlines()]
    by_id={}; ambiguous=set()
    for r in records:
        if r['status']=='ok':
            by_id.update({x['id']:x['context'].strip() for x in r['verdicts']})
        else:
            # Recover only IDs that occur exactly once in the first response.
            results=json.loads(r['attempts'][0]['response'])['results']; count=collections.Counter(x['id'] for x in results)
            for x in results:
                if (x['id'] in r['ids'] and count[x['id']]==1 and isinstance(x.get('context'),str)
                    and x['context'].strip() and len(x['context'])<=300 and '\n' not in x['context']):
                    by_id[x['id']]=x['context'].strip()
                elif x['id'] in r['ids'] and count[x['id']]>1: ambiguous.add(x['id'])
    remaining=[x for x in items if x['id'] not in by_id]
    groups=[[x] for x in remaining if x['id'] in ambiguous]
    other=[x for x in remaining if x['id'] not in ambiguous]
    groups += [other[i:i+10] for i in range(0,len(other),10)]
    model='qwen3.5-4b-bf16:latest';c=p.client()
    if model not in {x.id for x in c.models.list().data}:raise SystemExit('4B unavailable')
    schema={'type':'object','additionalProperties':False,'required':['results'],
            'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,
             'required':['id','context'],'properties':{'id':{'type':'integer'},'context':{'type':'string'}}}}}}
    with recover.open('x',encoding='utf8') as fh:
        for group in groups:
            want={x['id'] for x in group}; collected={};attempts=[]
            for _ in range(4):
                pending=[x for x in group if x['id'] not in collected]; ids={x['id'] for x in pending}; t=time.time();item=None
                try:
                    resp=c.chat.completions.create(model=model,temperature=0,max_tokens=4096,
                        messages=[{'role':'system','content':q.NOTE_PROMPT},{'role':'user','content':json.dumps(pending,ensure_ascii=False)}],
                        response_format={'type':'json_schema','json_schema':{'name':'contexts','schema':schema,'strict':True}},
                        extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                    item={'seconds':round(time.time()-t,3),'finish_reason':resp.choices[0].finish_reason,
                          'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content}; attempts.append(item)
                    if item['finish_reason']!='stop':continue
                    obj=json.loads(item['response']);rs=obj.get('results',[]);freq=collections.Counter(x.get('id') for x in rs)
                    valid=[x for x in rs if set(x)=={'id','context'} and type(x['id']) is int and x['id'] in ids
                           and freq[x['id']]==1 and isinstance(x['context'],str) and x['context'].strip()
                           and len(x['context'])<=300 and '\n' not in x['context']]
                    collected.update({x['id']:x['context'].strip() for x in valid})
                    item['accepted_count']=len(valid);item['missing_count']=len(want-set(collected))
                    if set(collected)==want:break
                except Exception as e:
                    if item is None:attempts.append({'seconds':round(time.time()-t,3),'error_type':type(e).__name__})
                    else:item['validation_error']=type(e).__name__
            ok=set(collected)==want
            fh.write(json.dumps({'ids':sorted(want),'status':'ok' if ok else 'failed','attempts':attempts,
                                 'notes':collected},ensure_ascii=False)+'\n');fh.flush()
            if not ok:raise SystemExit('Recovery incomplete; evidence preserved')
            by_id.update(collected);print(f'recovery: {len(by_id)}/{len(items)}',flush=True)
    if set(by_id)!={x['id'] for x in items}:raise SystemExit('Incomplete final notes')
    q.NOTES.write_text(json.dumps({'status':'ok','plan_sha256':p.sha(q.PLAN),
        'recovery_script':'scripts/bench/five_paper_context_recover.py',
        'notes':{str(k):v for k,v in sorted(by_id.items())}},indent=2,ensure_ascii=False)+'\n',encoding='utf8')


if __name__=='__main__':main()
