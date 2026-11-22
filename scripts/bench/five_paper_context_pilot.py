"""4B context notes vs verbatim neighbors for the matched one-call 27B funnel.

Prepare and commit plan before inference. Generate notes, commit them, then run
27B on both note and verbatim context conditions. No gold labels enter requests.
"""
import json
import subprocess
import sys
import time

import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/five_paper_context_pilot_2026-11-22_rev1'
PLAN=DEST/'plan.json'; NOTES=DEST/'4b_context_notes.json'; RAW=DEST/'4b_context_batches.jsonl'
NOTE_PROMPT=('Explain what each target sentence refers to using its neighboring sentences. '
             'Write one factual, concise context note of at most 20 words per item; '
             'do not decide whether it is a research gap, add unsupported claims, or omit any item. '
             'Return JSON {"results":[{"id": integer, "context": string}, ...]}.')
CAP=800


def source():
    data=d.make_data()['check']; rows=data['rows']; keys=data['keys']; selected=data['candidate_keys']; items=[]
    for i,r in enumerate(rows):
        if keys[i] not in selected: continue
        prev=rows[i-1]['sentence'][-CAP:] if i and rows[i-1]['pid']==r['pid'] else ''
        nxt=rows[i+1]['sentence'][:CAP] if i+1<len(rows) and rows[i+1]['pid']==r['pid'] else ''
        items.append({'id':i,'previous':prev,'sentence':r['sentence'],'next':nxt})
    if len(items)!=len(selected): raise ValueError('Candidate keys mismatch')
    return data,items


def plan_for(data,items):
    baseline=d.DEST/'check_neutral_funnel.json'; selection=d.SELECTION
    return {'purpose':'Evaluate 4B factual context notes and verbatim neighboring text for identical one-call funnel candidates',
      'split':'previous five-paper development check; exploratory, not independent',
      'model_4b':'qwen3.5-4b-bf16:latest','model_27b':'qwen3.8-27b',
      'model_4b_ollama_digest':'23f8c64d937d3062759377d28559235f278fe675fa782da87699e583ddfb40e0',
      'note_prompt':NOTE_PROMPT,'note_prompt_sha256':digest(NOTE_PROMPT),'qwen_prompt':d.system('neutral'),
      'qwen_prompt_sha256':digest(d.system('neutral')),'candidates':len(items),
      'neighbor_rule':'previous original sentence last 800 characters and next original sentence first 800 characters, within paper; no ground truth sent',
      'note_batch_size':20,'temperature':0,'seed':0,'thinking':False,
      'revision_note':'Rev1 follows an initial batch that returned 19 of 20 notes; retain valid partial answers and retry missing IDs, preserving the original failure.',
      'qwen_methods':'baseline no local context vs 4B-generated note vs verbatim previous/next; one 27B request per condition',
      'input_sha256':digest(items),'baseline_result_sha256':p.sha(baseline),'dev_plan_sha256':p.sha(d.PLAN),
      'selection_sha256':p.sha(selection),'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_context_pilot.py')}


def check_tracked(path):
    rel=path.relative_to(p.ROOT).as_posix()
    if (subprocess.run(['git','status','--porcelain','--',rel],cwd=p.ROOT,capture_output=True,text=True).stdout.strip()
        or subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=p.ROOT,capture_output=True).returncode):
        raise SystemExit(f'Commit {rel} first')


def frozen():
    d.frozen();check_tracked(d.SELECTION);check_tracked(d.DEST/'check_neutral_funnel.json')
    data,items=source()
    if not PLAN.exists() or json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(data,items): raise SystemExit('Plan or data changed')
    check_tracked(PLAN);check_tracked(p.ROOT/'scripts/bench/five_paper_context_pilot.py')
    return data,items


def notes():
    data,items=frozen()
    if RAW.exists() or NOTES.exists(): raise SystemExit('Preserve existing evidence')
    c=p.client();model='qwen3.5-4b-bf16:latest'
    if model not in {x.id for x in c.models.list().data}: raise SystemExit('4B unavailable')
    schema={'type':'object','additionalProperties':False,'required':['results'],
            'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,
             'required':['id','context'],'properties':{'id':{'type':'integer'},'context':{'type':'string'}}}}}}
    notes_by_id={}
    with RAW.open('x',encoding='utf8') as fh:
        for start in range(0,len(items),20):
            batch=items[start:start+20];want={x['id'] for x in batch};attempts=[];verdict=None; collected={}
            for _ in range(4):
                remaining=[x for x in batch if x['id'] not in collected];pending={x['id'] for x in remaining}
                t=time.time();item=None
                try:
                    resp=c.chat.completions.create(model=model,temperature=0,max_tokens=4096,
                        messages=[{'role':'system','content':NOTE_PROMPT},{'role':'user','content':json.dumps(remaining,ensure_ascii=False)}],
                        response_format={'type':'json_schema','json_schema':{'name':'contexts','schema':schema,'strict':True}},
                        extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                    item={'seconds':round(time.time()-t,3),'finish_reason':resp.choices[0].finish_reason,
                          'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
                    if item['finish_reason']!='stop':continue
                    obj=json.loads(item['response']);result=obj.get('results',[])
                    if (set(obj)!={'results'} or not isinstance(result,list) or
                        any(set(r)!={'id','context'} or type(r['id']) is not int or not isinstance(r['context'],str)
                            or not r['context'].strip() or len(r['context'])>300 or '\n' in r['context'] for r in result) or
                        len({r['id'] for r in result})!=len(result) or not {r['id'] for r in result}<=pending):
                        item['validation_error']='duplicate, unexpected or invalid context notes';continue
                    collected.update({r['id']:r for r in result})
                    item['accepted_count']=len(result);item['missing_count']=len(want-set(collected))
                    if len(collected)==len(want):verdict=[collected[i] for i in sorted(want)];break
                except Exception as e:
                    err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
                    if item is None:attempts.append({'seconds':round(time.time()-t,3),**err})
                    else:item['validation_error']=err
            record={'start':start,'ids':sorted(want),'status':'ok' if verdict is not None else 'failed',
                    'attempts':attempts,'verdicts':verdict}
            fh.write(json.dumps(record,ensure_ascii=False)+'\n');fh.flush()
            if verdict is None:raise SystemExit('Failed batch; evidence preserved')
            notes_by_id.update({str(r['id']):r['context'].strip() for r in verdict})
            print(f'4B context: {min(start+20,len(items))}/{len(items)}',flush=True)
    out={'status':'ok','plan_sha256':p.sha(PLAN),'notes':notes_by_id}
    NOTES.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')


def run(name):
    if name not in ('notes','neighbors'):raise SystemExit('Unknown condition')
    data,items=frozen();check_tracked(NOTES);check_tracked(RAW)
    note_data=json.loads(NOTES.read_text(encoding='utf8'))
    if note_data['status']!='ok' or note_data['plan_sha256']!=p.sha(PLAN):raise SystemExit('Notes incomplete')
    notes_by_id=note_data['notes'];want={str(x['id']) for x in items}
    if set(notes_by_id)!=want:raise SystemExit('Note IDs mismatch')
    path=DEST/f'27b_{name}.json'
    if path.exists():raise SystemExit('Preserve existing evidence')
    keys=data['keys'];rows=data['rows'];mapping={x['id']:x for x in items};lines=[];last=None
    for i,r in enumerate(rows):
        if i not in mapping:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        x=mapping[i]
        context=notes_by_id[str(i)] if name=='notes' else ('Previous: '+x['previous']+' Next: '+x['next'])
        lines.append(f'[{i}] {r["sentence"]}\nContext: {context}')
    user='\n'.join(lines);system=d.system('neutral');c=p.client();model='qwen3.8-27b'
    if model not in {x.id for x in c.models.list().data}:raise SystemExit('27B unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None
    for _ in range(2):
        t=time.time();item=None
        try:
            resp=c.chat.completions.create(model=model,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-t,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives')
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or str(i) not in want for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','method':name,'plan_sha256':p.sha(PLAN),'notes_sha256':p.sha(NOTES),
                    'input_sha256':digest(user),'system_sha256':digest(system),'candidate_count':len(items),
                    'attempts':attempts,'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(rows,kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-t,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','method':name,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'method':name,'status':result['status'],'metrics':result.get('metrics'),
                      'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok':raise SystemExit('Request failed; evidence preserved')


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);data,items=source()
        PLAN.write_text(json.dumps(plan_for(data,items),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        print('Plan prepared; commit before inference')
    elif cmd=='notes':notes()
    elif cmd=='run':run(sys.argv[2])
    else:raise SystemExit('Unknown command')
