"""Retry a failed call while preserving its original responses and costs.

Adds an explicit uniqueness reminder to the original system prompt. A retry is
accepted only if every input ID is present and repeated IDs have identical
booleans. This is a documented post-failure prompt deviation.
"""
import json,sys,time
import complete_split_batch_comparison as e
import five_paper_pilot as p
from baseline_evidence import digest

REMINDER='\n\nFormatting reminder: Return each input ID exactly once. Never repeat an ID, including with a different decision.'

def recover(split,batch,index,method):
    _,groups=e.frozen();x,g=next((x,g) for x,g in groups if x['split']==split and x['batch']==batch and x['index']==index)
    target=e.result_path(x,method);original=json.loads(target.read_text(encoding='utf8'))
    if original.get('status')!='failed' or original.get('plan_sha256')!=p.sha(e.PLAN):raise ValueError('Not a frozen failed run')
    failed=target.with_name(target.stem+'_failed.json')
    if failed.exists():raise ValueError('Failed evidence already exists')
    oldsha=p.sha(target);valid,user=e.input_for(g,method);keys=g['keys'];input_ids={i for i,k in enumerate(keys) if k in valid}
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    attempts=list(original['attempts']);out=None;c=p.client()
    for _ in range(2):
        start=time.time();a=None
        try:
            resp=c.chat.completions.create(model=e.base.MODEL,temperature=0,max_tokens=e.MAXOUT,messages=[{'role':'system','content':e.system(len(g['paper_ids']))+REMINDER},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'verdicts','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            a={'seconds':round(time.time()-start,2),'finish_reason':resp.choices[0].finish_reason,'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content,'recovery_reminder':REMINDER};attempts.append(a)
            if a['finish_reason']!='stop':raise ValueError('Truncated')
            obj=json.loads(a['response']);ent=obj['results']
            if set(obj)!={'results'} or not isinstance(ent,list) or any(not isinstance(z,dict) or set(z)!={'id','keep'} or type(z['id'])!=int or type(z['keep'])!=bool for z in ent):raise ValueError('Bad schema')
            good=[z for z in ent if z['id'] in input_ids];extra=[z for z in ent if z['id'] not in input_ids];ids=[z['id'] for z in good]
            if set(ids)!=input_ids:raise ValueError('Missing input ID')
            by_id={}
            for z in good:by_id.setdefault(z['id'],[]).append(z['keep'])
            if any(len(set(v))!=1 for v in by_id.values()):raise ValueError('Conflicting input ID')
            a['duplicate_input_ids']={str(k):len(v) for k,v in by_id.items() if len(v)>1}
            a['excluded_extra_entries']=extra;kept={keys[z['id']] for z in good if z['keep']}
            out={'status':'ok','plan_sha256':p.sha(e.PLAN),'split':split,'batch':batch,'index':index,'method':method,'input_sha256':digest(user),'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept),'attempts':attempts,'recovered_with_uniqueness_reminder':True,'failed_output_sha256':oldsha,'prior_recovery_attempts_not_saved':2 if (split,batch,index,method)==('test','one',11,'fine_no_context') else 0};break
        except Exception as err:
            if a is None:attempts.append({'seconds':round(time.time()-start,2),'error_type':type(err).__name__,'recovery_reminder':REMINDER})
            else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
    if out is None:
        print(json.dumps({'status':'failed','retry_attempts':attempts[len(original['attempts']):]}),flush=True);raise SystemExit('Recovery failed; original unchanged')
    target.rename(failed);target.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'status':'ok','split':split,'batch':batch,'index':index,'method':method,'metrics':out['metrics']}),flush=True)

if __name__=='__main__':recover(sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4])
