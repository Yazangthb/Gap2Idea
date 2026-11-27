"""Recover missing context-flag target decisions, preserving partial responses."""
import json,sys,time
import adaptive_context_flag_gpu0_development as e
import five_paper_pilot as p
from baseline_evidence import digest

def recover(i,layout):
    g=e.frozen()[i-1];path=e.DEST/f'paper{i:02d}_{layout}_first.json';old=json.loads(path.read_text(encoding='utf8'))
    if old.get('status')!='failed' or old.get('plan_sha256')!=p.sha(e.PLAN):raise ValueError('Not matching failed output')
    keys=g['keys'];expected={j for j,k in enumerate(keys) if k in g['fine']};a=old['attempts'][0]
    if a.get('finish_reason')!='stop':raise ValueError('Partial response unavailable')
    obj=json.loads(a['response']);ent=obj['results'];fields={'id','keep','needs_more_context'}
    if set(obj)!={'results'} or not isinstance(ent,list) or any(not isinstance(z,dict) or set(z)!=fields or type(z['id'])!=int or type(z['keep'])!=bool or type(z['needs_more_context'])!=bool for z in ent):raise ValueError('Invalid partial response')
    good=[z for z in ent if z['id'] in expected];ids=[z['id'] for z in good]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate IDs')
    missing=expected-set(ids)
    if not missing:raise ValueError('Nothing missing')
    selected={keys[j] for j in missing};user=e.prior.fmt.render({**g,'fine':selected},layout);attempts=list(old['attempts']);c=p.client();recovered=None
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep','needs_more_context'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'},'needs_more_context':{'type':'boolean'}}}}}}
    for _ in range(2):
        t=time.time();x=None
        try:
            r=c.chat.completions.create(model=e.prior.base.MODEL,temperature=0,max_tokens=e.MAXOUT,messages=[{'role':'system','content':e.probe.prompt()},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'flag','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            x={'seconds':round(time.time()-t,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content,'recovery_missing_ids':sorted(missing)};attempts.append(x)
            if x['finish_reason']!='stop':raise ValueError('Truncated')
            z=json.loads(x['response']);items=z['results']
            if set(z)!={'results'} or not isinstance(items,list) or any(not isinstance(v,dict) or set(v)!=fields or type(v['id'])!=int or type(v['keep'])!=bool or type(v['needs_more_context'])!=bool for v in items):raise ValueError('Bad schema')
            actual=[v for v in items if v['id'] in missing];outids=[v['id'] for v in actual]
            if set(outids)!=missing or len(outids)!=len(set(outids)):raise ValueError('Missing or duplicate IDs')
            x['excluded_extra_entries']=[v for v in items if v['id'] not in missing];recovered=actual;break
        except Exception as err:
            if x is None:attempts.append({'seconds':round(time.time()-t,2),'error_type':type(err).__name__})
            else:x['error_type']=type(err).__name__;x['error']=str(err)[:200]
    if recovered is None:raise ValueError('Recovery failed; original unchanged')
    oldsha=p.sha(path);failed=path.with_name(path.stem+'_failed.json')
    if failed.exists():raise ValueError('Failed evidence exists')
    out={'status':'ok','plan_sha256':p.sha(e.PLAN),'paper':i,'layout':layout,'stage':'first','selected_sha256':digest(sorted(g['fine'])),
      'input_sha256':digest(e.prior.fmt.render(g,layout)),'decisions':[{'key':list(keys[z['id']]),'keep':z['keep'],'needs_more_context':z['needs_more_context']} for z in good+recovered],
      'attempts':attempts,'recovered_missing_ids':sorted(missing),'failed_output_sha256':oldsha}
    path.rename(failed);path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'paper':i,'layout':layout,'missing':sorted(missing),'recovered':recovered}),flush=True)

if __name__=='__main__':recover(int(sys.argv[1]),sys.argv[2])
