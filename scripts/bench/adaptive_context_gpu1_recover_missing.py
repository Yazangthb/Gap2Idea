"""Preserve a partial first-pass response and request only its missing IDs."""
import json,sys,time
import adaptive_context_gpu1_pilot as e
import five_paper_pilot as p
from baseline_evidence import digest

def recover(i,layout):
    g=e.frozen()[i-1];path=e.DEST/f'paper{i:02d}_{layout}_first.json';old=json.loads(path.read_text(encoding='utf8'))
    if old.get('status')!='failed' or old.get('plan_sha256')!=p.sha(e.PLAN):raise ValueError('Not a matching failed first pass')
    keys=g['keys'];expected={j for j,k in enumerate(keys) if k in g['fine']};a=old['attempts'][0]
    if a.get('finish_reason')!='stop':raise ValueError('No complete partial response')
    obj=json.loads(a['response']);ent=obj['results']
    if set(obj)!={'results'} or not isinstance(ent,list):raise ValueError('Bad schema')
    if any(not isinstance(z,dict) or set(z)!={'id','decision'} or type(z['id'])!=int or z['decision'] not in ('keep','drop','unsure') for z in ent):raise ValueError('Bad item')
    good=[z for z in ent if z['id'] in expected];ids=[z['id'] for z in good]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate input verdicts')
    missing=expected-set(ids)
    if not missing:raise ValueError('No missing IDs')
    selected={keys[j] for j in missing};user=e.fmt.render({**g,'fine':selected},layout);attempts=list(old['attempts']);c=p.client(); recovered=None
    for _ in range(2):
        t=time.time();x=None
        try:
            resp=c.chat.completions.create(model=e.base.MODEL,temperature=0,max_tokens=e.MAXOUT,messages=[{'role':'system','content':e.prompt('first')},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'decisions','schema':e.prior.schema(True),'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            x={'seconds':round(time.time()-t,2),'finish_reason':resp.choices[0].finish_reason,'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content,'recovery_missing_ids':sorted(missing)};attempts.append(x)
            if x['finish_reason']!='stop':raise ValueError('Truncated')
            z=json.loads(x['response']);entries=z['results']
            if set(z)!={'results'} or not isinstance(entries,list):raise ValueError('Bad schema')
            if any(not isinstance(v,dict) or set(v)!={'id','decision'} or type(v['id'])!=int or v['decision'] not in ('keep','drop','unsure') for v in entries):raise ValueError('Bad item')
            actual=[v for v in entries if v['id'] in missing];identifiers=[v['id'] for v in actual]
            if set(identifiers)!=missing or len(identifiers)!=len(set(identifiers)):raise ValueError('Missing or duplicate IDs')
            x['excluded_extra_entries']=[v for v in entries if v['id'] not in missing];recovered=actual;break
        except Exception as err:
            if x is None:attempts.append({'seconds':round(time.time()-t,2),'error_type':type(err).__name__})
            else:x['error_type']=type(err).__name__;x['error']=str(err)[:200]
    if recovered is None:raise ValueError('Recovery failed; original unchanged')
    source_hash=p.sha(path);failed=path.with_name(path.stem+'_failed.json')
    if failed.exists():raise ValueError('Recovery evidence exists')
    decisions=[{'key':list(keys[z['id']]),'decision':z['decision']} for z in good+recovered]
    result={'status':'ok','plan_sha256':p.sha(e.PLAN),'paper':i,'layout':layout,'stage':'first','selected_sha256':digest(sorted(g['fine'])),
      'input_sha256':digest(e.fmt.render(g,layout)),'decisions':decisions,'attempts':attempts,'recovered_missing_ids':sorted(missing),'failed_output_sha256':source_hash}
    path.rename(failed);path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print(json.dumps({'status':'ok','paper':i,'layout':layout,'missing_ids':sorted(missing),'recovered':recovered}),flush=True)

if __name__=='__main__':recover(int(sys.argv[1]),sys.argv[2])
