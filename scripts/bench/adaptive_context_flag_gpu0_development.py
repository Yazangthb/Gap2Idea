"""Ten-paper adaptive context test using separate context-needed flags on GPU 0."""
import concurrent.futures,json,os,sys,time,urllib.request,threading
import adaptive_context_gpu1_pilot as prior
import adaptive_context_flag_gpu0_probe as probe
import complete_split_batch_comparison as complete
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/adaptive_context_flag_gpu0_development_2026-11-27'
PLAN=DEST/'plan.json';PRE=DEST/'preflight.json';RUNNER=p.ROOT/'scripts/bench/adaptive_context_flag_gpu0_development.py'
LAYOUTS=('no_context','surrounding');MAXOUT=8192;LOCK=threading.Lock()

def data():return prior.frozen()

def baseline_source(g,layout,index):
    if index>9:return None
    q=json.loads(complete.PLAN.read_text(encoding='utf8'))
    x=next(x for x in q['groups'] if x['split']=='val' and x['batch']=='one' and x['paper_ids']==g['paper_ids'])
    method='fine_no_context' if layout=='no_context' else 'fine_surrounding';path=complete.result_path(x,method)
    r=json.loads(path.read_text(encoding='utf8'))
    if r.get('status')!='ok' or r.get('plan_sha256')!=p.sha(complete.PLAN) or r.get('input_sha256')!=digest(prior.fmt.render(g,layout)) or complete.system(1)!=prior.prompt('baseline'):raise ValueError('Unmatched baseline')
    return path

def probe_source(i,layout):return probe.DEST/f'paper{i:02d}_{layout}.json' if i<=2 else None

def plan():
    gs=data();sources={}
    for i,g in enumerate(gs,1):
        for layout in LAYOUTS:
            b=baseline_source(g,layout,i);q=probe_source(i,layout)
            if b:sources[f'baseline_{i}_{layout}']={'path':b.relative_to(p.ROOT).as_posix(),'sha256':p.sha(b)}
            if q:
                r=json.loads(q.read_text(encoding='utf8'))
                if r.get('status')!='ok' or r.get('plan_sha256')!=p.sha(probe.PLAN):raise ValueError('Bad probe')
                sources[f'first_{i}_{layout}']={'path':q.relative_to(p.ROOT).as_posix(),'sha256':p.sha(q)}
    return {'purpose':'Test a separate context-needed boolean, routing only flagged candidates to a wider-context second call, with and without first-pass neighbors, on 10 development papers.',
      'selection':'Same ten arXiv development papers as the previous pilot. Nine papers each use one request; the last is divided into two matched candidate groups. 682 original fine-tuned NLI candidates.',
      'first_prompt':probe.prompt(),'baseline_prompt':prior.prompt('baseline'),'followup_prompt':prior.prompt('followup'),
      'first_layouts':list(LAYOUTS),'followup':'Only flagged numbered targets; up to 3 previous and 3 following original sentences, each <=800 characters. Second-pass boolean overrides first-pass provisional keep.',
      'reuse':'Two-paper flag probe responses reused after matching input and prompt; saved single-paper GPU0 one-pass baselines reused for first nine papers after exact input and prompt verification. Last paper split into two groups so baselines rerun.',
      'note':'Flag scheme was proposed after tri-state uncertainty produced zero unsure and after seeing a two-paper probe with zero flags; results are exploratory and those papers are not a holdout. No new test data.',
      'labels':'LLM-generated, previously studied, not human adjudicated. Qwen3.8-27B-FP8 GPU0, vLLM 0.32.0, same revision and deterministic settings as GPU0 baselines. GPU1 service not used.',
      'max_output_tokens':MAXOUT,'temperature':0,'seed':0,'thinking':False,'runner_sha256':p.sha(RUNNER),'prior_plan_sha256':p.sha(prior.PLAN),
      'probe_plan_sha256':p.sha(probe.PLAN),'baseline_plan_sha256':p.sha(complete.PLAN),'reuse_sha256':sources,
      'groups':[{'paper_ids':g['paper_ids'],'candidate_count':len(g['fine']),'rows_sha256':digest(g['rows']),'candidates_sha256':digest(sorted(g['fine'])),
       'inputs':{n:digest(prior.fmt.render(g,n)) for n in LAYOUTS},'max_followup_sha256':digest(prior.prior.expanded(g,g['fine']))} for g in gs]}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or data changed')
    return prior.data()

def preflight():
    gs=frozen();entries=[];key=os.environ['QWEN_API_KEY'];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize'
    for i,g in enumerate(gs,1):
        cases=[('first_'+n,probe.prompt(),prior.fmt.render(g,n)) for n in LAYOUTS]+[('max_followup',prior.prompt('followup'),prior.prior.expanded(g,g['fine']))]
        for name,system,user in cases:
            body=json.dumps({'model':prior.base.MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as r:o=json.load(r)
            n=o['count'];limit=o.get('max_model_len',262144);z={'group':i,'case':name,'tokens':n,'limit':limit,'fits':n+MAXOUT+512<=limit};entries.append(z)
        print('preflight',i,'ok',all(x['fits'] for x in entries if x['group']==i),flush=True)
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'status':'ok' if all(z['fits'] for z in entries) else 'too_large','entries':entries},indent=2)+'\n',encoding='utf8')
    if not all(z['fits'] for z in entries):raise SystemExit('Context exceeds budget')

def source_record(g,i,layout,stage):
    path=DEST/f'paper{i:02d}_{layout}_{stage}.json'
    if path.exists():
        r=json.loads(path.read_text(encoding='utf8'))
        if r.get('status')=='ok' and r.get('plan_sha256')==p.sha(PLAN) and r.get('selected_sha256')==digest(sorted(g['fine'])):return r
        raise ValueError('Existing failed or changed result')
    source=baseline_source(g,layout,i) if stage=='baseline' else probe_source(i,layout)
    if source is None:return None
    r=json.loads(source.read_text(encoding='utf8'))
    if stage=='baseline':
        kept={tuple(k) for k in r['kept']};decisions=[{'key':list(k),'decision':'keep' if k in kept else 'drop'} for k in g['keys'] if k in g['fine']]
    else:decisions=r['decisions']
    out={'status':'ok','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'stage':stage,'selected_sha256':digest(sorted(g['fine'])),
      'input_sha256':digest(prior.fmt.render(g,layout)),'decisions':decisions,'attempts':r['attempts'],'reused_from':source.relative_to(p.ROOT).as_posix(),'source_sha256':p.sha(source)}
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8');return out

def first(g,i,layout,c):
    cached=source_record(g,i,layout,'first')
    if cached:return cached
    path=DEST/f'paper{i:02d}_{layout}_first.json';keys=g['keys'];expected={j for j,k in enumerate(keys) if k in g['fine']};user=prior.fmt.render(g,layout)
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep','needs_more_context'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'},'needs_more_context':{'type':'boolean'}}}}}}
    attempts=[];out=None
    for _ in range(2):
        t=time.time();a=None
        try:
            r=c.chat.completions.create(model=prior.base.MODEL,temperature=0,max_tokens=MAXOUT,messages=[{'role':'system','content':probe.prompt()},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'flag','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            a={'seconds':round(time.time()-t,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
            if a['finish_reason']!='stop':raise ValueError('Truncated')
            obj=json.loads(a['response']);items=obj['results']
            if set(obj)!={'results'} or not isinstance(items,list) or any(not isinstance(z,dict) or set(z)!={'id','keep','needs_more_context'} or type(z['id'])!=int or type(z['keep'])!=bool or type(z['needs_more_context'])!=bool for z in items):raise ValueError('Bad schema')
            good=[z for z in items if z['id'] in expected];ids=[z['id'] for z in good]
            if set(ids)!=expected or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate target IDs')
            a['excluded_extra_entries']=[z for z in items if z['id'] not in expected]
            out={'status':'ok','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'stage':'first','selected_sha256':digest(sorted(g['fine'])),'input_sha256':digest(user),
             'decisions':[{'key':list(keys[z['id']]),'keep':z['keep'],'needs_more_context':z['needs_more_context']} for z in good],'attempts':attempts};break
        except Exception as err:
            if a is None:attempts.append({'seconds':round(time.time()-t,2),'error_type':type(err).__name__})
            else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
    if out is None:out={'status':'failed','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'stage':'first','attempts':attempts}
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    with LOCK:print(json.dumps({'paper':i,'layout':layout,'stage':'first','status':out['status'],'targets':len(expected),'needs_more_context':sum(z['needs_more_context'] for z in out.get('decisions',[]))}),flush=True)
    if out['status']!='ok':raise ValueError('Failed first pass')
    return out

def run_pair(g,i,layout):
    c=p.client()
    if source_record(g,i,layout,'baseline') is None:prior.invoke(g,i,layout,'baseline',g['fine'],c)
    r=first(g,i,layout,c);flags={tuple(z['key']) for z in r['decisions'] if z['needs_more_context']}
    prior.invoke(g,i,layout,'followup',flags,c)

def run():
    gs=frozen();guard.check_committed(PRE);q=json.loads(PRE.read_text(encoding='utf8'))
    if q['status']!='ok' or q['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    # Reuse the audited baseline/followup runner with this study's output root.
    prior.DEST=DEST;prior.PLAN=PLAN
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_pair,g,i,n) for i,g in enumerate(gs,1) for n in LAYOUTS]
        for job in concurrent.futures.as_completed(jobs):
            try:job.result()
            except Exception:
                for j in jobs:j.cancel()
                raise

def summary():
    gs=frozen();result={}
    for n in LAYOUTS:
        rows={};baseline=set();kept=set();flags=set();changed_to_keep=set();bi=bo=ai=ao=bc=ac=0
        for i,g in enumerate(gs,1):
            b,f,s=[json.loads((DEST/f'paper{i:02d}_{n}_{stage}.json').read_text(encoding='utf8')) for stage in ('baseline','first','followup')]
            if any(r.get('status')!='ok' or r.get('plan_sha256')!=p.sha(PLAN) for r in (b,f,s)):raise ValueError('Incomplete')
            flag={tuple(z['key']) for z in f['decisions'] if z['needs_more_context']}
            if s['selected_sha256']!=digest(sorted(flag)) or {tuple(z['key']) for z in s['decisions']}!=flag:raise ValueError('Followup mismatch')
            for r in g['rows']:rows[(r['pid'],r['idx'])]=r
            baseline|={tuple(z['key']) for z in b['decisions'] if z['decision']=='keep'}
            kept|={tuple(z['key']) for z in f['decisions'] if z['keep'] and not z['needs_more_context']};flags|=flag
            accepted={tuple(z['key']) for z in s['decisions'] if z['decision']=='keep'};changed_to_keep|=accepted;kept|=accepted
            for a in b['attempts']:
                u=a.get('usage') or {};bi+=u.get('prompt_tokens',0);bo+=u.get('completion_tokens',0);bc+=1
            for r in (f,s):
                for a in r['attempts']:
                    u=a.get('usage') or {};ai+=u.get('prompt_tokens',0);ao+=u.get('completion_tokens',0);ac+=1
        rr=list(rows.values());result[n]={'papers':10,'request_groups':len(gs),'candidates':sum(len(g['fine']) for g in gs),'flagged':len(flags),'followup_kept':len(changed_to_keep),
         'baseline':{'kept':len(baseline),'metrics':p.metrics(rr,baseline),'input_tokens':bi,'output_tokens':bo,'calls_including_retries':bc},
         'adaptive':{'kept':len(kept),'metrics':p.metrics(rr,kept),'input_tokens':ai,'output_tokens':ao,'calls_including_retries':ac}}
    out={'plan_sha256':p.sha(PLAN),'results':result};print(json.dumps(out,indent=2),flush=True)
    if (DEST/'summary.json').exists():raise ValueError('Summary exists')
    (DEST/'summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='summary':summary()
