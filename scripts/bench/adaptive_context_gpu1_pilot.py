"""Matched 10-paper uncertainty-routing pilot on existing GPU 1 Qwen service."""
import collections,concurrent.futures,json,os,sys,time,urllib.request,threading
import adaptive_context_development as prior
import five_paper_context_format_34 as fmt
import five_paper_68_verdicts as base
import five_paper_development_prompt as rubric
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/adaptive_context_gpu1_pilot_2026-11-27'
PLAN=DEST/'plan.json';PRE=DEST/'preflight.json';RUNNER=p.ROOT/'scripts/bench/adaptive_context_gpu1_pilot.py'
LAYOUTS=('no_context','surrounding');MAXOUT=8192;LOCK=threading.Lock()
OUTPUT=rubric.NEW
TRISTATE=('Return JSON of the form {"results":[{"id":<id>,"decision":"keep"|"drop"|"unsure"}, ...]} with exactly one entry for every numbered target and the original ID. '
 'Choose "keep" when the target itself clearly states a gap and "drop" when it clearly does not. '
 'Choose "unsure" whenever the target cannot be reliably judged without more original surrounding text: '
 'for example, a reference such as "this", "these", "it", or "the approach" hides what is limited, or the target '
 'could be either a limitation or an ordinary observation depending on its context. When context could change '
 'the decision, prefer "unsure". Only numbered targets are judged; unnumbered neighbors are context.')

def data():
    original=base.frozen()[:2];result=[]
    for old in original:
        for pid in old['paper_ids']:
            rows=[r for r in old['rows'] if r['pid']==pid];keys=[(r['pid'],r['idx']) for r in rows]
            result.append({'paper_ids':[pid],'rows':rows,'keys':keys,'fine':old['fine']&set(keys)})
    if len(result)!=10:raise ValueError('Expected ten dev papers')
    return result

def prompt(mode):
    s=base.prompt.prompt('verdicts');boundary='There are five separate papers below.'
    if s.count(boundary)!=1:raise ValueError('Changed paper instructions')
    s=s.replace(boundary,'There is one paper below.')
    if mode=='first':
        if s.count(OUTPUT)!=1:raise ValueError('Changed output rubric')
        s=s.replace(OUTPUT,TRISTATE)
    elif mode=='followup':s+='\n\nThe numbered targets were marked unsure in a first pass. Use the original neighboring sentences to disambiguate but judge the target itself. Give a final keep or drop decision for every target.'
    elif mode!='baseline':raise ValueError('Unknown mode')
    return s

def plan():
    gs=data()
    return {'purpose':'Matched uncertainty-routing pilot with and without first-pass neighbors, and matching one-pass baselines, on GPU 1.',
      'dataset':'The ten arXiv development papers in the first two groups of the earlier 34-paper study. One paper per request. They have prior development exposure and LLM-generated labels.',
      'stages':'Baseline: one boolean call per paper and layout. Adaptive: initial keep/drop/unsure call; only unsure targets get a second boolean call with up to three previous and three following original sentences (800 characters per neighbor).',
      'first_layouts':list(LAYOUTS),'prompts':{m:prompt(m) for m in ('baseline','first','followup')},
      'backend':'Existing GPU 1 shared vLLM 0.34.0 service via SSH tunnel; Qwen3.8-27B-FP8 snapshot 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, max context 49152. GPU 0 runs were stopped and are excluded.',
      'decoding':'temperature 0, seed 0, thinking disabled, 8192 max output tokens, strict JSON schema; every target ID must appear exactly once, extras discarded and recorded.',
      'calibration_note':'The first wording on GPU 0 returned zero unsure outcomes across six calls on 15 papers. This pilot uses a more explicit uncertainty instruction after observing that failure, and is exploratory.',
      'limitations':'Development labels are LLM-generated and previously studied. One paper per request; ten papers, no test set and no human adjudication. Different backend and batching from earlier five-paper runs.',
      'runner_sha256':p.sha(RUNNER),'base_plan_sha256':p.sha(base.PLAN),'expansion_script_sha256':p.sha(p.ROOT/'scripts/bench/adaptive_context_development.py'),
      'groups':[{'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),'candidate_sha256':digest(sorted(g['fine'])),'candidate_count':len(g['fine']),
        'first_input_sha256':{n:digest(fmt.render(g,n)) for n in LAYOUTS},'max_followup_sha256':digest(prior.expanded(g,g['fine']))} for g in gs]}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or inputs changed')
    return data()

def preflight():
    gs=frozen();url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize';entries=[]
    for i,g in enumerate(gs,1):
        cases=[('first_'+n,prompt('first'),fmt.render(g,n)) for n in LAYOUTS]+[('max_followup',prompt('followup'),prior.expanded(g,g['fine']))]
        for name,system,user in cases:
            payload=json.dumps({'model':base.MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(url,data=payload,headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['QWEN_API_KEY']})
            with urllib.request.urlopen(req,timeout=180) as r:o=json.load(r)
            n=o['count'];limit=o.get('max_model_len',49152);z={'group':i,'case':name,'input_tokens':n,'limit':limit,'fits':n+MAXOUT+512<=limit};entries.append(z);print(json.dumps(z),flush=True)
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'status':'ok' if all(z['fits'] for z in entries) else 'too_large','entries':entries},indent=2)+'\n',encoding='utf8')
    if not all(z['fits'] for z in entries):raise SystemExit('Pilot input exceeds context')

def invoke(g,i,layout,stage,selected,c):
    path=DEST/f'paper{i:02d}_{layout}_{stage}.json'
    if path.exists():
        r=json.loads(path.read_text(encoding='utf8'))
        if r.get('status')=='ok' and r.get('plan_sha256')==p.sha(PLAN) and r.get('selected_sha256')==digest(sorted(selected)):return r
        raise ValueError('Existing failed or changed output')
    if not selected:
        out={'status':'ok','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'stage':stage,'selected_sha256':digest([]),'decisions':[],'attempts':[],'skipped_empty':True};path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');return out
    first=stage=='first';user=prior.expanded(g,selected) if stage=='followup' else fmt.render(g,layout);keys=g['keys'];ids={j for j,k in enumerate(keys) if k in selected};schema=prior.schema(first);attempts=[];out=None
    for _ in range(2):
        start=time.time();a=None
        try:
            resp=c.chat.completions.create(model=base.MODEL,temperature=0,max_tokens=MAXOUT,messages=[{'role':'system','content':prompt(stage)},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'decisions','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            a={'seconds':round(time.time()-start,2),'finish_reason':resp.choices[0].finish_reason,'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(a)
            if a['finish_reason']!='stop':raise ValueError('Truncated')
            obj=json.loads(a['response']);items=obj['results'];field='decision' if first else 'keep'
            if set(obj)!={'results'} or not isinstance(items,list):raise ValueError('Bad schema')
            for z in items:
                if not isinstance(z,dict) or set(z)!={'id',field} or type(z['id'])!=int:raise ValueError('Bad item')
                if first and z[field] not in ('keep','drop','unsure'):raise ValueError('Bad decision')
                if not first and type(z[field])!=bool:raise ValueError('Bad verdict')
            good=[z for z in items if z['id'] in ids];extra=[z for z in items if z['id'] not in ids];got=[z['id'] for z in good]
            if set(got)!=ids or len(got)!=len(set(got)):raise ValueError('Missing or duplicated target ID')
            a['excluded_extra_entries']=extra; decisions=[{'key':list(keys[z['id']]),'decision':z[field] if first else ('keep' if z[field] else 'drop')} for z in good]
            out={'status':'ok','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'stage':stage,'selected_sha256':digest(sorted(selected)),'input_sha256':digest(user),'decisions':decisions,'attempts':attempts};break
        except Exception as err:
            if a is None:attempts.append({'seconds':round(time.time()-start,2),'error_type':type(err).__name__})
            else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
    if out is None:out={'status':'failed','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'stage':stage,'selected_sha256':digest(sorted(selected)),'attempts':attempts}
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    with LOCK:print(json.dumps({'paper':i,'layout':layout,'stage':stage,'status':out['status'],'targets':len(selected),'decisions':dict(collections.Counter(z['decision'] for z in out.get('decisions',[]))),'usage':attempts[-1].get('usage')}),flush=True)
    if out['status']!='ok':raise ValueError(f'Failed: {path.name}')
    return out

def run_pair(g,i,layout):
    c=p.client();invoke(g,i,layout,'baseline',g['fine'],c);first=invoke(g,i,layout,'first',g['fine'],c)
    unsure={tuple(z['key']) for z in first['decisions'] if z['decision']=='unsure'}
    invoke(g,i,layout,'followup',unsure,c)

def run():
    gs=frozen();guard.check_committed(PRE);q=json.loads(PRE.read_text(encoding='utf8'))
    if q['status']!='ok' or q['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_pair,g,i,n) for i,g in enumerate(gs,1) for n in LAYOUTS]
        for job in concurrent.futures.as_completed(jobs):
            try:job.result()
            except Exception:
                for j in jobs:j.cancel()
                raise

def summary():
    gs=frozen();out={}
    for layout in LAYOUTS:
        rows=[];baseline=set();kept=set();unsure=set();followup_yes=set();binp=bout=ainp=aout=bcalls=acalls=0
        for i,g in enumerate(gs,1):
            br,fr,sr=[json.loads((DEST/f'paper{i:02d}_{layout}_{stage}.json').read_text(encoding='utf8')) for stage in ('baseline','first','followup')]
            if any(r.get('status')!='ok' or r.get('plan_sha256')!=p.sha(PLAN) for r in (br,fr,sr)):raise ValueError('Incomplete output')
            uu={tuple(z['key']) for z in fr['decisions'] if z['decision']=='unsure'}
            if sr['selected_sha256']!=digest(sorted(uu)) or {tuple(z['key']) for z in sr['decisions']}!=uu:raise ValueError('Followup mismatch')
            rows+=g['rows'];baseline|={tuple(z['key']) for z in br['decisions'] if z['decision']=='keep'};kept|={tuple(z['key']) for z in fr['decisions'] if z['decision']=='keep'};unsure|=uu
            yes={tuple(z['key']) for z in sr['decisions'] if z['decision']=='keep'};followup_yes|=yes;kept|=yes
            for a in br['attempts']:
                u=a.get('usage') or {};binp+=u.get('prompt_tokens',0);bout+=u.get('completion_tokens',0);bcalls+=1
            for r in (fr,sr):
                for a in r['attempts']:
                    u=a.get('usage') or {};ainp+=u.get('prompt_tokens',0);aout+=u.get('completion_tokens',0);acalls+=1
        out[layout]={'papers':10,'candidates':sum(len(g['fine']) for g in gs),'unsure':len(unsure),'followup_kept':len(followup_yes),
          'baseline':{'kept':len(baseline),'metrics':p.metrics(rows,baseline),'input_tokens':binp,'output_tokens':bout,'calls':bcalls},
          'adaptive':{'kept':len(kept),'metrics':p.metrics(rows,kept),'input_tokens':ainp,'output_tokens':aout,'calls':acalls}}
    result={'plan_sha256':p.sha(PLAN),'results':out};print(json.dumps(result,indent=2),flush=True)
    path=DEST/'summary.json'
    if path.exists():raise ValueError('Summary exists')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='summary':summary()
