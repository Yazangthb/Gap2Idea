"""Qwen abstention followed by wider context, with and without first-pass neighbors."""
import concurrent.futures,json,os,sys,time,urllib.request,threading
import five_paper_context_format_34 as fmt
import five_paper_68_verdicts as base
import five_paper_development_prompt as rubric
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/adaptive_context_development_2026-11-27'
PLAN=DEST/'plan.json';PRE=DEST/'preflight.json';RUNNER=p.ROOT/'scripts/bench/adaptive_context_development.py'
LAYOUTS=('no_context','surrounding');MAXOUT=16384;LOCK=threading.Lock()
OUTPUT=rubric.NEW
FIRST_OUTPUT=('Return JSON of the form {"results":[{"id":<id>,"decision":"keep"|"drop"|"unsure"}, ...]} '
 'with EXACTLY one entry for every numbered target, using its original id and input order. '
 'Use "keep" when the target states a research gap, "drop" when it does not, and "unsure" only when '
 'additional original surrounding text is needed to decide reliably. If the available text is sufficient, '
 'choose keep or drop. Judge the numbered target itself; any unnumbered neighbors only clarify it.')

def first_system():
    s=base.prompt.prompt('verdicts')
    if s.count(OUTPUT)!=1:raise ValueError('Unexpected output rubric')
    return s.replace(OUTPUT,FIRST_OUTPUT)

def second_system():
    return base.prompt.prompt('verdicts')+'\n\nThe numbered targets were uncertain in a first pass. Use the surrounding original sentences to disambiguate, but decide whether the numbered target itself states a gap. Give a final keep or drop decision for every target.'

def expanded(g,selected):
    rows=g['rows'];keys=g['keys'];lines=[];last=None
    for i,r in enumerate(rows):
        if keys[i] not in selected:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        before=[rows[j]['sentence'][-800:] for j in range(max(0,i-3),i) if rows[j]['pid']==r['pid']]
        after=[rows[j]['sentence'][:800] for j in range(i+1,min(len(rows),i+4)) if rows[j]['pid']==r['pid']]
        lines.append('Before: '+' '.join(before)+f'\n[{i}] {r["sentence"]}\nAfter: '+' '.join(after))
    return '\n'.join(lines)

def groups():return base.frozen()[:8]

def baseline(i,layout):return fmt.DEST/f'group{i:02d}_{layout}.json'

def plan():
    gs=groups()
    return {'purpose':'Test Qwen uncertainty-routing, starting with or without one original neighbor on each side; compare against corresponding existing single-pass baselines on all 34 development papers.',
      'stages':'First call classifies all frozen NLI candidates as keep/drop/unsure. Second call contains only unsure targets, with up to 3 previous and 3 next original sentences (each <=800 characters); its boolean decision replaces unsure. Other first-pass decisions remain fixed.',
      'first_layouts':list(LAYOUTS),'first_system':first_system(),'second_system':second_system(),
      'groups':'Same existing paper groups of up to five, 28 arXiv and 6 biomedical; same fine-tuned NLI candidates and threshold; one first call per group/layout and at most one followup call.',
      'output_validation':'Every target must occur exactly once and have valid decision. Extra out-of-input IDs discarded and recorded; missing, duplicated or conflicting target IDs, truncated output and malformed responses fail.',
      'metrics':'Precision, recall and F1 against all labeled sentences; report first-pass unsure counts, final kept counts, Qwen input/output tokens and calls for both passes including retries. No test set or human judgments are used.',
      'caveats':'Development papers were previously studied and used in model/prompt selection; labels are LLM-generated, not human-validated. One deterministic run per setup. Different prompts and multi-call costs mean this is exploratory.',
      'model':base.MODEL,'revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','temperature':0,'seed':0,'thinking':False,'max_output_tokens':MAXOUT,
      'base_plan_sha256':p.sha(base.PLAN),'layout_plan_sha256':p.sha(fmt.PLAN),'runner_sha256':p.sha(RUNNER),
      'baseline_sha256':{f'{i}_{n}':p.sha(baseline(i,n)) for i in range(1,9) for n in LAYOUTS},
      'groups':[{'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),'candidates':len(g['fine']),'candidate_sha256':digest(sorted(g['fine'])),
       'first_input_sha256':{n:digest(fmt.render(g,n)) for n in LAYOUTS},'maximum_second_input_sha256':digest(expanded(g,g['fine']))} for g in gs]}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or data changed')
    return groups()

def preflight():
    gs=frozen();key=os.environ['QWEN_API_KEY'];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize';entries=[]
    for i,g in enumerate(gs,1):
        cases=[('first_'+n,first_system(),fmt.render(g,n)) for n in LAYOUTS]+[('max_followup',second_system(),expanded(g,g['fine']))]
        for name,system,user in cases:
            data=json.dumps({'model':base.MODEL,'messages':[{'role':'system','content':system},{'role':'user','content':user}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(url,data=data,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as r:o=json.load(r)
            n=o['count'];limit=o.get('max_model_len',262144);z={'group':i,'case':name,'input_tokens':n,'limit':limit,'fits':n+MAXOUT+512<=limit};entries.append(z);print(json.dumps(z),flush=True)
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'status':'ok' if all(z['fits'] for z in entries) else 'too_large','entries':entries},indent=2)+'\n',encoding='utf8')
    if not all(z['fits'] for z in entries):raise SystemExit('Context exceeds budget')

def schema(first):
    prop={'type':'string','enum':['keep','drop','unsure']} if first else {'type':'boolean'};key='decision' if first else 'keep'
    return {'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id',key],'properties':{'id':{'type':'integer'},key:prop}}}}}

def invoke(g,i,layout,first,selected,c):
    suffix='first' if first else 'followup';path=DEST/f'group{i:02d}_{layout}_{suffix}.json'
    if path.exists():
        out=json.loads(path.read_text(encoding='utf8'))
        if out.get('status')=='ok' and out.get('plan_sha256')==p.sha(PLAN) and out.get('selected_sha256')==digest(sorted(selected)):return out
        raise ValueError('Failed/changed existing output')
    if not selected:
        out={'status':'ok','plan_sha256':p.sha(PLAN),'group':i,'layout':layout,'stage':suffix,'selected_sha256':digest([]),'decisions':[],'attempts':[],'skipped_empty':True}
        path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');return out
    user=fmt.render(g,layout) if first else expanded(g,selected);system=first_system() if first else second_system();keys=g['keys'];input_ids={j for j,k in enumerate(keys) if k in selected};attempts=[];out=None
    for _ in range(2):
        t=time.time();a=None
        try:
            r=c.chat.completions.create(model=base.MODEL,temperature=0,max_tokens=MAXOUT,messages=[{'role':'system','content':system},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'decisions','schema':schema(first),'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            a={'seconds':round(time.time()-t,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
            if a['finish_reason']!='stop':raise ValueError('Truncated')
            obj=json.loads(a['response']);items=obj['results'];field='decision' if first else 'keep'
            if set(obj)!={'results'} or not isinstance(items,list):raise ValueError('Bad schema')
            for z in items:
                if not isinstance(z,dict) or set(z)!={'id',field} or type(z['id'])!=int:raise ValueError('Bad item')
                if first and z[field] not in ('keep','drop','unsure'):raise ValueError('Bad decision')
                if not first and type(z[field])!=bool:raise ValueError('Bad verdict')
            good=[z for z in items if z['id'] in input_ids];extra=[z for z in items if z['id'] not in input_ids];ids=[z['id'] for z in good]
            if set(ids)!=input_ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate target ID')
            a['excluded_extra_entries']=extra
            decisions=[{'key':list(keys[z['id']]),'decision':z[field] if first else ('keep' if z[field] else 'drop')} for z in good]
            out={'status':'ok','plan_sha256':p.sha(PLAN),'group':i,'layout':layout,'stage':suffix,'selected_sha256':digest(sorted(selected)),'input_sha256':digest(user),'decisions':decisions,'attempts':attempts};break
        except Exception as err:
            if a is None:attempts.append({'seconds':round(time.time()-t,2),'error_type':type(err).__name__})
            else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
    if out is None:out={'status':'failed','plan_sha256':p.sha(PLAN),'group':i,'layout':layout,'stage':suffix,'selected_sha256':digest(sorted(selected)),'attempts':attempts}
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    with LOCK:print(json.dumps({'group':i,'layout':layout,'stage':suffix,'status':out['status'],'targets':len(selected),'decisions':dict(__import__('collections').Counter(z['decision'] for z in out.get('decisions',[]))),'usage':attempts[-1].get('usage')}),flush=True)
    if out['status']!='ok':raise ValueError(f'Invalid response: {path.name}')
    return out

def run_pair(g,i,layout):
    c=p.client();first=invoke(g,i,layout,True,g['fine'],c)
    unsure={tuple(z['key']) for z in first['decisions'] if z['decision']=='unsure'}
    invoke(g,i,layout,False,unsure,c)

def run():
    gs=frozen();guard.check_committed(PRE);pre=json.loads(PRE.read_text(encoding='utf8'))
    if pre['status']!='ok' or pre['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run_pair,g,i,n) for i,g in enumerate(gs,1) for n in LAYOUTS]
        for f in concurrent.futures.as_completed(jobs):
            try:f.result()
            except Exception:
                for j in jobs:j.cancel()
                raise

def summary():
    gs=frozen();results={}
    for n in LAYOUTS:
        rows=[];keep=set();baseline_keep=set();unsure=set();second_keep=set();ti=to=0;calls=0;btin=btout=0;domain_results={}
        for i,g in enumerate(gs,1):
            first=json.loads((DEST/f'group{i:02d}_{n}_first.json').read_text(encoding='utf8'));second=json.loads((DEST/f'group{i:02d}_{n}_followup.json').read_text(encoding='utf8'));bl=json.loads(baseline(i,n).read_text(encoding='utf8'))
            if any(r.get('status')!='ok' for r in (first,second,bl)) or first['plan_sha256']!=p.sha(PLAN) or second['plan_sha256']!=p.sha(PLAN) or bl['plan_sha256']!=p.sha(fmt.PLAN):raise ValueError('Incomplete output')
            first_unsure={tuple(z['key']) for z in first['decisions'] if z['decision']=='unsure'}
            if second['selected_sha256']!=digest(sorted(first_unsure)) or {tuple(z['key']) for z in second['decisions']}!=first_unsure:raise ValueError('Followup mismatch')
            rows+=g['rows'];unsure|=first_unsure;keep|={tuple(z['key']) for z in first['decisions'] if z['decision']=='keep'}
            second_yes={tuple(z['key']) for z in second['decisions'] if z['decision']=='keep'};second_keep|=second_yes;keep|=second_yes
            baseline_keep.update(tuple(z) for z in bl['kept'])
            for r in (first,second):
                for a in r['attempts']:
                    u=a.get('usage') or {};ti+=u.get('prompt_tokens',0);to+=u.get('completion_tokens',0);calls+=1
            for a in bl['attempts']:
                u=a.get('usage') or {};btin+=u.get('prompt_tokens',0);btout+=u.get('completion_tokens',0)
        for dom in ('arxiv_cs','biomed_sal'):
            rr=[r for r in rows if r['dom']==dom];domain_results[dom]={'adaptive':p.metrics(rr,keep),'baseline':p.metrics(rr,baseline_keep)}
        results[n]={'candidates':sum(len(g['fine']) for g in gs),'unsure':len(unsure),'followup_kept':len(second_keep),
          'baseline':{'kept':len(baseline_keep),'metrics':p.metrics(rows,baseline_keep),'input_tokens':btin,'output_tokens':btout},
          'adaptive':{'kept':len(keep),'metrics':p.metrics(rows,keep),'input_tokens':ti,'output_tokens':to,'calls_including_retries':calls},'by_domain':domain_results}
    out={'plan_sha256':p.sha(PLAN),'results':results};print(json.dumps(out,indent=2),flush=True)
    path=DEST/'summary.json'
    if path.exists():raise ValueError('Summary exists')
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='summary':summary()
