"""Two-paper probe: ask for a separate context-needed boolean per candidate."""
import json,os,sys,time,urllib.request
import adaptive_context_gpu1_pilot as prior
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/adaptive_context_flag_gpu0_probe_2026-11-27'
PLAN=DEST/'plan.json';RUNNER=p.ROOT/'scripts/bench/adaptive_context_flag_gpu0_probe.py';PRE=DEST/'preflight.json'
OUTPUT=prior.rubric.NEW
FLAG=('Return JSON {"results":[{"id":<id>,"keep":<bool>,"needs_more_context":<bool>}, ...]} with exactly one entry for every numbered target. '
 'Give your best provisional keep/drop decision in "keep". Independently, set "needs_more_context" to true when '
 'you need additional original surrounding text for a reliable decision, even if you have a tentative guess. '
 'Mark true when a reference such as "this", "these", "it" or "the method" hides what is limited or when the '
 'target could reasonably be a gap or an ordinary observation depending on its context. Otherwise mark false. '
 'Only numbered targets are classified; unnumbered neighbors are context.')

def prompt():
    s=prior.prompt('baseline')
    if s.count(OUTPUT)!=1:raise ValueError('Unexpected baseline rubric')
    return s.replace(OUTPUT,FLAG)

def plan():
    gs=prior.frozen()[:2]
    return {'purpose':'Probe a separate context-needed boolean after the tri-state prompt routed zero candidates; two development papers, both first-pass layouts.',
      'limitations':'This is a small post-hoc pilot on previously studied LLM-labeled development papers; no test, no full evaluation, no independent inference.',
      'backend':'Existing GPU 0 Qwen3.8-27B FP8 vLLM service 0.32.0; no GPU 1 inference.',
      'system':prompt(),'model':prior.base.MODEL,'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192,
      'prior_plan_sha256':p.sha(prior.PLAN),'runner_sha256':p.sha(RUNNER),
      'groups':[{'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),'candidate_sha256':digest(sorted(g['fine'])),'candidate_count':len(g['fine']),
       'inputs':{n:digest(prior.fmt.render(g,n)) for n in prior.LAYOUTS}} for g in gs]}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan changed')
    return prior.data()[:2]

def preflight():
    gs=frozen();out=[];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize'
    for i,g in enumerate(gs,1):
        for layout in prior.LAYOUTS:
            text=prior.fmt.render(g,layout);body=json.dumps({'model':prior.base.MODEL,'messages':[{'role':'system','content':prompt()},{'role':'user','content':text}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(url,data=body,headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ['QWEN_API_KEY']})
            with urllib.request.urlopen(req,timeout=120) as r:o=json.load(r)
            n=o['count'];limit=o.get('max_model_len',49152);out.append({'paper':i,'layout':layout,'tokens':n,'limit':limit,'fits':n+8192+512<=limit})
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'entries':out,'status':'ok' if all(x['fits'] for x in out) else 'too_large'},indent=2)+'\n',encoding='utf8')
    print(json.dumps(out),flush=True)
    if not all(x['fits'] for x in out):raise SystemExit('Input too large')

def run():
    gs=frozen();guard.check_committed(PRE);q=json.loads(PRE.read_text(encoding='utf8'))
    if q['status']!='ok' or q['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    c=p.client();schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep','needs_more_context'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'},'needs_more_context':{'type':'boolean'}}}}}}
    for i,g in enumerate(gs,1):
        for layout in prior.LAYOUTS:
            path=DEST/f'paper{i:02d}_{layout}.json'
            if path.exists():raise ValueError('Existing output')
            keys=g['keys'];expected={j for j,k in enumerate(keys) if k in g['fine']};text=prior.fmt.render(g,layout);attempts=[];result=None
            for _ in range(2):
                t=time.time();a=None
                try:
                    r=c.chat.completions.create(model=prior.base.MODEL,temperature=0,max_tokens=8192,messages=[{'role':'system','content':prompt()},{'role':'user','content':text}],response_format={'type':'json_schema','json_schema':{'name':'flag','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                    a={'seconds':round(time.time()-t,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
                    if a['finish_reason']!='stop':raise ValueError('Truncated')
                    obj=json.loads(a['response']);items=obj['results']
                    if set(obj)!={'results'} or not isinstance(items,list) or any(not isinstance(z,dict) or set(z)!={'id','keep','needs_more_context'} or type(z['id'])!=int or type(z['keep'])!=bool or type(z['needs_more_context'])!=bool for z in items):raise ValueError('Bad schema')
                    good=[z for z in items if z['id'] in expected];ids=[z['id'] for z in good]
                    if set(ids)!=expected or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate IDs')
                    a['excluded_extra_entries']=[z for z in items if z['id'] not in expected]
                    result={'status':'ok','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'decisions':[{'key':list(keys[z['id']]),'keep':z['keep'],'needs_more_context':z['needs_more_context']} for z in good],'attempts':attempts};break
                except Exception as err:
                    if a is None:attempts.append({'seconds':round(time.time()-t,2),'error_type':type(err).__name__})
                    else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
            if result is None:result={'status':'failed','plan_sha256':p.sha(PLAN),'paper':i,'layout':layout,'attempts':attempts}
            path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
            print(json.dumps({'paper':i,'layout':layout,'status':result['status'],'targets':len(expected),'needs_more_context':sum(z['needs_more_context'] for z in result.get('decisions',[])),'usage':attempts[-1].get('usage')}),flush=True)
            if result['status']!='ok':raise SystemExit('Failed; evidence saved')

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run()
