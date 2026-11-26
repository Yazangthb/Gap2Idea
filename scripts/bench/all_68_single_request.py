"""Compare one 68-paper no-context funnel call with prior groups of <=5."""
import json, os, sys, time, urllib.request
import five_paper_68_verdicts as old
import five_paper_context_format_34 as fmt
import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/all_68_single_request_2026-11-26'
PLAN=DEST/'plan.json';PRE=DEST/'preflight.json';OUT=DEST/'single.json'
RUNNER=p.ROOT/'scripts/bench/all_68_single_request.py';MAXOUT=65536
DEV=p.ROOT/'docs/experiments/five_paper_context_format_34_2026-11-26'
TEST=p.ROOT/'docs/experiments/five_paper_context_test_34_2026-11-26'

def data():
    groups=old.frozen(); keys=[k for g in groups for k in g['keys']];rows=[r for g in groups for r in g['rows']]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate sentence key')
    g={'keys':keys,'rows':rows,'fine':set().union(*(x['fine'] for x in groups))}
    return groups,g

def system():
    orig=old.prompt.prompt('verdicts');txt='There are five separate papers below.'
    if orig.count(txt)!=1:raise ValueError('Unexpected prompt')
    return orig.replace(txt,'There are 68 separate papers below.')

def baseline_path(i): return (DEV if i<=8 else TEST)/f'group{(i-1)%8+1:02d}_no_context.json'

def plan():
    groups,g=data(); user=fmt.render(g,'no_context')
    return {'purpose':'Compare one 68-paper call to 16 calls on the same 68 papers and same fine-NLI candidates without neighboring context.',
      'caveat':'Full-paper and neighbor-context 68-paper calls exceed the served context limit, so no-context funnel only. All papers have prior development/test exposure and LLM-generated labels.',
      'comparison':'Original order of papers and targets preserved, only grouping and numbering change; system paper count updated to 68. Results grouped by original split/domain, not an independent test.',
      'model':old.MODEL,'model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','temperature':0,'seed':0,'thinking':False,'max_tokens':MAXOUT,
      'validation':'Require exactly one boolean verdict per real candidate. Discard and record extra IDs only when every real candidate appears exactly once; incomplete answers are failures.',
      'prompt':system(),'input_sha256':digest(user),'rows_sha256':digest(g['rows']),'candidate_count':len(g['fine']),
      'original_plan_sha256':p.sha(old.PLAN),'runner_sha256':p.sha(RUNNER),'format_runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_context_format_34.py'),
      'baseline_sha256':{str(i):p.sha(baseline_path(i)) for i in range(1,17)},
      'groups':[{'split':x['split'],'domain':x['domain'],'paper_ids':x['paper_ids']} for x in groups]}

def frozen():
    d.check_committed(PLAN);d.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan changed')
    return data()

def preflight():
    _,g=frozen(); user=fmt.render(g,'no_context');key=os.environ['QWEN_API_KEY'];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize'
    payload=json.dumps({'model':old.MODEL,'messages':[{'role':'system','content':system()},{'role':'user','content':user}],'add_generation_prompt':True},ensure_ascii=False).encode()
    req=urllib.request.Request(url,data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:o=json.load(r)
    count=o['count'];limit=o.get('max_model_len',262144);result={'plan_sha256':p.sha(PLAN),'input_tokens':count,'max_tokens':MAXOUT,'limit':limit,'fits':count+MAXOUT+512<=limit}
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)
    if not result['fits']:raise SystemExit('Too long')

def run():
    _,g=frozen();d.check_committed(PRE);pre=json.loads(PRE.read_text(encoding='utf8'))
    if not pre['fits'] or pre['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    if OUT.exists():raise ValueError('Existing result')
    keys=g['keys']; input_ids={i for i,k in enumerate(keys) if k in g['fine']};user=fmt.render(g,'no_context');c=p.client()
    if old.MODEL not in {x.id for x in c.models.list().data}:raise ValueError('Model not available')
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    attempts=[];out=None
    for _ in range(2):
        start=time.time();a=None
        try:
            r=c.chat.completions.create(model=old.MODEL,temperature=0,max_tokens=MAXOUT,messages=[{'role':'system','content':system()},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'verdicts','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            a={'seconds':round(time.time()-start,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
            if a['finish_reason']!='stop':raise ValueError('Truncated output')
            obj=json.loads(a['response']);entries=obj['results']
            if set(obj)!={'results'} or not isinstance(entries,list):raise ValueError('Bad schema')
            if any(not isinstance(x,dict) or set(x)!={'id','keep'} or type(x['id'])!=int or type(x['keep'])!=bool for x in entries):raise ValueError('Bad verdict')
            good=[x for x in entries if x['id'] in input_ids];extra=[x for x in entries if x['id'] not in input_ids]; ids=[x['id'] for x in good]
            if set(ids)!=input_ids or len(ids)!=len(set(ids)):raise ValueError('Missing or repeated input ID')
            a['excluded_extra_entries']=extra;kept={keys[x['id']] for x in good if x['keep']}
            out={'status':'ok','plan_sha256':p.sha(PLAN),'input_sha256':digest(user),'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept),'attempts':attempts};break
        except Exception as err:
            if a is None:attempts.append({'seconds':round(time.time()-start,2),'error_type':type(err).__name__})
            else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
    if out is None:out={'status':'failed','plan_sha256':p.sha(PLAN),'attempts':attempts}
    OUT.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'status':out['status'],'metrics':out.get('metrics'),'usage':attempts[-1].get('usage')}),flush=True)
    if out['status']!='ok':raise SystemExit('Failed, evidence preserved')

def summary():
    groups,_=frozen();r=json.loads(OUT.read_text(encoding='utf8'))
    if r['status']!='ok' or r['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad output')
    one={tuple(k) for k in r['kept']};result={}
    for subset in ('all','val','test'):
        rows=[];many=set();tokenin=tokenout=0
        for i,g in enumerate(groups,1):
            if subset!='all' and g['split']!=subset:continue
            x=json.loads(baseline_path(i).read_text(encoding='utf8'))
            if x['status']!='ok':raise ValueError('Bad baseline')
            rows+=g['rows'];many.update(tuple(k) for k in x['kept'])
            for a in x['attempts']:
                u=a.get('usage') or {};tokenin+=u.get('prompt_tokens',0);tokenout+=u.get('completion_tokens',0)
        valid={(x['pid'],x['idx']) for x in rows};result[subset]={'one_request':p.metrics(rows,one&valid),
         'grouped_requests':p.metrics(rows,many),'grouped_input_tokens':tokenin,'grouped_output_tokens':tokenout}
    u=r['attempts'][-1].get('usage') or {};out={'plan_sha256':p.sha(PLAN),'single_usage':u,'results':result}
    print(json.dumps(out,indent=2),flush=True)
    target=DEST/'summary.json'
    if target.exists():raise ValueError('Summary exists')
    target.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='summary':summary()
