"""Two matched 17-paper single-request comparisons, length-selected to fit."""
import collections,json,os,sys,time,urllib.request
import five_paper_68_verdicts as base
import five_paper_context_format_34 as fmt
import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/seventeen_paper_comparison_2026-11-26'
PLAN=DEST/'plan.json';PRE=DEST/'preflight.json';RUNNER=p.ROOT/'scripts/bench/seventeen_paper_comparison.py'
METHODS=('full','zero','fine');MAXOUT=65536

def data():
    original=base.frozen();groups={}
    for split in ('val','test'):
        gs=[g for g in original if g['split']==split];papers=collections.defaultdict(list)
        for g in gs:
            for r in g['rows']:papers[r['pid']].append(r)
        ids=sorted(papers,key=lambda pid:(sum(len(r['sentence']) for r in papers[pid]),pid))[:17]
        rows=[r for pid in ids for r in papers[pid]];keys=[(r['pid'],r['idx']) for r in rows]; valid=set(keys)
        groups[split]={'paper_ids':ids,'rows':rows,'keys':keys,
          'zero':set().union(*(g['zero'] for g in gs))&valid,'fine':set().union(*(g['fine'] for g in gs))&valid}
    return groups

def system():
    s=base.prompt.prompt('verdicts');old='There are five separate papers below.'
    if s.count(old)!=1:raise ValueError('Prompt changed')
    return s.replace(old,'There are 17 separate papers below.')

def user(g,m):
    if m=='full':return base.prior.input_for(g,'full')
    return g[m],fmt.render({**g,'fine':g[m]},'surrounding')

def plan():
    gs=data()
    return {'purpose':'Compare full papers, zero-shot NLI and fine-tuned NLI with one 17-paper request per method per split.',
      'selection':'In each original split, select 17 papers with the fewest total original sentence characters; ties paper ID. Required for full-paper calls to fit. Label-independent but biased toward shorter papers.',
      'limitations':'Each group includes a mixture of arXiv and biomedical papers; report domains separately. Validation was used for development and test was examined in earlier work; labels are LLM-generated, not fully human-validated.',
      'method':'Same 17 papers and prompt for each method. Full sends all sentences; funnel methods send their original NLI candidates with surrounding original neighbors (max 800 chars per neighbor). Original zero and fine thresholds fixed.',
      'model':base.MODEL,'revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','temperature':0,'seed':0,'thinking':False,'max_output_tokens':MAXOUT,'prompt':system(),
      'validation':'Require exactly one boolean for each input ID; discard and record extra IDs only when all real IDs occur exactly once. Invalid or truncated response is failure.',
      'source_plan_sha256':p.sha(base.PLAN),'runner_sha256':p.sha(RUNNER),'format_script_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_context_format_34.py'),
      'groups':{s:{'paper_ids':g['paper_ids'],'domains':dict(collections.Counter(g['rows'][next(i for i,r in enumerate(g['rows']) if r['pid']==pid)]['dom'] for pid in g['paper_ids'])),
       'rows_sha256':digest(g['rows']),'inputs':{m:{'sentences':len(user(g,m)[0]),'sha256':digest(user(g,m)[1])} for m in METHODS}} for s,g in gs.items()}}

def frozen():
    d.check_committed(PLAN);d.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or data changed')
    return data()

def preflight():
    groups=frozen(); entries=[];key=os.environ['QWEN_API_KEY'];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize'
    for split,g in groups.items():
        for method in METHODS:
            ids,text=user(g,method);payload=json.dumps({'model':base.MODEL,'messages':[{'role':'system','content':system()},{'role':'user','content':text}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(url,data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as r:o=json.load(r)
            n=o['count'];limit=o.get('max_model_len',262144);item={'split':split,'method':method,'sentences':len(ids),'input_tokens':n,'limit':limit,'fits':n+MAXOUT+512<=limit};entries.append(item);print(json.dumps(item),flush=True)
    if PRE.exists():raise ValueError('Existing preflight')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'entries':entries,'status':'ok' if all(x['fits'] for x in entries) else 'too_large'},indent=2)+'\n',encoding='utf8')
    if not all(x['fits'] for x in entries):raise SystemExit('Context too large')

def run():
    gs=frozen();d.check_committed(PRE);pre=json.loads(PRE.read_text(encoding='utf8'))
    if pre['status']!='ok' or pre['plan_sha256']!=p.sha(PLAN):raise ValueError('Preflight mismatch')
    c=p.client()
    if base.MODEL not in {x.id for x in c.models.list().data}:raise ValueError('Model missing')
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    for split,g in gs.items():
        for method in METHODS:
            path=DEST/f'{split}_{method}.json'
            if path.exists():
                prev=json.loads(path.read_text(encoding='utf8'))
                if prev.get('status')=='ok' and prev.get('plan_sha256')==p.sha(PLAN):continue
                raise ValueError('Existing failed output; investigate')
            valid,text=user(g,method);keys=g['keys'];input_ids={i for i,k in enumerate(keys) if k in valid};attempts=[];out=None
            for _ in range(2):
                t=time.time();a=None
                try:
                    r=c.chat.completions.create(model=base.MODEL,temperature=0,max_tokens=MAXOUT,messages=[{'role':'system','content':system()},{'role':'user','content':text}],response_format={'type':'json_schema','json_schema':{'name':'verdicts','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                    a={'seconds':round(time.time()-t,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
                    if a['finish_reason']!='stop':raise ValueError('Truncated')
                    obj=json.loads(a['response']);ent=obj['results']
                    if set(obj)!={'results'} or not isinstance(ent,list) or any(not isinstance(x,dict) or set(x)!={'id','keep'} or type(x['id'])!=int or type(x['keep'])!=bool for x in ent):raise ValueError('Schema invalid')
                    good=[x for x in ent if x['id'] in input_ids];extra=[x for x in ent if x['id'] not in input_ids]; ids=[x['id'] for x in good]
                    if set(ids)!=input_ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate input ID')
                    a['excluded_extra_entries']=extra;keep={keys[x['id']] for x in good if x['keep']}
                    out={'status':'ok','split':split,'method':method,'plan_sha256':p.sha(PLAN),'input_sha256':digest(text),'kept':[list(k) for k in sorted(keep)],'metrics':p.metrics(g['rows'],keep),'attempts':attempts};break
                except Exception as err:
                    if a is None:attempts.append({'seconds':round(time.time()-t,2),'error_type':type(err).__name__})
                    else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
            if out is None:out={'status':'failed','split':split,'method':method,'plan_sha256':p.sha(PLAN),'attempts':attempts}
            path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
            print(json.dumps({'split':split,'method':method,'status':out['status'],'metrics':out.get('metrics'),'usage':attempts[-1].get('usage')}),flush=True)
            if out['status']!='ok':raise SystemExit('Failed response, preserved')

def summary():
    groups=frozen();result={}
    for s,g in groups.items():
        result[s]={}
        for m in METHODS:
            r=json.loads((DEST/f'{s}_{m}.json').read_text(encoding='utf8'))
            if r['status']!='ok' or r['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad result')
            kept={tuple(k) for k in r['kept']};u=r['attempts'][-1].get('usage') or {}
            result[s][m]={'all':p.metrics(g['rows'],kept),'by_domain':{dom:p.metrics([x for x in g['rows'] if x['dom']==dom],kept) for dom in ('arxiv_cs','biomed_sal')},'input_tokens':u.get('prompt_tokens'),'output_tokens':u.get('completion_tokens'),'sentences_passed':len(user(g,m)[0])}
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
