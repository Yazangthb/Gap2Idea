"""Full development/test coverage: one-paper and up-to-17 matched groups.

Five methods share paper groups, Qwen weights, decoding, and output format. The
first 17-paper group reuses three byte-matched results from the earlier study.
Large-group packing uses label-free full-input token counts and a fixed output
budget; it never truncates papers. Prior 5-paper results remain separate.
"""
import concurrent.futures,collections,json,os,sys,time,urllib.request,threading
import five_paper_68_verdicts as base
import five_paper_context_format_34 as fmt
import seventeen_paper_comparison as prior17
import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/complete_split_batch_comparison_2026-11-26'
PLAN=DEST/'plan.json';RUNNER=p.ROOT/'scripts/bench/complete_split_batch_comparison.py'
METHODS=('full','zero','fine_current','fine_surrounding','fine_no_context')
MAXOUT=65536;MARGIN=512; io_lock=threading.Lock()

def universe():
    old=base.frozen(); out={}
    for split in ('val','test'):
        gs=[g for g in old if g['split']==split];papers=collections.defaultdict(list)
        for g in gs:
            for r in g['rows']:papers[r['pid']].append(r)
        out[split]={'papers':dict(papers),'zero':set().union(*(g['zero'] for g in gs)),
                    'fine':set().union(*(g['fine'] for g in gs))}
    return out

def assemble(u,ids):
    rows=[r for pid in ids for r in u['papers'][pid]];keys=[(r['pid'],r['idx']) for r in rows];valid=set(keys)
    return {'paper_ids':ids,'rows':rows,'keys':keys,'zero':u['zero']&valid,'fine':u['fine']&valid}

def system(n):
    text=base.prompt.prompt('verdicts'); old='There are five separate papers below.'
    if text.count(old)!=1:raise ValueError('Prompt changed')
    return text.replace(old, 'There is one paper below.' if n==1 else f'There are {n} separate papers below.')

def input_for(g,m):
    if m=='full':return base.prior.input_for(g,'full')
    if m=='zero':return g['zero'],fmt.render({**g,'fine':g['zero']},'surrounding')
    if m=='fine_current':return base.prior.input_for(g,'fine')
    if m=='fine_surrounding':return g['fine'],fmt.render(g,'surrounding')
    if m=='fine_no_context':return g['fine'],fmt.render(g,'no_context')
    raise ValueError('Bad method')

def tokenize(g,m):
    ids,text=input_for(g,m);body=json.dumps({'model':base.MODEL,'messages':[{'role':'system','content':system(len(g['paper_ids']))},{'role':'user','content':text}],'add_generation_prompt':True},ensure_ascii=False).encode()
    url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize'
    req=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+os.environ['QWEN_API_KEY'],'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as r:x=json.load(r)
    return x['count'],x.get('max_model_len',262144)

def group_ids(u):
    sorted_ids=sorted(u['papers'],key=lambda pid:(sum(len(r['sentence']) for r in u['papers'][pid]),pid))
    groups=[sorted_ids[:17]];rest=sorted_ids[17:];running=[]
    for pid in rest:
        trial=running+[pid];g=assemble(u,trial);n,limit=tokenize(g,'full')
        if n+MAXOUT+MARGIN>limit or len(trial)>17:
            if not running:raise ValueError('A single paper cannot fit')
            groups.append(running);running=[pid]
        else:running=trial
    if running:groups.append(running)
    return groups

def prepare():
    if PLAN.exists():raise ValueError('Plan exists')
    u=universe();entries=[]
    for split in ('val','test'):
        ids_sorted=sorted(u[split]['papers'])
        large=group_ids(u[split]); print('PACKING',split,[len(x) for x in large],flush=True)
        for batch,sets in [('one',[[pid] for pid in ids_sorted]),('up_to_17',large)]:
            for index,ids in enumerate(sets,1):
                g=assemble(u[split],ids)
                entries.append({'split':split,'batch':batch,'index':index,'paper_ids':ids,'rows_sha256':digest(g['rows']),
                  'inputs':{m:{'sha256':digest(input_for(g,m)[1]),'sentences':len(input_for(g,m)[0])} for m in METHODS}})
    out={'description':'Compare all 34 development and 34 test papers per method for batches of 1 and up to 17, plus reuse existing batches of up to 5.',
      'selection':'Within each split sort papers by total original sentence characters, then paper ID. First group is 17 shortest (the prior experiment). Greedily pack remaining papers in that order without exceeding full-paper context budget or 17 papers; one-paper jobs cover all papers.',
      'fairness':'Each method uses exactly the same papers, order, and groups for a given split and batch size. Full input includes all sentences; NLI methods send Stage A+B candidates, with the specified context layout.',
      'caveats':'Up-to-17 groups may contain fewer papers to fit; group composition and prompts differ from earlier up-to-5 study. Development was used for tuning, test was examined before, and labels are LLM-generated.',
      'methods':METHODS,'model':base.MODEL,'revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','temperature':0,'seed':0,'thinking':False,'max_output_tokens':MAXOUT,
      'output_validation':'Each input ID must receive exactly one boolean decision. Extra out-of-input IDs are excluded and reported; missing, duplicate, malformed or truncated real IDs cause a failure.',
      'source_plan_sha256':p.sha(base.PLAN),'runner_sha256':p.sha(RUNNER),
      'reuse_17_sha256':{f'{s}_{m}':p.sha(prior17.DEST/f'{s}_{oldm}.json') for s in ('val','test') for m,oldm in [('full','full'),('zero','zero'),('fine_surrounding','fine')]},'groups':entries}
    DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared',flush=True)

def frozen():
    d.check_committed(PLAN);d.check_committed(RUNNER);plan=json.loads(PLAN.read_text(encoding='utf8'))
    if plan['runner_sha256']!=p.sha(RUNNER) or plan['source_plan_sha256']!=p.sha(base.PLAN):raise ValueError('Sources changed')
    for name,sha in plan['reuse_17_sha256'].items():
        split,method=name.split('_',1);oldm={'full':'full','zero':'zero','fine_surrounding':'fine'}[method]
        if p.sha(prior17.DEST/f'{split}_{oldm}.json')!=sha:raise ValueError('Prior output changed')
    u=universe();groups=[]
    for x in plan['groups']:
        g=assemble(u[x['split']],x['paper_ids'])
        if digest(g['rows'])!=x['rows_sha256'] or any(digest(input_for(g,m)[1])!=x['inputs'][m]['sha256'] for m in METHODS):raise ValueError('Group changed')
        groups.append((x,g))
    return plan,groups

def preflight():
    plan,groups=frozen();out=[]
    for x,g in groups:
        for m in METHODS:
            n,limit=tokenize(g,m);out.append({'split':x['split'],'batch':x['batch'],'index':x['index'],'method':m,'input_tokens':n,'limit':limit,'fits':n+MAXOUT+MARGIN<=limit})
        print('preflight',x['split'],x['batch'],x['index'],len(x['paper_ids']),'papers',flush=True)
    result={'plan_sha256':p.sha(PLAN),'entries':out,'status':'ok' if all(z['fits'] for z in out) else 'too_large'}
    path=DEST/'preflight.json'
    if path.exists():raise ValueError('Preflight exists')
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if result['status']!='ok':raise SystemExit('Context too large; no inference')

def result_path(x,m):return DEST/f'{x["split"]}_{x["batch"]}_{x["index"]:02d}_{m}.json'

def run_job(x,g,m):
    path=result_path(x,m)
    if path.exists():
        r=json.loads(path.read_text(encoding='utf8'))
        if r.get('status')=='ok' and r.get('plan_sha256')==p.sha(PLAN):return
        raise ValueError(f'Prior failure: {path.name}')
    if x['batch']=='up_to_17' and x['index']==1 and m in ('full','zero','fine_surrounding'):
        oldm={'full':'full','zero':'zero','fine_surrounding':'fine'}[m];source=prior17.DEST/f'{x["split"]}_{oldm}.json';r=json.loads(source.read_text(encoding='utf8'))
        if r['status']!='ok' or r['input_sha256']!=digest(input_for(g,m)[1]) or system(17)!=prior17.system():raise ValueError('Prior call not byte-matched')
        out={'status':'ok','plan_sha256':p.sha(PLAN),'split':x['split'],'batch':x['batch'],'index':x['index'],'method':m,'reused_from':source.relative_to(p.ROOT).as_posix(),'source_sha256':p.sha(source),'kept':r['kept'],'metrics':r['metrics'],'attempts':r['attempts']}
        path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8');return
    valid,user=input_for(g,m);keys=g['keys'];input_ids={i for i,k in enumerate(keys) if k in valid};c=p.client()
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    attempts=[];out=None
    for _ in range(2):
        t=time.time();a=None
        try:
            r=c.chat.completions.create(model=base.MODEL,temperature=0,max_tokens=MAXOUT,messages=[{'role':'system','content':system(len(g['paper_ids']))},{'role':'user','content':user}],response_format={'type':'json_schema','json_schema':{'name':'verdicts','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            a={'seconds':round(time.time()-t,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
            if a['finish_reason']!='stop':raise ValueError('Truncated')
            obj=json.loads(a['response']);ent=obj['results']
            if set(obj)!={'results'} or not isinstance(ent,list) or any(not isinstance(z,dict) or set(z)!={'id','keep'} or type(z['id'])!=int or type(z['keep'])!=bool for z in ent):raise ValueError('Invalid schema')
            good=[z for z in ent if z['id'] in input_ids];extra=[z for z in ent if z['id'] not in input_ids];ids=[z['id'] for z in good]
            if set(ids)!=input_ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate target ID')
            a['excluded_extra_entries']=extra;kept={keys[z['id']] for z in good if z['keep']}
            out={'status':'ok','plan_sha256':p.sha(PLAN),'split':x['split'],'batch':x['batch'],'index':x['index'],'method':m,'input_sha256':digest(user),'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept),'attempts':attempts};break
        except Exception as err:
            if a is None:attempts.append({'seconds':round(time.time()-t,2),'error_type':type(err).__name__})
            else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
    if out is None:out={'status':'failed','plan_sha256':p.sha(PLAN),'split':x['split'],'batch':x['batch'],'index':x['index'],'method':m,'attempts':attempts}
    path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    with io_lock:print(json.dumps({'split':x['split'],'batch':x['batch'],'index':x['index'],'method':m,'status':out['status']}),flush=True)
    if out['status']!='ok':raise ValueError(f'Failed response: {path.name}')

def run():
    plan,groups=frozen();pre=DEST/'preflight.json';d.check_committed(pre);q=json.loads(pre.read_text(encoding='utf8'))
    if q['status']!='ok' or q['plan_sha256']!=p.sha(PLAN):raise ValueError('Preflight mismatch')
    jobs=[(x,g,m) for x,g in groups for m in METHODS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run_job,*job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            try:future.result()
            except Exception:
                for f in futures:f.cancel()
                raise

def summary():
    _,groups=frozen();result={}
    for split in ('val','test'):
        result[split]={}
        for batch in ('one','up_to_17'):
            result[split][batch]={}
            for m in METHODS:
                rows=[];kept=set();inp=outp=0;fails=[]; extras=0;papers=[]
                for x,g in groups:
                    if x['split']!=split or x['batch']!=batch:continue
                    path=result_path(x,m)
                    if not path.exists():fails.append(x['index']);continue
                    r=json.loads(path.read_text(encoding='utf8'))
                    if r.get('status')!='ok' or r.get('plan_sha256')!=p.sha(PLAN):fails.append(x['index']);continue
                    rows+=g['rows'];papers+=g['paper_ids'];kept.update(tuple(k) for k in r['kept'])
                    for a in r['attempts']:
                        u=a.get('usage') or {};inp+=u.get('prompt_tokens',0);outp+=u.get('completion_tokens',0);extras+=len(a.get('excluded_extra_entries',[]))
                result[split][batch][m]={'complete':not fails and len(papers)==34,'papers':len(papers),'missing_groups':fails,'metrics':p.metrics(rows,kept) if rows else None,'input_tokens':inp,'output_tokens':outp,'extras':extras}
    out={'plan_sha256':p.sha(PLAN),'results':result};print(json.dumps(out,indent=2),flush=True)
    if all(v['complete'] for split in result.values() for batch in split.values() for v in batch.values()):
        path=DEST/'summary.json'
        if path.exists():raise ValueError('Summary exists')
        path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='run':run()
    elif sys.argv[1]=='summary':summary()
