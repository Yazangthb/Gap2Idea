"""Frozen 75/25 NLI fusion on test, then matched Qwen and full-paper comparison."""
import json,os,sys,time,urllib.request
import numpy as np
import nli_fusion_development as dev
import complete_split_batch_comparison as batch
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc

DEST=p.ROOT/'docs/experiments/nli_fusion_qwen_test_2026-11-27'
PLAN=DEST/'plan.json';PRE=DEST/'preflight.json';RUNNER=p.ROOT/'scripts/bench/nli_fusion_qwen_test.py'
TZ=p.ROOT/'data/gold/nli_test_scores.npz';TF=p.ROOT/'data/gold/nli_ft/nli_ft_gold_test_scores.npz'
DEVRESULT=dev.DEST/'results.json';WEIGHT=.75;MAXOUT=65536

def data():
    dev.frozen();vr,(vz,vf),a,_=dev.data();select=np.array([r['pid'] in a and not r['junk'] for r in vr])
    _,rows=load_val('test');test=[]
    for path in (TZ,TF):
        z=np.load(path);test.append(np.stack([evc(z,h) for h in TYPES],1).max(1))
    if any(len(x)!=len(rows) for x in test):raise ValueError('Test score mismatch')
    z,f=test;fz=np.sort(vz[select]);ff=np.sort(vf[select]);score=.25*(np.searchsorted(fz,z,side='right')/len(fz))+.75*(np.searchsorted(ff,f,side='right')/len(ff))
    saved=json.loads(DEVRESULT.read_text(encoding='utf8'))['results'];tau=saved['fusion_fine_0.75']['threshold'];fine_tau=saved['fine']['threshold'];zero_tau=saved['zero']['threshold']
    lookup={(r['pid'],r['idx']):i for i,r in enumerate(rows)}
    if len(lookup)!=len(rows):raise ValueError('Duplicate test keys')
    clean=np.array([not r['junk'] for r in rows]); masks={'fusion':clean&(score>=tau),'fine_devtune':clean&(f>=fine_tau),'zero_devtune':clean&(z>=zero_tau)}
    sets={name:{(r['pid'],r['idx']) for i,r in enumerate(rows) if mask[i]} for name,mask in masks.items()}
    _,allgroups=batch.frozen();groups=[]
    for x,g in allgroups:
        if x['split']=='test' and x['batch']=='up_to_17':groups.append((x,{**g,'fine':sets['fusion']&set(g['keys'])}))
    if sum(len(x['paper_ids']) for x,g in groups)!=34:raise ValueError('Missing test papers')
    return rows,sets,groups,{'fusion':tau,'fine_devtune':fine_tau,'zero_devtune':zero_tau}

def user(g):return batch.input_for(g,'fine_surrounding')[1]

def plan():
    rows,sets,groups,taus=data()
    return {'purpose':'Frozen 75% fine-tuned / 25% zero-shot NLI fusion on all 34 test papers, then Qwen; compare with saved original fine-NLI funnel and full papers.',
      'definition':'Scores are max of typed entail/(entail+contradiction), transformed to empirical percentiles of Stage-A-passing development half-A scores, combined 75% fine 25% zero. Threshold from previously frozen development half A.',
      'selection_note':'75/25 was selected after inspecting development half B; test was examined in earlier experiments. No independent holdout. Labels are LLM-generated and not fully human-validated.',
      'baseline_note':'Saved Qwen fine baseline uses original full-development-selected NLI threshold, not half-A threshold. Half-A single-model Stage B metrics are reported separately; no new Qwen baseline.',
      'qwen':'one request per same four original groups (17,10,5,2), surrounding original neighbors 800-char caps, same model, prompt, decoding and per-input boolean schema as original benchmark.',
      'model':batch.base.MODEL,'revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','temperature':0,'seed':0,'thinking':False,'max_output_tokens':MAXOUT,'thresholds':taus,
      'runner_sha256':p.sha(RUNNER),'dev_plan_sha256':p.sha(dev.PLAN),'dev_results_sha256':p.sha(DEVRESULT),
      'test_scores_sha256':{'zero':p.sha(TZ),'fine':p.sha(TF)},'batch_plan_sha256':p.sha(batch.PLAN),'rows_sha256':digest(rows),
      'stage_b_key_sha256':{k:digest(sorted(v)) for k,v in sets.items()},
      'groups':[{'index':x['index'],'paper_ids':g['paper_ids'],'candidate_count':len(g['fine']),'candidate_sha256':digest(sorted(g['fine'])),'input_sha256':digest(user(g)),
        'baseline_sha256':{m:p.sha(batch.result_path(x,m)) for m in ('full','fine_surrounding')}} for x,g in groups]}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or source changed')
    return data()

def preflight():
    _,_,groups,_=frozen();entries=[];url=os.environ['QWEN_BASE_URL'].removesuffix('/v1')+'/tokenize';key=os.environ['QWEN_API_KEY']
    for x,g in groups:
        body=json.dumps({'model':batch.base.MODEL,'messages':[{'role':'system','content':batch.system(len(g['paper_ids']))},{'role':'user','content':user(g)}],'add_generation_prompt':True},ensure_ascii=False).encode()
        req=urllib.request.Request(url,data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as r:o=json.load(r)
        count=o['count'];limit=o.get('max_model_len',262144);item={'index':x['index'],'candidates':len(g['fine']),'input_tokens':count,'limit':limit,'fits':count+MAXOUT+512<=limit};entries.append(item);print(json.dumps(item),flush=True)
    if PRE.exists():raise ValueError('Preflight exists')
    PRE.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'status':'ok' if all(x['fits'] for x in entries) else 'too_large','entries':entries},indent=2)+'\n',encoding='utf8')
    if not all(x['fits'] for x in entries):raise SystemExit('Context too large')

def run():
    _,_,groups,_=frozen();guard.check_committed(PRE);pre=json.loads(PRE.read_text(encoding='utf8'))
    if pre['status']!='ok' or pre['plan_sha256']!=p.sha(PLAN):raise ValueError('Bad preflight')
    c=p.client();schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    for x,g in groups:
        path=DEST/f'group{x["index"]:02d}.json'
        if path.exists():
            r=json.loads(path.read_text(encoding='utf8'))
            if r.get('status')=='ok' and r.get('plan_sha256')==p.sha(PLAN):continue
            raise ValueError('Existing failed output; investigate')
        keys=g['keys'];input_ids={i for i,k in enumerate(keys) if k in g['fine']};text=user(g);attempts=[];out=None
        for _ in range(2):
            start=time.time();a=None
            try:
                r=c.chat.completions.create(model=batch.base.MODEL,temperature=0,max_tokens=MAXOUT,messages=[{'role':'system','content':batch.system(len(g['paper_ids']))},{'role':'user','content':text}],response_format={'type':'json_schema','json_schema':{'name':'verdicts','schema':schema,'strict':True}},extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
                a={'seconds':round(time.time()-start,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content};attempts.append(a)
                if a['finish_reason']!='stop':raise ValueError('Truncated')
                obj=json.loads(a['response']);ent=obj['results']
                if set(obj)!={'results'} or not isinstance(ent,list) or any(not isinstance(z,dict) or set(z)!={'id','keep'} or type(z['id'])!=int or type(z['keep'])!=bool for z in ent):raise ValueError('Invalid schema')
                good=[z for z in ent if z['id'] in input_ids];extra=[z for z in ent if z['id'] not in input_ids];ids=[z['id'] for z in good]
                if set(ids)!=input_ids or len(ids)!=len(set(ids)):raise ValueError('Missing or duplicate real ID')
                a['excluded_extra_entries']=extra;kept={keys[z['id']] for z in good if z['keep']}
                out={'status':'ok','plan_sha256':p.sha(PLAN),'group':x['index'],'input_sha256':digest(text),'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept),'attempts':attempts};break
            except Exception as err:
                if a is None:attempts.append({'seconds':round(time.time()-start,2),'error_type':type(err).__name__})
                else:a['error_type']=type(err).__name__;a['error']=str(err)[:200]
        if out is None:out={'status':'failed','plan_sha256':p.sha(PLAN),'group':x['index'],'attempts':attempts}
        path.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        print(json.dumps({'group':x['index'],'status':out['status'],'metrics':out.get('metrics'),'usage':attempts[-1].get('usage')}),flush=True)
        if out['status']!='ok':raise SystemExit('Failed; evidence saved')

def summary():
    rows,sets,groups,taus=frozen();results={'stage_b':{},'end_to_end':{}}
    for n,keep in sets.items():results['stage_b'][n]={'sentences_passed':len(keep),'metrics':p.metrics(rows,keep)}
    original=set().union(*(g['fine'] for x,g in batch.frozen()[1] if x['split']=='test' and x['batch']=='up_to_17'))
    results['stage_b']['fine_original']={'sentences_passed':len(original),'metrics':p.metrics(rows,original)}
    for method in ('fusion','fine_surrounding','full'):
        kept=set();tin=tout=0;per=[]
        for x,g in groups:
            path=(DEST/f'group{x["index"]:02d}.json') if method=='fusion' else batch.result_path(x,method)
            r=json.loads(path.read_text(encoding='utf8'));expected=p.sha(PLAN) if method=='fusion' else p.sha(batch.PLAN)
            if r['status']!='ok' or r['plan_sha256']!=expected:raise ValueError('Incomplete output')
            k={tuple(z) for z in r['kept']};kept|=k;per.append({'group':x['index'],'metrics':p.metrics(g['rows'],k)})
            for a in r['attempts']:
                u=a.get('usage') or {};tin+=u.get('prompt_tokens',0);tout+=u.get('completion_tokens',0)
        results['end_to_end'][method]={'kept':len(kept),'metrics':p.metrics(rows,kept),'by_domain':{dom:p.metrics([r for r in rows if r['dom']==dom],kept) for dom in ('arxiv_cs','biomed_sal')},'input_tokens':tin,'output_tokens':tout,'groups':per}
    out={'plan_sha256':p.sha(PLAN),'thresholds':taus,'results':results};print(json.dumps(out,indent=2),flush=True)
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
