"""Matched full-text / zero-shot NLI / fine-tuned NLI on 68 studied papers."""
import json, math, os, random, sys, time, urllib.request
import numpy as np
import five_paper_pilot as p
import five_paper_dev_tuning as d
import five_paper_development_prompt as prompt
import five_paper_fp8_comparison as prior
from baseline_evidence import digest
from nli_variants import load_val

DEST=p.ROOT/'docs/experiments/five_paper_68_verdicts_2026-11-26'
PLAN=DEST/'plan.json'; PRE=DEST/'preflight.json'; METHODS=('full','zero','fine')
MAXOUT=65536; MODEL='qwen3.8-27b'; SEED=20261126
SCORES={'val':(p.ROOT/'data/gold/nli_val_scores.npz',prior.VAL_FT),
        'test':(prior.TEST_ZS,prior.TEST_FT)}

def load():
    _,val=load_val('val'); vs=prior.typed(prior.VAL_FT).max(1)
    y=np.array([r['label']!='none' for r in val]); clean=np.array([not r['junk'] for r in val])
    tau=float(np.sort(vs[y&clean])[::-1][math.ceil(.9*y.sum())-1])
    result=[]
    for split in ('val','test'):
        _,rows=load_val(split); zp,fp=SCORES[split]; zs=prior.typed(zp).max(1); fs=prior.typed(fp).max(1)
        if len(zs)!=len(rows) or len(fs)!=len(rows): raise ValueError('NLI score length mismatch')
        lookup={(r['pid'],r['idx']):i for i,r in enumerate(rows)}
        for domain in ('arxiv_cs','biomed_sal'):
            ids=sorted({r['pid'] for r in rows if r['dom']==domain}); random.Random(SEED+(split=='test')+(domain=='biomed_sal')*2).shuffle(ids)
            for pos in range(0,len(ids),5):
                pids=ids[pos:pos+5]; rr=[r for pid in pids for r in rows if r['pid']==pid]; keys=[(r['pid'],r['idx']) for r in rr]
                if len(set(keys))!=len(keys): raise ValueError('Duplicate key')
                z={k for k in keys if not rows[lookup[k]]['junk'] and zs[lookup[k]]>=prior.ZS_TAU}
                f={k for k in keys if not rows[lookup[k]]['junk'] and fs[lookup[k]]>=tau}
                result.append({'split':split,'domain':domain,'paper_ids':pids,'rows':rr,'keys':keys,'zero':z,'fine':f})
    if sum(len(x['paper_ids']) for x in result)!=68: raise ValueError('Unexpected paper count')
    return result,tau

def plan():
    groups,tau=load(); files=['scripts/bench/five_paper_68_verdicts.py','scripts/bench/five_paper_fp8_comparison.py',
     'scripts/bench/five_paper_development_prompt.py','scripts/bench/nli_variants.py','data/gold/gap_gold_final.jsonl',
     'data/gold/splits.json','data/paper_texts.jsonl','data/gold/nli_val_scores.npz','data/gold/nli_test_scores.npz',
     'data/gold/nli_ft/nli_ft_gold_val_scores.npz','data/gold/nli_ft/nli_ft_gold_test_scores.npz']
    return {'purpose':'Exploratory 68-paper matched Stage C benchmark: full paper, zero-shot NLI and fine-tuned NLI.',
      'limitations':'Development papers tuned NLI and prompt; test papers were examined in earlier experiments. Gold labels are LLM-generated and incomplete/uncertain; no new human evaluation. Domains and original splits are reported separately.',
      'group_rule':'Within each original split and domain, shuffle sorted paper IDs with random.Random(20261126 + test + 2*biomedical); consecutive groups of at most five. Final groups contain three or one papers.',
      'calls':'One request per group per method; full input is all sentences, funnel inputs are Stage A plus frozen NLI candidates with immediate original neighbors, clipped to 800 characters per side.',
      'output':'Exact one boolean verdict per numbered input; validate complete coverage, duplicates and IDs. Invalid or truncated output is a failure, never scored as negative.',
      'model':MODEL,'model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','temperature':0,'seed':0,'thinking':False,
      'max_output_tokens':MAXOUT,'prompt':prompt.prompt('verdicts'),'zero_threshold':prior.ZS_TAU,'fine_threshold':tau,
      'source_sha256':{f:p.sha(p.ROOT/f) for f in files},
      'groups':[{'split':g['split'],'domain':g['domain'],'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),
       'inputs':{m:{'count':len(prior.input_for(g,m)[0]),'sha256':digest(prior.input_for(g,m)[1])} for m in METHODS}} for g in groups]}

def frozen():
    d.check_committed(PLAN); d.check_committed(p.ROOT/'scripts/bench/five_paper_68_verdicts.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan(): raise ValueError('Plan or source changed')
    return load()[0]

def preflight():
    groups=frozen(); entries=[]; key=os.environ['QWEN_API_KEY']; base=os.environ['QWEN_BASE_URL'].removesuffix('/v1')
    for i,g in enumerate(groups,1):
        for method in METHODS:
            valid,user=prior.input_for(g,method)
            payload=json.dumps({'model':MODEL,'messages':[{'role':'system','content':prompt.prompt('verdicts')},{'role':'user','content':user}],'add_generation_prompt':True},ensure_ascii=False).encode()
            req=urllib.request.Request(base+'/tokenize',data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as r: result=json.load(r)
            count=result['count']; limit=result.get('max_model_len',262144)
            item={'group':i,'method':method,'sentences':len(valid),'input_tokens':count,'context_limit':limit,'fits':count+MAXOUT+512<=limit}
            entries.append(item); print(json.dumps(item),flush=True)
    out={'plan_sha256':p.sha(PLAN),'entries':entries,'status':'ok' if all(e['fits'] for e in entries) else 'context_exceeded'}
    if PRE.exists(): raise ValueError('Existing preflight')
    PRE.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    if out['status']!='ok': raise SystemExit('Do not run; context exceeded')

def run(groups,i,method,c):
    if i not in range(1,len(groups)+1) or method not in METHODS: raise ValueError('Invalid condition')
    target=DEST/f'group{i:02d}_{method}.json'
    if target.exists():
        previous=json.loads(target.read_text(encoding='utf8'))
        if previous.get('status')=='ok' and previous.get('plan_sha256')==p.sha(PLAN): print('existing',i,method,flush=True); return
        raise ValueError('Prior failure preserved; investigate')
    g=groups[i-1]; valid,user=prior.input_for(g,method); keys=g['keys']
    schema={'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}}
    attempts=[]; outcome=None
    for _ in range(2):
        started=time.time(); item=None
        try:
            r=c.chat.completions.create(model=MODEL,temperature=0,max_tokens=MAXOUT,
              messages=[{'role':'system','content':prompt.prompt('verdicts')},{'role':'user','content':user}],
              response_format={'type':'json_schema','json_schema':{'name':'verdicts','schema':schema,'strict':True}},
              extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-started,2),'finish_reason':r.choices[0].finish_reason,'usage':r.usage.model_dump() if r.usage else None,'response':r.choices[0].message.content}; attempts.append(item)
            if item['finish_reason']!='stop': raise ValueError('truncated')
            obj=json.loads(item['response']); entries=obj['results']; ids=[]; yes=[]
            if set(obj)!={'results'} or not isinstance(entries,list): raise ValueError('schema')
            for x in entries:
                if set(x)!={'id','keep'} or type(x['id'])!=int or type(x['keep'])!=bool: raise ValueError('bad verdict')
                ids.append(x['id'])
                if x['keep']:yes.append(x['id'])
            if len(ids)!=len(set(ids)) or any(i<0 or i>=len(keys) for i in ids) or {keys[i] for i in ids}!=valid: raise ValueError('missing, repeated or invalid ids')
            kept={keys[i] for i in yes}
            outcome={'status':'ok','group':i,'method':method,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
             'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept),'attempts':attempts}; break
        except Exception as e:
            if item is None: attempts.append({'seconds':round(time.time()-started,2),'error_type':type(e).__name__})
            else: item['error_type']=type(e).__name__; item['error']=str(e)[:200]
    if outcome is None: outcome={'status':'failed','group':i,'method':method,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    target.write_text(json.dumps(outcome,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'group':i,'method':method,'status':outcome['status'],'metrics':outcome.get('metrics'),'usage':attempts[-1].get('usage')}),flush=True)
    if outcome['status']!='ok': raise SystemExit('Invalid response, evidence saved; stop')

def run_all():
    groups=frozen(); d.check_committed(PRE); pre=json.loads(PRE.read_text(encoding='utf8'))
    if pre['status']!='ok' or pre['plan_sha256']!=p.sha(PLAN): raise ValueError('Preflight mismatch')
    c=p.client()
    if MODEL not in {x.id for x in c.models.list().data}: raise ValueError('Model unavailable')
    for i in range(1,len(groups)+1):
        for method in METHODS: run(groups,i,method,c)

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists(): raise ValueError('Plan already exists')
        DEST.mkdir(parents=True,exist_ok=True); PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8'); print('Prepared plan',flush=True)
    elif sys.argv[1]=='preflight': preflight()
    elif sys.argv[1]=='run': run_all()
