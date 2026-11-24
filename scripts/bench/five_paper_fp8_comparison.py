"""Matched 25-paper FP8 comparison: full text, zero-shot NLI, fine-tuned NLI.

Five fixed groups of five previously studied arXiv test papers. Freeze the plan,
commit a tokenization preflight, then call each method once per group. Invalid
answers are recorded, not scored as empty. This is not a new blind test.
"""
import json
import math
import random
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import five_paper_context_pilot as context
import five_paper_dev_tuning as prompts
import five_paper_pilot as p
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc

DEST=p.ROOT/'docs/experiments/five_paper_fp8_25paper_2026-11-24'
PLAN=DEST/'plan.json'; PREFLIGHT=DEST/'preflight.json'; SUMMARY=DEST/'summary.json'
METHODS=('full','zero','fine');SEED=20261124
ZS_TAU=0.35411001352163435; MODEL='qwen3.8-27b'; MAX_OUTPUT=8192
TEST_ZS=p.ROOT/'data/gold/nli_test_scores.npz'
TEST_FT=p.ROOT/'data/gold/nli_ft/nli_ft_gold_test_scores.npz'
VAL_FT=p.ROOT/'data/gold/nli_ft/nli_ft_gold_val_scores.npz'


def typed(path):
    z=np.load(path)
    return np.stack([evc(z,h) for h in TYPES],1)


def load():
    _,test=load_val('test');_,val=load_val('val')
    ids=sorted({r['pid'] for r in test if r['dom']=='arxiv_cs'});random.Random(SEED).shuffle(ids)
    if len(ids)!=28:raise ValueError('Unexpected test paper universe')
    selected=ids[:25]; groups=[selected[i:i+5] for i in range(0,25,5)]
    val_scores=typed(VAL_FT).max(1);y=np.array([r['label']!='none' for r in val]);ok=np.array([not r['junk'] for r in val])
    ranked=np.sort(val_scores[y&ok])[::-1];ft_tau=float(ranked[math.ceil(.9*y.sum())-1])
    zs=typed(TEST_ZS).max(1);ft=typed(TEST_FT).max(1)
    if len(zs)!=len(test) or len(ft)!=len(test):raise ValueError('NLI scores do not match test rows')
    lookup={(r['pid'],r['idx']):i for i,r in enumerate(test)}
    if len(lookup)!=len(test):raise ValueError('Duplicate row keys')
    result=[]
    for group in groups:
        rows=[r for pid in group for r in test if r['pid']==pid]
        keys=[(r['pid'],r['idx']) for r in rows]
        zkeys={k for k in keys if not test[lookup[k]]['junk'] and zs[lookup[k]]>=ZS_TAU}
        fkeys={k for k in keys if not test[lookup[k]]['junk'] and ft[lookup[k]]>=ft_tau}
        result.append({'paper_ids':group,'rows':rows,'keys':keys,'zero':zkeys,'fine':fkeys})
    return result,ids[25:],ft_tau


def input_for(group,method):
    rows=group['rows'];keys=group['keys'];valid=set(keys) if method=='full' else group[method]
    lines=[];last=None
    for i,r in enumerate(rows):
        if keys[i] not in valid:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        line=f'[{i}] {r["sentence"]}'
        if method!='full':
            before=rows[i-1]['sentence'][-800:] if i and rows[i-1]['pid']==r['pid'] else ''
            after=rows[i+1]['sentence'][:800] if i+1<len(rows) and rows[i+1]['pid']==r['pid'] else ''
            line+='\nContext: Previous: '+before+' Next: '+after
        lines.append(line)
    return valid,'\n'.join(lines)


def plan_for(groups,unused,ft_tau):
    files=['scripts/bench/five_paper_fp8_comparison.py','scripts/bench/five_paper_dev_tuning.py',
           'scripts/bench/nli_variants.py','scripts/bench/stage_c_nli_qwen.py',
           'src/gap2idea/pipeline/gap_funnel.py','data/gold/gap_gold_final.jsonl',
           'data/gold/splits.json','data/paper_texts.jsonl','data/gold/nli_test_scores.npz',
           'data/gold/nli_ft/nli_ft_gold_test_scores.npz','data/gold/nli_ft/nli_ft_gold_val_scores.npz']
    return {'purpose':'25-paper, one-call-per-five-papers comparison: full Qwen vs zero-shot and fine-tuned NLI candidates with local neighbors',
      'status':'exploratory: previously studied test pool; LLM-generated labels, no human adjudication',
      'selection':'shuffle sorted 28 arXiv test paper IDs with random.Random(20261124); first 25 in five consecutive groups; no label-dependent selection',
      'unused_papers':unused,'grouping_seed':SEED,'model':MODEL,'model_repo':'Qwen/Qwen3.8-27B-FP8',
      'model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a','temperature':0,'seed':0,'thinking':False,
      'max_output_tokens':MAX_OUTPUT,'system_prompt':prompts.system('neutral'),'system_sha256':digest(prompts.system('neutral')),
      'zero_threshold':ZS_TAU,'fine_threshold':ft_tau,
      'nli_note':'same Stage A cleanup and max three type NLI scores; thresholds previously fixed on validation; fine-tuned model/threshold selected on validation',
      'method_note':'full uses all sentences; zero and fine use their Stage A+B candidates, each with previous/next original sentence capped at 800 characters per side',
      'output_rule':'one JSON positives list; deduplicate valid input IDs; reject invalid IDs or incomplete output; record all attempts, never count failed calls as negatives',
      'groups':[{'paper_ids':g['paper_ids'],'row_keys_sha256':digest(g['keys']),
                 'rows_sha256':digest(g['rows']),
                 'inputs':{m:{'sentence_count':len(input_for(g,m)[0]),'input_sha256':digest(input_for(g,m)[1])} for m in METHODS}}
                for g in groups],
      'source_sha256':{f:p.sha(p.ROOT/f) for f in files}}


def check(path):context.check_tracked(path)


def frozen():
    groups,unused,tau=load()
    if not PLAN.exists() or json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(groups,unused,tau):raise SystemExit('Plan, source or input changed')
    check(PLAN);check(p.ROOT/'scripts/bench/five_paper_fp8_comparison.py')
    return groups


def preflight():
    import os
    groups=frozen()
    if PREFLIGHT.exists():raise SystemExit('Preserve preflight')
    key=os.environ.get('QWEN_API_KEY');base=os.environ.get('QWEN_BASE_URL','').removesuffix('/v1')
    if not key or not base:raise SystemExit('Set QWEN_BASE_URL and QWEN_API_KEY')
    entries=[];system=prompts.system('neutral')
    for gi,g in enumerate(groups,1):
        for method in METHODS:
            _,user=input_for(g,method)
            payload=json.dumps({'model':MODEL,'prompt':system+'\n'+user},ensure_ascii=False).encode('utf8')
            req=urllib.request.Request(base+'/tokenize',data=payload,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=120) as response:x=json.load(response)
            count=x['count'];limit=x['max_model_len'];ok=count+MAX_OUTPUT+512<=limit
            entries.append({'group':gi,'method':method,'token_count_without_chat_template':count,'max_model_len':limit,'fits_with_512_token_margin':ok})
            print('preflight',gi,method,count,'fits',ok,flush=True)
    result={'plan_sha256':p.sha(PLAN),'entries':entries,'status':'ok' if all(x['fits_with_512_token_margin'] for x in entries) else 'exceeds_context'}
    PREFLIGHT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    if result['status']!='ok':raise SystemExit('A group exceeds the context budget; no model calls made')


def run(group_no,method):
    if group_no not in range(1,6) or method not in METHODS:raise SystemExit('Invalid condition')
    groups=frozen();check(PREFLIGHT);pre=json.loads(PREFLIGHT.read_text(encoding='utf8'))
    if pre['status']!='ok' or pre['plan_sha256']!=p.sha(PLAN):raise SystemExit('Preflight mismatch')
    path=DEST/f'group_{group_no:02d}_{method}.json'
    if path.exists():raise SystemExit('Preserve existing evidence')
    group=groups[group_no-1];valid,user=input_for(group,method);keys=group['keys'];system=prompts.system('neutral');c=p.client()
    models={x.id:x for x in c.models.list().data}
    if MODEL not in models:raise SystemExit('FP8 model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None
    for _ in range(2):
        start=time.time();item=None
        try:
            resp=c.chat.completions.create(model=MODEL,temperature=0,max_tokens=MAX_OUTPUT,
                messages=[{'role':'system','content':system},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-start,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives')
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in valid for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','group':group_no,'method':method,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
                    'candidate_count':len(valid),'attempts':attempts,'kept':[list(k) for k in sorted(kept)],
                    'metrics':p.metrics(group['rows'],kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-start,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','group':group_no,'method':method,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'group':group_no,'method':method,'status':result['status'],'metrics':result.get('metrics'),
                      'attempt_count':len(attempts),'usage':attempts[-1].get('usage')}),flush=True)


def analyze():
    groups=frozen();runs={};all_rows=[r for g in groups for r in g['rows']]
    for method in METHODS:
        kept=set();status=[];input_tokens=output_tokens=0;seconds=0
        for i,g in enumerate(groups,1):
            path=DEST/f'group_{i:02d}_{method}.json';check(path);record=json.loads(path.read_text(encoding='utf8'))
            if record.get('plan_sha256')!=p.sha(PLAN):raise SystemExit('Run plan mismatch')
            status.append({'group':i,'status':record['status'],'metrics':record.get('metrics')})
            if record['status']=='ok':kept.update(tuple(k) for k in record['kept'])
            for a in record['attempts']:
                u=a.get('usage') or {};input_tokens+=u.get('prompt_tokens',0);output_tokens+=u.get('completion_tokens',0);seconds+=a.get('seconds',0)
        complete=all(x['status']=='ok' for x in status)
        result={'complete':complete,'groups':status,'total_candidates_sent':sum(len(input_for(g,method)[0]) for g in groups),
                'all_attempt_input_tokens':input_tokens,'all_attempt_output_tokens':output_tokens,'all_attempt_seconds':round(seconds,3)}
        if complete:result['metrics']=p.metrics(all_rows,kept)
        runs[method]=result
    if SUMMARY.exists():raise SystemExit('Preserve summary')
    SUMMARY.write_text(json.dumps({'plan_sha256':p.sha(PLAN),'methods':runs},indent=2)+'\n',encoding='utf8')
    for method,r in runs.items():print(method,'complete',r['complete'],'metrics',r.get('metrics'),'input_tokens',r['all_attempt_input_tokens'])


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);g,u,t=load()
        PLAN.write_text(json.dumps(plan_for(g,u,t),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Plan prepared; commit before preflight.')
    elif cmd=='preflight':preflight()
    elif cmd=='run':run(int(sys.argv[2]),sys.argv[3])
    elif cmd=='analyze':analyze()
    else:raise SystemExit('Unknown command')
