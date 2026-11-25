"""Explore precision-oriented fine-tuned NLI thresholds on development papers.

Lock three thresholds and two new five-paper development groups first. Select
on the tuning group only, then compare selected vs original on the check group.
The fine-tuned checkpoint and original threshold used the full validation set.
"""
import json
import math
import random
import sys
import time

import numpy as np
import five_paper_context_pilot as context
import five_paper_dev_tuning as d
import five_paper_pilot as p
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc

DEST=p.ROOT/'docs/experiments/five_paper_precision_thresholds_2026-11-25'
PLAN=DEST/'plan.json';SELECTION=DEST/'selection.json'
FT=p.ROOT/'data/gold/nli_ft/nli_ft_gold_val_scores.npz';MODEL='qwen3.8-27b'
NAMES=('original','medium','high');SEED=20261125


def data():
    prior=d.make_data();excluded=set(p.IDS)|set(prior['tune']['paper_ids'])|set(prior['check']['paper_ids'])
    _,val=load_val('val');ids=sorted({r['pid'] for r in val if r['dom']=='arxiv_cs'}-excluded)
    random.Random(SEED).shuffle(ids)
    if len(ids)<10:raise ValueError('Not enough additional development papers')
    z=np.load(FT);scores=np.stack([evc(z,h) for h in TYPES],1).max(1)
    y=np.array([r['label']!='none' for r in val]);ok=np.array([not r['junk'] for r in val])
    ranked=np.sort(scores[y&ok])[::-1];original=float(ranked[math.ceil(.9*y.sum())-1])
    thresholds={'original':original,'medium':.01,'high':.05}
    lookup={(r['pid'],r['idx']):i for i,r in enumerate(val)};out={}
    for name,pids in [('tune',ids[:5]),('check',ids[5:10])]:
        rows=[r for pid in pids for r in val if r['pid']==pid];keys=[(r['pid'],r['idx']) for r in rows]
        cand={n:{k for k in keys if not val[lookup[k]]['junk'] and scores[lookup[k]]>=tau} for n,tau in thresholds.items()}
        out[name]={'paper_ids':pids,'rows':rows,'keys':keys,'candidates':cand}
    return out,thresholds,ids[10:]


def input_for(g,name):
    rows=g['rows'];keys=g['keys'];valid=g['candidates'][name];lines=[];last=None
    for i,r in enumerate(rows):
        if keys[i] not in valid:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        prev=rows[i-1]['sentence'][-800:] if i and rows[i-1]['pid']==r['pid'] else ''
        nxt=rows[i+1]['sentence'][:800] if i+1<len(rows) and rows[i+1]['pid']==r['pid'] else ''
        lines.append(f'[{i}] {r["sentence"]}\nContext: Previous: {prev} Next: {nxt}')
    return '\n'.join(lines)


def plan_for(groups,thresholds,unused):
    return {'purpose':'Test more selective fine-tuned NLI thresholds with unchanged neutral FP8 Qwen prompt and neighbors',
      'selection':'highest tuning-set final precision with >=60% overall recall; ties higher recall then name; if none meet floor choose highest recall and flag unmet',
      'recall_floor_note':'Exploratory 60% floor; not a scientific or publication requirement',
      'sample_rule':'Exclude previous 15 arXiv validation papers used in recent five-paper pilots; shuffle sorted remaining IDs with random.Random(20261125); first five tune, next five check',
      'unused_development_papers':unused,'thresholds':thresholds,'neutral_prompt_sha256':digest(d.system('neutral')),
      'model':MODEL,'model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
      'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192,
      'groups':{k:{'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),
          'candidates':{n:{'count':len(g['candidates'][n]),'keys_sha256':digest(sorted(g['candidates'][n])),
                           'input_sha256':digest(input_for(g,n))} for n in NAMES}} for k,g in groups.items()},
      'ft_scores_sha256':p.sha(FT),'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_precision_thresholds.py'),
      'prior_dev_plan_sha256':p.sha(d.PLAN),
      'caveat':'The checkpoint and original threshold were selected on all validation papers, including these; groups may have historical experiment exposure and labels are LLM-generated.'}


def frozen():
    d.frozen();groups,thresholds,unused=data();context.check_tracked(PLAN);context.check_tracked(p.ROOT/'scripts/bench/five_paper_precision_thresholds.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(groups,thresholds,unused):raise SystemExit('Plan or data changed')
    return groups


def run(split,name):
    if split not in ('tune','check') or name not in NAMES:raise SystemExit('Invalid condition')
    groups=frozen();g=groups[split]
    if split=='check':
        context.check_tracked(SELECTION);s=json.loads(SELECTION.read_text(encoding='utf8'))
        if s['plan_sha256']!=p.sha(PLAN) or name not in ('original',s['threshold_name']):raise SystemExit('Not in locked selection')
    path=DEST/f'{split}_{name}.json'
    if path.exists():raise SystemExit('Preserve existing output')
    user=input_for(g,name);prompt=d.system('neutral');c=p.client()
    if MODEL not in {x.id for x in c.models.list().data}:raise SystemExit('Expected model unavailable')
    schema={'type':'object','additionalProperties':False,'required':['positives'],
            'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    attempts=[];result=None
    for _ in range(2):
        started=time.time();item=None
        try:
            resp=c.chat.completions.create(model=MODEL,temperature=0,max_tokens=8192,
                messages=[{'role':'system','content':prompt},{'role':'user','content':user}],
                response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
                extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-started,3),'finish_reason':resp.choices[0].finish_reason,
                  'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content};attempts.append(item)
            if item['finish_reason']!='stop':continue
            obj=json.loads(item['response']);ids=obj.get('positives');keys=g['keys'];valid=g['candidates'][name]
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in valid for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            # p.metrics expects baseline row keys and labels, which load_val supplies.
            result={'status':'ok','split':split,'threshold_name':name,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
                    'candidate_count':len(valid),'kept':[list(k) for k in sorted(kept)],'attempts':attempts,
                    'metrics':p.metrics(g['rows'],kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-started,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','split':split,'threshold_name':name,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'split':split,'threshold':name,'status':result['status'],'metrics':result.get('metrics'),
                      'candidates':len(g['candidates'][name]),'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok':raise SystemExit('Failed; evidence preserved')


def select():
    frozen();results=[]
    for n in NAMES:
        path=DEST/f'tune_{n}.json';context.check_tracked(path);r=json.loads(path.read_text(encoding='utf8'))
        if r['status']!='ok' or r['plan_sha256']!=p.sha(PLAN):raise SystemExit('Incomplete tuning runs')
        results.append({'threshold_name':n,'metrics':r['metrics'],'candidate_count':r['candidate_count']})
    eligible=[x for x in results if x['metrics']['recall']>=.60]
    if eligible:eligible.sort(key=lambda x:(-x['metrics']['precision'],-x['metrics']['recall'],x['threshold_name']));pick=eligible[0]
    else:results.sort(key=lambda x:(-x['metrics']['recall'],-x['metrics']['precision'],x['threshold_name']));pick=results[0]
    if SELECTION.exists():raise SystemExit('Selection exists')
    result={'threshold_name':pick['threshold_name'],'recall_floor_met':bool(eligible),'plan_sha256':p.sha(PLAN),'tuning_results':results}
    SELECTION.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        d.frozen();groups,thresholds,unused=data();DEST.mkdir(parents=True,exist_ok=True)
        PLAN.write_text(json.dumps(plan_for(groups,thresholds,unused),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared; commit before inference')
    elif cmd=='run':run(sys.argv[2],sys.argv[3])
    elif cmd=='select':select()
    else:raise SystemExit('Unknown command')
