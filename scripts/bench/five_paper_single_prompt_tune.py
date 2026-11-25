"""Single-call prompt tuning on three development groups, then two check groups."""
import json, math, random, sys, time
import numpy as np
import five_paper_pilot as p
import five_paper_dev_tuning as d
import five_paper_precision_tuning as prior
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES, evc, candidates

DEST=p.ROOT/'docs/experiments/five_paper_single_prompt_2026-11-25'
PLAN=DEST/'plan.json'
SELECTION=DEST/'selection.json'
NAMES=('neutral','strict','contrastive','focus')
EXTRA={
 'contrastive': ('Classify the candidate sentence itself; use neighbors only to interpret it. '
  'A specific limitation of this work or prior methods, an unresolved question, or a proposed future study can be a gap. '
  'A method description, an ordinary experimental result, general background, or a positive contribution is not by itself a gap. '
  'Examples: "The method has not been evaluated in hospitals" is a gap; "We evaluate the method on three datasets" is not. '
  '"Current models fail to capture rare cases" is a gap; "Our method improves accuracy" is not.'),
 'focus': ('Some candidates may be incorrect. Include a sentence when it states a research limitation, an unresolved problem, '
  'or a future research direction, including limitations of prior work. Do not infer a gap solely from background, '
  'a method description, or a performance result. Judge the candidate itself; neighbors only clarify its meaning.')}

def prompt(name):
    return d.system(name) if name in ('neutral','strict') else d.system('neutral')+'\n\n'+EXTRA[name]

def data():
    raw=p.baseline.load_rows('gold','dev'); ids=sorted({r['pid'] for r in raw}); random.Random(20261125).shuffle(ids)
    if len(ids)<25: raise ValueError('Need 25 development papers')
    _,val=load_val('val'); z=np.load(prior.FT); typed=np.stack([evc(z,h) for h in TYPES],1)
    y=np.array([r['label']!='none' for r in val]); ok=np.array([not r['junk'] for r in val])
    tau=float(np.sort(typed.max(1)[y&ok])[::-1][math.ceil(.9*y.sum())-1]); pool=candidates(val,typed,tau)
    groups=[]
    for i in range(5):
        pids=ids[i*5:i*5+5]; rows=[r for pid in pids for r in raw if r['pid']==pid]
        keys=[(r['pid'],r['idx']) for r in rows]; valid={(r['id'],r['idx']) for r in pool if r['id'] in pids}
        if len(set(keys))!=len(keys) or not valid<=set(keys): raise ValueError('Candidate mismatch')
        groups.append({'paper_ids':pids,'rows':rows,'keys':keys,'ft_candidates':valid})
    return groups,tau

def plan():
    groups,tau=data()
    return {'description':'One request per five-paper group. Tune on groups 1-3; freeze selection; check groups 4-5.',
      'split':'shuffle sorted arxiv development paper ids using random.Random(20261125), take first 25 in groups of five',
      'selection':'Highest pooled tuning precision among prompts with at least 60% overall recall; ties higher recall then alphabetical. If none qualify, select highest recall.',
      'caveat':'Development papers and fine-tuned NLI were previously studied; labels are LLM-generated, and check groups are not an untouched test set.',
      'model':'qwen3.8-27b','model_repo':'Qwen/Qwen3.8-27B-FP8','revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
      'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192,'threshold':tau,
      'prompts':{n:prompt(n) for n in NAMES},'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_single_prompt_tune.py'),
      'ft_scores_sha256':p.sha(prior.FT),'groups':[{'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),
       'candidates':len(g['ft_candidates']),'input_sha256':digest(prior.input_for(g))} for g in groups]}

def frozen():
    prior.d.frozen(); d.check_committed(PLAN); d.check_committed(p.ROOT/'scripts/bench/five_paper_single_prompt_tune.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan(): raise ValueError('Plan or data changed')
    return data()[0]

def run(group,name):
    groups=frozen(); g=groups[group-1]
    if group>3:
        d.check_committed(SELECTION); sel=json.loads(SELECTION.read_text(encoding='utf8'))
        if name not in ('neutral',sel['prompt']): raise ValueError('Prompt not selected')
    path=DEST/f'group{group}_{name}.json'
    if path.exists(): raise ValueError('Existing evidence preserved')
    schema={'type':'object','additionalProperties':False,'required':['positives'],'properties':{'positives':{'type':'array','items':{'type':'integer'}}}}
    user=prior.input_for(g); c=p.client(); attempts=[]; result=None
    for _ in range(2):
        started=time.time(); item=None
        try:
            response=c.chat.completions.create(model='qwen3.8-27b',temperature=0,max_tokens=8192,
              messages=[{'role':'system','content':prompt(name)},{'role':'user','content':user}],
              response_format={'type':'json_schema','json_schema':{'name':'positives','schema':schema,'strict':True}},
              extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-started,2),'finish_reason':response.choices[0].finish_reason,
                  'usage':response.usage.model_dump() if response.usage else None,'response':response.choices[0].message.content}; attempts.append(item)
            obj=json.loads(item['response']); ids=obj['positives']; keys=g['keys']
            if item['finish_reason']!='stop' or set(obj)!={'positives'} or not isinstance(ids,list) or any(type(i)!=int or i<0 or i>=len(keys) or keys[i] not in g['ft_candidates'] for i in ids): raise ValueError('Invalid or truncated output')
            item['duplicates']=len(ids)-len(set(ids)); kept={keys[i] for i in ids}
            result={'status':'ok','group':group,'prompt':name,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
                    'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept),'attempts':attempts}; break
        except Exception as e:
            if item is None: attempts.append({'seconds':round(time.time()-started,2),'error_type':type(e).__name__})
            else: item['error_type']=type(e).__name__
    if result is None: result={'status':'failed','group':group,'prompt':name,'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'group':group,'prompt':name,'status':result['status'],'metrics':result.get('metrics'),'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok': raise SystemExit('Failed; evidence saved')

def aggregate(groups,indices,name):
    rows=[]; kept=set()
    for i in indices:
        path=DEST/f'group{i}_{name}.json'; r=json.loads(path.read_text(encoding='utf8'))
        if r['status']!='ok' or r['plan_sha256']!=p.sha(PLAN): raise ValueError('Incomplete output')
        rows.extend(groups[i-1]['rows']); kept.update(tuple(k) for k in r['kept'])
    return p.metrics(rows,kept)

def select():
    groups=frozen()
    for i in range(1,4):
        for name in NAMES: d.check_committed(DEST/f'group{i}_{name}.json')
    results=[{'prompt':name,'metrics':aggregate(groups,range(1,4),name)} for name in NAMES]
    eligible=[r for r in results if r['metrics']['recall']>=.60]
    rank=lambda x:(-x['metrics']['precision'],-x['metrics']['recall'],x['prompt'])
    picked=sorted(eligible,key=rank)[0] if eligible else sorted(results,key=lambda x:(-x['metrics']['recall'],-x['metrics']['precision'],x['prompt']))[0]
    selection={'prompt':picked['prompt'],'recall_floor_met':bool(eligible),'plan_sha256':p.sha(PLAN),'tuning_results':results}
    if SELECTION.exists(): raise ValueError('Selection exists')
    SELECTION.write_text(json.dumps(selection,indent=2)+'\n',encoding='utf8'); print(json.dumps(selection),flush=True)

if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':
        if PLAN.exists(): raise ValueError('Plan exists')
        d.frozen(); DEST.mkdir(parents=True,exist_ok=True); PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
        print('Prepared plan')
    elif cmd=='run': run(int(sys.argv[2]),sys.argv[3])
    elif cmd=='select': select()
    elif cmd=='summary':
        groups=frozen(); s=json.loads(SELECTION.read_text(encoding='utf8'))
        print(json.dumps({'selection':s,'check':{name:aggregate(groups,(4,5),name) for name in dict.fromkeys(('neutral',s['prompt']))}},indent=2))
    else: raise ValueError('Unknown command')
