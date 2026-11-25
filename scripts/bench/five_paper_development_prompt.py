"""Further single-request development tuning, including per-candidate output."""
import json, sys, time
import five_paper_single_prompt_tune as old
import five_paper_pilot as p
import five_paper_dev_tuning as d
from baseline_evidence import digest

DEST=p.ROOT/'docs/experiments/five_paper_development_prompt_2026-11-25'
PLAN=DEST/'plan.json'; SELECTION=DEST/'selection.json'
NAMES=('neutral','grounded','unresolved','verdicts')
CLAUSES={
 'grounded': ('Decide whether the target sentence conveys a substantive limitation, an unresolved research problem, '
  'or a concrete future investigation. A limitation of a previous method qualifies even if the new method addresses it. '
  'Do not count an ordinary method description, a standalone observation or numerical result, a hypothetical risk '
  'without an identified limitation, or a general property presented only as background. The neighboring sentences '
  'may resolve references but cannot supply the gap if the target does not convey it.'),
 'unresolved': ('Identify the problem actually stated by the target. Keep a specific limitation of this work or existing '
  'approaches, an open problem, or a future study proposed by the authors. A limitation can be a constraint or weakness '
  'even when another approach is offered as a solution. Exclude descriptions of completed methods or experimental '
  'results unless the sentence itself presents a limitation, and exclude generic background or hypothetical concerns '
  'that do not identify a research limitation.')}
OLD='Judge each sentence on the rules above, without favouring YES or NO.'
OUT='Return JSON of the form {"positives":[<id>, ...]}: the ids of every sentence you answer YES for, in increasing order. Use only ids that appear in the input. If no sentence qualifies, return {"positives":[]}.'
NEW='Return JSON of the form {"results":[{"id":<id>,"keep":<bool>}, ...]} with EXACTLY one entry for every numbered input candidate, using its original id. Set keep to true for YES and false for NO, in input order.'

def prompt(name):
    text=old.prompt('neutral')
    if name in CLAUSES:
        if text.count(OLD)!=1: raise ValueError('Unexpected neutral prompt')
        text=text.replace(OLD,CLAUSES[name])
    if name=='verdicts':
        if text.count(OUT)!=1: raise ValueError('Unexpected output instructions')
        text=text.replace(OUT,NEW)
    return text

def plan():
    groups=old.frozen()
    return {'description':'Single Qwen request per five papers; fixed fine-tuned NLI and neighbor context. Tune on groups 1-3 and check selected versus neutral on groups 4-5.',
      'limitations':'All groups have earlier development exposure, including the check groups; labels are automatic and include apparent disagreements. The selection rule is exploratory, not an independent publication test.',
      'selection':'Highest aggregate precision with overall recall >=70% on 15 tuning papers; ties higher recall then name. If none qualify choose highest recall. Reuse prior neutral results for all five groups.',
      'paper_groups':[g['paper_ids'] for g in groups], 'prior_plan_sha256':p.sha(old.PLAN),
      'prior_neutral_runs_sha256':{str(i):p.sha(old.DEST/f'group{i}_neutral.json') for i in range(1,6)},
      'prompt_texts':{n:prompt(n) for n in NAMES},'model':'qwen3.8-27b','temperature':0,'seed':0,'thinking':False,
      'max_tokens_positive_ids':8192,'max_tokens_full_verdicts':16384,
      'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_development_prompt.py')}

def frozen():
    d.check_committed(PLAN); d.check_committed(p.ROOT/'scripts/bench/five_paper_development_prompt.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan(): raise ValueError('Plan changed')
    return old.data()[0]

def path(i,n): return old.DEST/f'group{i}_neutral.json' if n=='neutral' else DEST/f'group{i}_{n}.json'

def run(i,name):
    groups=frozen(); g=groups[i-1]
    if name not in NAMES[1:] or i not in range(1,6): raise ValueError('Invalid condition')
    if i>3:
        d.check_committed(SELECTION)
        if name!=json.loads(SELECTION.read_text(encoding='utf8'))['prompt']: raise ValueError('Not selected')
    target=path(i,name)
    if target.exists(): raise ValueError('Existing result preserved')
    full=name=='verdicts'; schema=({'type':'object','additionalProperties':False,'required':['results'],'properties':{'results':{'type':'array','items':{'type':'object','additionalProperties':False,'required':['id','keep'],'properties':{'id':{'type':'integer'},'keep':{'type':'boolean'}}}}}} if full else {'type':'object','additionalProperties':False,'required':['positives'],'properties':{'positives':{'type':'array','items':{'type':'integer'}}}})
    c=p.client(); user=old.prior.input_for(g); attempts=[]; result=None
    for _ in range(2):
        start=time.time(); item=None
        try:
            resp=c.chat.completions.create(model='qwen3.8-27b',temperature=0,max_tokens=16384 if full else 8192,
              messages=[{'role':'system','content':prompt(name)},{'role':'user','content':user}],
              response_format={'type':'json_schema','json_schema':{'name':'verdicts' if full else 'positives','schema':schema,'strict':True}},
              extra_body={'seed':0,'chat_template_kwargs':{'enable_thinking':False}})
            item={'seconds':round(time.time()-start,2),'finish_reason':resp.choices[0].finish_reason,
             'usage':resp.usage.model_dump() if resp.usage else None,'response':resp.choices[0].message.content}; attempts.append(item)
            if item['finish_reason']!='stop': raise ValueError('Truncated')
            obj=json.loads(item['response']); keys=g['keys']; valid=g['ft_candidates']
            if full:
                if set(obj)!={'results'} or not isinstance(obj['results'],list): raise ValueError('Invalid schema')
                entries=obj['results']; ids=[]; yes=[]
                for x in entries:
                    if set(x)!={'id','keep'} or type(x['id'])!=int or type(x['keep'])!=bool: raise ValueError('Bad verdict')
                    ids.append(x['id'])
                    if x['keep']: yes.append(x['id'])
                if len(ids)!=len(set(ids)) or any(i<0 or i>=len(keys) for i in ids) or {keys[i] for i in ids}!=valid: raise ValueError('Missing, duplicate or invalid verdict')
                ids=yes
            else:
                if set(obj)!={'positives'} or not isinstance(obj['positives'],list): raise ValueError('Invalid schema')
                ids=obj['positives']
                if any(type(x)!=int or x<0 or x>=len(keys) or keys[x] not in valid for x in ids): raise ValueError('Invalid ID')
                item['duplicates']=len(ids)-len(set(ids))
            kept={keys[x] for x in ids}
            result={'status':'ok','group':i,'prompt':name,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
              'kept':[list(k) for k in sorted(kept)],'metrics':p.metrics(g['rows'],kept),'attempts':attempts}; break
        except Exception as e:
            if item is None: attempts.append({'seconds':round(time.time()-start,2),'error_type':type(e).__name__})
            else: item['error_type']=type(e).__name__; item['error']=str(e)[:200]
    if result is None: result={'status':'failed','group':i,'prompt':name,'attempts':attempts}
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'group':i,'prompt':name,'status':result['status'],'metrics':result.get('metrics'),'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok': raise SystemExit('Failed; evidence saved')

def aggregate(groups,indices,name):
    rows=[]; kept=set()
    for i in indices:
        r=json.loads(path(i,name).read_text(encoding='utf8'))
        expected=p.sha(old.PLAN) if name=='neutral' else p.sha(PLAN)
        if r['status']!='ok' or r['plan_sha256']!=expected: raise ValueError('Invalid result')
        rows+=groups[i-1]['rows']; kept.update(tuple(x) for x in r['kept'])
    return p.metrics(rows,kept)

def select():
    groups=frozen()
    for i in range(1,4):
        for name in NAMES: d.check_committed(path(i,name))
    results=[{'prompt':name,'metrics':aggregate(groups,range(1,4),name)} for name in NAMES]
    eligible=[r for r in results if r['metrics']['recall']>=.70]
    chosen=sorted(eligible,key=lambda x:(-x['metrics']['precision'],-x['metrics']['recall'],x['prompt']))[0] if eligible else sorted(results,key=lambda x:(-x['metrics']['recall'],-x['metrics']['precision'],x['prompt']))[0]
    if SELECTION.exists(): raise ValueError('Selection exists')
    s={'prompt':chosen['prompt'],'recall_floor_met':bool(eligible),'plan_sha256':p.sha(PLAN),'tuning':results}
    SELECTION.write_text(json.dumps(s,indent=2)+'\n',encoding='utf8'); print(json.dumps(s),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists(): raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True); PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8'); print('Prepared')
    elif sys.argv[1]=='run':run(int(sys.argv[2]),sys.argv[3])
    elif sys.argv[1]=='select':select()
    elif sys.argv[1]=='summary':
        groups=frozen(); s=json.loads(SELECTION.read_text(encoding='utf8'))
        print(json.dumps({'selection':s,'check':{n:aggregate(groups,(4,5),n) for n in dict.fromkeys(('neutral',s['prompt']))}},indent=2))
