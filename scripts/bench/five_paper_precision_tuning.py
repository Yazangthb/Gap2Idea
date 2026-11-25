"""Precision-oriented FP8 prompt pilot on fine-tuned NLI development candidates.

Freeze/commit plan; run neutral, strict and specific prompts on five tuning
papers; commit runs and select; then run neutral and selected prompt on a
separate five-paper development group. These are historically studied LLM labels.
"""
import json
import math
import sys
import time

import numpy as np
import five_paper_context_pilot as context
import five_paper_dev_tuning as d
import five_paper_pilot as p
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc,candidates

DEST=p.ROOT/'docs/experiments/five_paper_precision_tuning_2026-11-25'
PLAN=DEST/'plan.json'; SELECTION=DEST/'selection.json'
FT=p.ROOT/'data/gold/nli_ft/nli_ft_gold_val_scores.npz'
NAMES=('neutral','strict','specific');MODEL='qwen3.8-27b'
EVIDENCE=('Be precise: keep a sentence only if it identifies a specific limitation, '
          'unresolved problem, or proposed future research direction in the sentence itself. '
          'A genuine limitation of an existing method or prior work also qualifies. '
          'Do not infer a gap merely from a general difficulty, motivation, method description, '
          'numerical outcome, positive claim, or a vague comparison; context may clarify what '
          'the sentence refers to but cannot turn a non-gap sentence into a gap. '
          'When the evidence is unclear, answer NO.')


def system(name):
    if name in ('neutral','strict'):return d.system(name)
    if name!='specific':raise ValueError('Invalid prompt')
    old=('Be STRICT. Answer YES only when the sentence itself explicitly states the limitation, the proposed direction, '
         'or the unsolved problem. Also answer NO for: a general property or difficulty of a method or domain given as '
         'background or motivation; a statement of why the authors\' approach is needed; an observed experimental outcome; '
         'and a passing remark that names no specific gap. When genuinely uncertain, answer NO.')
    text=d.system('strict')
    if text.count(old)!=1:raise ValueError('Unexpected strict rubric')
    return text.replace(old,EVIDENCE)


def data():
    splits=d.make_data();_,val=load_val('val');z=np.load(FT);typed=np.stack([evc(z,h) for h in TYPES],1)
    y=np.array([r['label']!='none' for r in val]);ok=np.array([not r['junk'] for r in val])
    ranked=np.sort(typed.max(1)[y&ok])[::-1];tau=float(ranked[math.ceil(.9*y.sum())-1]);pool=candidates(val,typed,tau)
    for split,g in splits.items():
        g['ft_candidates']={(r['id'],r['idx']) for r in pool if r['id'] in g['paper_ids']}
        if not g['ft_candidates']<=set(g['keys']):raise ValueError('Candidate mismatch')
    return splits,tau


def input_for(g):
    rows=g['rows'];keys=g['keys'];valid=g['ft_candidates'];lines=[];last=None
    for i,r in enumerate(rows):
        if keys[i] not in valid:continue
        if r['pid']!=last:last=r['pid'];lines.append('PAPER '+last)
        prev=rows[i-1]['sentence'][-800:] if i and rows[i-1]['pid']==r['pid'] else ''
        nxt=rows[i+1]['sentence'][:800] if i+1<len(rows) and rows[i+1]['pid']==r['pid'] else ''
        lines.append(f'[{i}] {r["sentence"]}\nContext: Previous: {prev} Next: {nxt}')
    return '\n'.join(lines)


def plan_for(splits,tau):
    return {'purpose':'Improve precision of fine-tuned NLI + Qwen 27B FP8 with local neighbors',
      'model':MODEL,'model_repo':'Qwen/Qwen3.8-27B-FP8','model_revision':'017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
      'selection':'highest tuning precision among prompts with >=60% overall recall, ties higher recall then prompt name; if none meet floor choose highest recall and flag floor unmet',
      'recall_floor_note':'60% is a practical exploratory guardrail chosen before scoring, not a publication requirement',
      'run_order':'tune neutral/strict/specific; freeze selection; check neutral and selected, without inspecting check labels before selection',
      'prompts':{n:{'text':system(n),'sha256':digest(system(n))} for n in NAMES},
      'groups':{k:{'paper_ids':g['paper_ids'],'rows_sha256':digest(g['rows']),
          'candidate_count':len(g['ft_candidates']),'candidate_sha256':digest(sorted(g['ft_candidates'])),
          'input_sha256':digest(input_for(g))} for k,g in splits.items()},
      'ft_threshold':tau,'ft_scores_sha256':p.sha(FT),'source_plan_sha256':p.sha(d.PLAN),
      'runner_sha256':p.sha(p.ROOT/'scripts/bench/five_paper_precision_tuning.py'),
      'model_revision_note':'FP8 service from H100, 1 request per five-paper group, 800-character original neighbors',
      'temperature':0,'seed':0,'thinking':False,'max_output_tokens':8192,
      'caveats':'Both dev groups and fine-tuned checkpoint/threshold have prior development exposure; labels are LLM-generated and not human-validated.'}


def frozen():
    d.frozen();splits,tau=data();context.check_tracked(PLAN);context.check_tracked(p.ROOT/'scripts/bench/five_paper_precision_tuning.py')
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(splits,tau):raise SystemExit('Plan or inputs changed')
    return splits


def run(split,name):
    if split not in ('tune','check') or name not in NAMES:raise SystemExit('Invalid condition')
    splits=frozen();g=splits[split]
    if split=='check':
        context.check_tracked(SELECTION);s=json.loads(SELECTION.read_text(encoding='utf8'))
        if s['plan_sha256']!=p.sha(PLAN) or name not in ('neutral',s['prompt']):raise SystemExit('Not authorized by frozen selection')
    path=DEST/f'{split}_{name}.json'
    if path.exists():raise SystemExit('Preserve existing output')
    user=input_for(g);prompt=system(name);c=p.client()
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
            obj=json.loads(item['response']);ids=obj.get('positives');keys=g['keys'];valid=g['ft_candidates']
            if (set(obj)!={'positives'} or not isinstance(ids,list) or
                any(type(i) is not int or i<0 or i>=len(keys) or keys[i] not in valid for i in ids)):
                item['validation_error']='invalid or non-input ids';continue
            item['duplicate_id_count']=len(ids)-len(set(ids));kept={keys[i] for i in ids}
            result={'status':'ok','split':split,'prompt':name,'plan_sha256':p.sha(PLAN),'input_sha256':digest(user),
                    'candidate_count':len(valid),'kept':[list(k) for k in sorted(kept)],'attempts':attempts,
                    'metrics':p.metrics(g['rows'],kept)};break
        except Exception as e:
            err={'error_type':type(e).__name__,'status_code':getattr(e,'status_code',None)}
            if item is None:attempts.append({'seconds':round(time.time()-started,3),**err})
            else:item['validation_error']=err
    if result is None:result={'status':'failed','split':split,'prompt':name,'plan_sha256':p.sha(PLAN),'attempts':attempts}
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print(json.dumps({'split':split,'prompt':name,'status':result['status'],'metrics':result.get('metrics'),
                      'usage':attempts[-1].get('usage')}),flush=True)
    if result['status']!='ok':raise SystemExit('Failed; evidence preserved')


def select():
    frozen();results=[]
    for n in NAMES:
        path=DEST/f'tune_{n}.json';context.check_tracked(path);r=json.loads(path.read_text(encoding='utf8'))
        if r['status']!='ok' or r['plan_sha256']!=p.sha(PLAN):raise SystemExit('Incomplete tuning results')
        results.append({'prompt':n,'metrics':r['metrics']})
    eligible=[x for x in results if x['metrics']['recall']>=.60]
    if eligible:eligible.sort(key=lambda x:(-x['metrics']['precision'],-x['metrics']['recall'],x['prompt']));pick=eligible[0]
    else:results.sort(key=lambda x:(-x['metrics']['recall'],-x['metrics']['precision'],x['prompt']));pick=results[0]
    if SELECTION.exists():raise SystemExit('Selection exists')
    result={'prompt':pick['prompt'],'recall_floor_met':bool(eligible),'plan_sha256':p.sha(PLAN),'tuning_results':results}
    SELECTION.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result),flush=True)


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':
        if PLAN.exists():raise SystemExit('Plan exists')
        d.frozen();splits,tau=data();DEST.mkdir(parents=True,exist_ok=True)
        PLAN.write_text(json.dumps(plan_for(splits,tau),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared; commit before inference')
    elif cmd=='run':run(sys.argv[2],sys.argv[3])
    elif cmd=='select':select()
    else:raise SystemExit('Unknown command')
