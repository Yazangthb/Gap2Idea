"""Threshold already accepted Qwen decisions on the complete development set.

No Qwen or NLI inference is run. The same original Qwen decisions are screened
afterward with precomputed fine-tuned NLI entail/(entail+contradiction) scores.
"""
import json,sys
import numpy as np
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc

DEST=p.ROOT/'docs/experiments/nli_post_qwen_development_2026-11-27'
PLAN=DEST/'plan.json';RUNNER=p.ROOT/'scripts/bench/nli_post_qwen_development.py'
BATCH=p.ROOT/'docs/experiments/complete_split_batch_comparison_2026-11-26'
FIVE=p.ROOT/'docs/experiments/five_paper_context_format_34_2026-11-26'
BASE=p.ROOT/'docs/experiments/five_paper_68_verdicts_2026-11-26'
SCORES=p.ROOT/'data/gold/nli_ft/nli_ft_gold_val_scores.npz'
FIXED=(0.003,0.005,0.01,0.02,0.05,0.10,0.20,0.50,0.90)

def load_data():
    ids,rows=load_val('val');z=np.load(SCORES);scores=np.stack([evc(z,h) for h in TYPES],1).max(1)
    if len(scores)!=len(rows):raise ValueError('Score/row mismatch')
    lookup={(r['pid'],r['idx']):float(scores[i]) for i,r in enumerate(rows)}
    if len(lookup)!=len(rows) or len(ids)!=34:raise ValueError('Unexpected data')
    return ids,rows,lookup

def baseline_paths():
    config=json.loads((BATCH/'plan.json').read_text(encoding='utf8'));out={}
    for batch in ('one','up_to_17'):
        xs=[x for x in config['groups'] if x['split']=='val' and x['batch']==batch]
        out[batch]=[BATCH/f'val_{batch}_{x["index"]:02d}_fine_surrounding.json' for x in xs]
    out['up_to_5']=[FIVE/f'group{i:02d}_surrounding.json' for i in range(1,9)]
    return out

def plan():
    ids,rows,lookup=load_data();base=json.loads((BASE/'plan.json').read_text(encoding='utf8'))
    thresholds=[base['fine_threshold'],*FIXED]
    return {'purpose':'Apply fixed fine-tuned NLI score thresholds after saved Qwen decisions; development only.',
      'model':'existing fine-tuned ModernBERT NLI; max over three typed entailment/(entailment+contradiction) scores',
      'rule':'Keep a previously Qwen-positive sentence iff its saved fine-tuned NLI score >= threshold. Qwen inputs and predictions remain fixed.',
      'baseline_threshold':base['fine_threshold'],'thresholds':thresholds,
      'groups':'Primary: original up-to-17 groups (17, 10, 5, 2). Secondary checks: original one-paper and up-to-5 groups. All use 34 development papers and surrounding context.',
      'caveats':'All development papers influenced prior model/prompt decisions. Labels are LLM-generated. No held-out confirmation, no GPU inference, no new Qwen tokens.',
      'dev_papers':ids,'rows_sha256':digest(rows),'score_sha256':p.sha(SCORES),'runner_sha256':p.sha(RUNNER),
      'batch_plan_sha256':p.sha(BATCH/'plan.json'),'five_plan_sha256':p.sha(FIVE/'plan.json'),
      'input_sha256':{name:{str(path.relative_to(p.ROOT).as_posix()):p.sha(path) for path in paths} for name,paths in baseline_paths().items()}}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or data changed')

def calculate():
    frozen();ids,rows,scores=load_data(); data={};valid=set(scores)
    for batch,paths in baseline_paths().items():
        kept=set()
        for path in paths:
            r=json.loads(path.read_text(encoding='utf8'))
            if r.get('status')!='ok':raise ValueError(f'Unsuccessful run {path}')
            expected=p.sha(FIVE/'plan.json') if batch=='up_to_5' else p.sha(BATCH/'plan.json')
            if r.get('plan_sha256')!=expected:raise ValueError(f'Run plan mismatch {path}')
            ks={tuple(k) for k in r['kept']}
            if kept&ks or not ks<=valid:raise ValueError('Overlap or invalid output IDs')
            kept|=ks
        series=[]
        for tau in json.loads(PLAN.read_text(encoding='utf8'))['thresholds']:
            filtered={k for k in kept if scores[k]>=tau};metrics=p.metrics(rows,filtered)
            domains={dom:p.metrics([r for r in rows if r['dom']==dom],filtered) for dom in ('arxiv_cs','biomed_sal')}
            series.append({'threshold':tau,'sentences_kept':len(filtered),'removed_from_qwen':len(kept)-len(filtered),'metrics':metrics,'by_domain':domains})
        data[batch]={'qwen_kept_before_postfilter':len(kept),'thresholds':series}
    out={'plan_sha256':p.sha(PLAN),'results':data}
    path=DEST/'results.json'
    if path.exists():raise ValueError('Results exist')
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for batch,result in data.items():
        print('\n',batch,flush=True)
        for r in result['thresholds']:print(r['threshold'],r['sentences_kept'],r['metrics'],flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='calculate':calculate()
    else:raise ValueError('Unknown command')
