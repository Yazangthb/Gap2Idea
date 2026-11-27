"""Evaluate two development-selected NLI postfilters on existing test outputs."""
import json,sys
import numpy as np
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc

DEST=p.ROOT/'docs/experiments/nli_post_qwen_test_2026-11-27'
PLAN=DEST/'plan.json';RUNNER=p.ROOT/'scripts/bench/nli_post_qwen_test.py'
BATCH=p.ROOT/'docs/experiments/complete_split_batch_comparison_2026-11-26'
BASE=p.ROOT/'docs/experiments/five_paper_68_verdicts_2026-11-26'
DEV=p.ROOT/'docs/experiments/nli_post_qwen_development_2026-11-27'
SCORES=p.ROOT/'data/gold/nli_ft/nli_ft_gold_test_scores.npz'
THRESHOLDS=(0.02,0.1)

def data():
    ids,rows=load_val('test');z=np.load(SCORES);arr=np.stack([evc(z,h) for h in TYPES],1).max(1)
    if len(arr)!=len(rows) or len(ids)!=34:raise ValueError('Unexpected test data')
    scores={(r['pid'],r['idx']):float(arr[i]) for i,r in enumerate(rows)}
    if len(scores)!=len(rows):raise ValueError('Duplicate keys')
    return ids,rows,scores

def paths():
    plan=json.loads((BATCH/'plan.json').read_text(encoding='utf8'))
    gs=[x for x in plan['groups'] if x['split']=='test' and x['batch']=='up_to_17']
    if sorted(len(x['paper_ids']) for x in gs)!=[2,5,10,17]:raise ValueError('Unexpected grouping')
    return {m:[BATCH/f'test_up_to_17_{x["index"]:02d}_{m}.json' for x in gs] for m in ('full','fine_surrounding')}

def plan():
    ids,rows,_=data()
    return {'purpose':'Evaluate user-selected post-Qwen thresholds 0.02 and 0.1 against existing full-paper Qwen on all 34 test papers.',
      'rule':'Filter previously Qwen-positive fine-NLI candidates by their existing fine-tuned NLI score >= threshold; Qwen inputs and outputs are unchanged.',
      'thresholds':list(THRESHOLDS),'score':'max over three typed entailment/(entailment+contradiction) scores; not a calibrated probability',
      'grouping':'Same original up-to-17 groups of 17, 10, 5 and 2 papers for full-paper and funnel; 28 arxiv and 6 biomedical papers.',
      'caveats':'Test papers were examined during earlier experiments; the LLM-generated labels are not human validated. Exploratory, no new inference; no untouched confirmation.',
      'papers':ids,'rows_sha256':digest(rows),'scores_sha256':p.sha(SCORES),'runner_sha256':p.sha(RUNNER),
      'batch_plan_sha256':p.sha(BATCH/'plan.json'),'development_plan_sha256':p.sha(DEV/'plan.json'),'development_results_sha256':p.sha(DEV/'results.json'),
      'input_sha256':{m:{path.relative_to(p.ROOT).as_posix():p.sha(path) for path in pp} for m,pp in paths().items()}}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or data changed')

def calculate():
    frozen();_,rows,scores=data();valid=set(scores);kept={};usage={}
    for m,pp in paths().items():
        selected=set();ti=to=0
        for path in pp:
            r=json.loads(path.read_text(encoding='utf8'))
            if r.get('status')!='ok' or r.get('plan_sha256')!=p.sha(BATCH/'plan.json'):raise ValueError(f'Invalid output: {path}')
            ks={tuple(x) for x in r['kept']}
            if not ks<=valid or selected&ks:raise ValueError('Invalid or overlapping sentence keys')
            selected|=ks
            for a in r['attempts']:
                u=a.get('usage') or {};ti+=u.get('prompt_tokens',0);to+=u.get('completion_tokens',0)
        kept[m]=selected;usage[m]={'input_tokens':ti,'output_tokens':to}
    result={}
    for name,selected in [('full_paper',kept['full']),('funnel_original',kept['fine_surrounding'])]+[(f'funnel_{tau:g}',{k for k in kept['fine_surrounding'] if scores[k]>=tau}) for tau in THRESHOLDS]:
        domains={dom:p.metrics([r for r in rows if r['dom']==dom],selected) for dom in ('arxiv_cs','biomed_sal')}
        result[name]={'sentences_kept':len(selected),'metrics':p.metrics(rows,selected),'by_domain':domains,
          'removed_from_qwen':len(kept['fine_surrounding'])-len(selected) if name!='full_paper' else None}
    out={'plan_sha256':p.sha(PLAN),'results':result,'usage':usage}
    path=DEST/'results.json'
    if path.exists():raise ValueError('Results exist')
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='calculate':calculate()
    else:raise ValueError('Unknown command')
