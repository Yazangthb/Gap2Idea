"""Development-only fusion of zero-shot and fine-tuned NLI scores.

Split papers by domain; fit score transforms and >=90% recall thresholds on
half A; score fixed versions on half B. No model inference or test data used.
"""
import json,math,random,sys
import numpy as np
import five_paper_pilot as p
import five_paper_dev_tuning as guard
from baseline_evidence import digest
from nli_variants import load_val
from stage_c_nli_qwen import TYPES,evc

DEST=p.ROOT/'docs/experiments/nli_fusion_development_2026-11-27'
PLAN=DEST/'plan.json';RUNNER=p.ROOT/'scripts/bench/nli_fusion_development.py'
ZS=p.ROOT/'data/gold/nli_val_scores.npz';FT=p.ROOT/'data/gold/nli_ft/nli_ft_gold_val_scores.npz'
SEED=20261127;FUSIONS=(0.25,0.5,0.75)

def data():
    _,rows=load_val('val'); scores=[]
    for path in (ZS,FT):
        z=np.load(path);scores.append(np.stack([evc(z,h) for h in TYPES],1).max(1))
    if any(len(x)!=len(rows) for x in scores):raise ValueError('Score mismatch')
    groups={}
    for dom in ('arxiv_cs','biomed_sal'):
        ids=sorted({r['pid'] for r in rows if r['dom']==dom});random.Random(SEED+(dom=='biomed_sal')).shuffle(ids)
        groups[dom]=ids
    a={pid for ids in groups.values() for pid in ids[:len(ids)//2]}
    b={r['pid'] for r in rows}-a
    if len(a)!=17 or len(b)!=17 or a&b:raise ValueError('Unexpected split')
    return rows,scores,a,b

def plan():
    rows,scores,a,b=data()
    return {'purpose':'Explore score fusion of zero-shot and fine-tuned NLI on 34 development papers, without new Qwen or NLI inference.',
      'split':'Within arxiv and biomedical separately shuffle sorted IDs with random.Random(20261127 + is_biomedical); first half A, rest B.',
      'score':'max over 3 type-specific entail/(entail+contradiction) scores; Stage A junk excluded.',
      'fusion':'Empirical percentile of each model score relative to Stage-A-passing half-A scores, then weighted mean. Weights 0.25, 0.5, 0.75 on fine-tuned model.',
      'thresholds':'For each model/fusion choose the highest score threshold with >=90% recall on all half-A gap labels including Stage A losses; freeze and evaluate on half B. OR/AND combine the two separately thresholded models.',
      'comparisons':'Report Stage B recall, precision, F1 and number of candidates on each half; Qwen was not rerun, so these are not end-to-end metrics.',
      'caveats':'Fine-tuned checkpoint and prior threshold selection used the entire validation pool, and these papers have prior development exposure; half B is not an independent holdout. Labels are LLM-generated.',
      'fine_weights':list(FUSIONS),'paper_ids_A':sorted(a),'paper_ids_B':sorted(b),'rows_sha256':digest(rows),
      'scores_sha256':{'zero':p.sha(ZS),'fine':p.sha(FT)},'runner_sha256':p.sha(RUNNER)}

def frozen():
    guard.check_committed(PLAN);guard.check_committed(RUNNER)
    if json.loads(PLAN.read_text(encoding='utf8'))!=plan():raise ValueError('Plan or data changed')

def calculate():
    frozen();rows,(zs,ft),a,b=data();clean=np.array([not r['junk'] for r in rows]);y=np.array([r['label']!='none' for r in rows]);ma=np.array([r['pid'] in a for r in rows]);mb=~ma
    def percentile(x):
        sorted_train=np.sort(x[ma&clean]);return np.searchsorted(sorted_train,x,side='right')/len(sorted_train)
    rz,rf=percentile(zs),percentile(ft);models={'zero':zs,'fine':ft}
    for w in FUSIONS:models[f'fusion_fine_{w:g}']=(1-w)*rz+w*rf
    def choose(x):
        ordered=np.sort(x[ma&clean&y])[::-1];need=math.ceil(.9*int((ma&y).sum()))
        if len(ordered)<need:raise ValueError('90% recall impossible after stage A')
        return float(ordered[need-1])
    taus={name:choose(x) for name,x in models.items()};passes={name:clean&(x>=taus[name]) for name,x in models.items()}
    passes['union']=passes['zero']|passes['fine'];passes['intersection']=passes['zero']&passes['fine']
    def metrics(mask,sub):
        keep=mask&sub;tp=int((keep&y).sum());fp=int((keep&~y).sum());positives=int((sub&y).sum());precision=tp/(tp+fp) if tp+fp else 0;recall=tp/positives if positives else 0
        return {'sentences_passed':tp+fp,'tp':tp,'fp':fp,'positives':positives,'precision':round(precision,4),'recall':round(recall,4),'f1':round(2*precision*recall/(precision+recall),4) if precision+recall else 0}
    result={name:{'threshold':taus.get(name),'A':metrics(mask,ma),'B':metrics(mask,mb)} for name,mask in passes.items()}
    out={'plan_sha256':p.sha(PLAN),'results':result};path=DEST/'results.json'
    if path.exists():raise ValueError('Results exist')
    path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    for name,x in result.items():print(name,'A',x['A'],'B',x['B'],flush=True)

if __name__=='__main__':
    if sys.argv[1]=='prepare':
        if PLAN.exists():raise ValueError('Plan exists')
        DEST.mkdir(parents=True,exist_ok=True);PLAN.write_text(json.dumps(plan(),indent=2,ensure_ascii=False)+'\n',encoding='utf8');print('Prepared')
    elif sys.argv[1]=='calculate':calculate()
    else:raise ValueError('Unknown command')
