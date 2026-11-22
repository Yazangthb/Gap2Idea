"""Development-only test of positive and explicitly negated NLI hypotheses.

Prepare and commit the plan before scoring. The remote scorer reads label-free
sentences, never labels. Analyze selects thresholds on stratified development
half A and reports fixed-threshold results on half B, at 80%, 90%, 95% targets.
Both halves were used in historical experiments and are not a blind test.
"""
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'docs/experiments/nli_explicit_negation_2026-11-22'
PLAN=DEST/'plan.json'; INPUT=DEST/'sentences_label_free.json'
HYPOTHESES={
 'pos_lim':'This is a limitation.', 'neg_lim':'This is not a limitation.',
 'pos_fw':'This is future work.', 'neg_fw':'This is not future work.',
 'pos_gap':'This is a research gap.', 'neg_gap':'This is not a research gap.',
}
MODEL_REV='ca476cb923a8637073d4ceb0f19f7fc236e260d4'
MODEL_SHA='f56738ff6d17e2c4b28eac0969290cc5325be886a7be069ddc2c6d69128f1e2a'
FT_SHA='15e6b7146df31ef09ffb5d7ed84381a53d9aad247e99a9173759b9de2f9412b9'
TARGETS=(.8,.9,.95)


def sha(path):
    b=path.read_bytes()
    if path.suffix in ('.py','.json','.jsonl'):b=b.replace(b'\r\n',b'\n')
    return hashlib.sha256(b).hexdigest()


def data():
    sys.path.insert(0,str(ROOT/'scripts/bench'))
    from nli_variants import load_val,halves
    pids,rows=load_val('val');A=halves(pids,{r['pid']:r['dom'] for r in rows})
    return rows,sorted(A),sorted(set(pids)-A)


def plan_for(rows,A,B):
    texts=[r['sentence'] for r in rows]
    text_digest=hashlib.sha256(json.dumps(texts,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    return {'purpose':'Test explicit negation on development data only, comparing zero-shot and fine-tuned NLI',
      'models':{'zero-shot':{'revision':MODEL_REV,'weights_sha256':MODEL_SHA},
                'fine-tuned':{'base_revision':MODEL_REV,'checkpoint_sha256':FT_SHA}},
      'hypotheses':HYPOTHESES,'sentence_count':len(rows),'input_sentence_sha256':text_digest,
      'split':'Existing 17-paper stratified validation half A chooses thresholds; 17-paper half B compares frozen thresholds; both halves were examined in prior work',
      'half_A_papers':A,'half_B_papers':B,
      'methods':{
        'positive':'max over types of E_positive / (E_positive + C_positive)',
        'negative':'max over types of C_negative / (E_negative + C_negative)',
        'mean':'max over types of the mean of the positive and negative scores',
        'either':'max over types of the larger of positive and negative scores',
        'both':'max over types of the smaller of positive and negative scores'},
      'threshold_rule':'For each score on half A, highest observed threshold retaining at least 80%, 90%, or 95% of all A positives including cleanup losses; thresholds fixed before B metrics',
      'scoring':'Stage A junk filter then sentence alone; pair tokenization max 256, only sentence truncated; softmax E/N/C; epsilon 1e-9',
      'targets':list(TARGETS),'batch_size':128,'dtype':'float16 autocast','seed':0,
      'source_sha256':{f:sha(ROOT/f) for f in ['scripts/bench/nli_explicit_negation.py','scripts/bench/nli_variants.py',
        'data/gold/gap_gold_final.jsonl','data/gold/splits.json','data/paper_texts.jsonl']}}


def check_committed(path):
    rel=path.relative_to(ROOT).as_posix()
    if (subprocess.run(['git','status','--porcelain','--',rel],cwd=ROOT,capture_output=True,text=True).stdout.strip()
        or subprocess.run(['git','cat-file','-e',f'HEAD:{rel}'],cwd=ROOT,capture_output=True).returncode):
        raise SystemExit('Commit '+rel+' before scoring')


def frozen():
    rows,A,B=data()
    if not PLAN.exists() or json.loads(PLAN.read_text(encoding='utf8'))!=plan_for(rows,A,B):raise SystemExit('Plan or data changed')
    check_committed(PLAN);check_committed(ROOT/'scripts/bench/nli_explicit_negation.py')
    return rows,A,B


def prepare():
    if PLAN.exists():raise SystemExit('Plan already exists')
    DEST.mkdir(parents=True,exist_ok=True);rows,A,B=data()
    PLAN.write_text(json.dumps(plan_for(rows,A,B),indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    print('Prepared plan; commit before scoring.')


def export():
    rows,_,_=frozen()
    INPUT.write_text(json.dumps([r['sentence'] for r in rows],ensure_ascii=False),encoding='utf8')
    print('Exported',len(rows),'label-free sentences.')


def score(sentences_path,out_path,model_dir,model_kind):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer
    if model_kind not in ('zero-shot','fine-tuned'):raise SystemExit('Invalid model')
    out=Path(out_path)
    if out.exists():raise SystemExit('Preserve existing output')
    sentences=json.loads(Path(sentences_path).read_text(encoding='utf8'))
    if not isinstance(sentences,list) or len(sentences)!=11201 or any(not isinstance(s,str) for s in sentences):raise SystemExit('Unexpected input')
    checkpoint=Path(model_dir)/'model.safetensors'
    expected=MODEL_SHA
    if sha(checkpoint)!=expected:raise SystemExit('Base model weights mismatch')
    torch.manual_seed(0);tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(model_dir,local_files_only=True).cuda()
    if model_kind=='fine-tuned':
        checkpoint=Path('/home/yazan/nli_ft/out/best.pt')
        if sha(checkpoint)!=FT_SHA:raise SystemExit('Fine-tuned checkpoint mismatch')
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
    model.eval(); order=[next(i for i,l in model.config.id2label.items() if l.lower().startswith(x)) for x in ('entail','neutral','contra')]
    by_length=np.argsort([len(s) for s in sentences]); saved={};start=time.time()
    for h,hypothesis in HYPOTHESES.items():
        probs=np.zeros((len(sentences),3),dtype=np.float32)
        for offset in range(0,len(sentences),128):
            idx=by_length[offset:offset+128]
            enc=tok([sentences[i] for i in idx],[hypothesis]*len(idx),return_tensors='pt',padding=True,
                    truncation='only_first',max_length=256).to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):logits=model(**enc).logits.float()
            probs[idx]=torch.softmax(logits,-1)[:,order].cpu().numpy()
        saved[h]=probs;print(model_kind,h,'seconds',round(time.time()-start,1),flush=True)
    np.savez_compressed(out,**saved,model_kind=np.array(model_kind),sentences_sha256=np.array(
        hashlib.sha256(json.dumps(sentences,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()))
    print(model_kind,'done',flush=True)


def analyze():
    rows,A,B=frozen(); y=np.array([r['label']!='none' for r in rows]);ok=np.array([not r['junk'] for r in rows])
    in_a=np.array([r['pid'] in A for r in rows]);in_b=np.array([r['pid'] in B for r in rows]);output={}
    for model in ('zero-shot','fine-tuned'):
        path=DEST/f'{model}_dev_scores.npz';check_committed(path);z=np.load(path)
        if str(z['model_kind'])!=model or str(z['sentences_sha256'])!=json.loads(PLAN.read_text(encoding='utf8'))['input_sentence_sha256']:
            raise SystemExit('Score provenance mismatch')
        def pair(kind,t):
            a=z[f'{kind}_{t}'];return a[:,0]/(a[:,0]+a[:,2]+1e-9) if kind=='pos' else a[:,2]/(a[:,0]+a[:,2]+1e-9)
        P=np.stack([pair('pos',t) for t in ('lim','fw','gap')],1)
        N=np.stack([pair('neg',t) for t in ('lim','fw','gap')],1)
        methods={'positive':P.max(1),'negative':N.max(1),'mean':((P+N)/2).max(1),
                 'either':np.maximum(P,N).max(1),'both':np.minimum(P,N).max(1)}
        results={}
        for name,sc in methods.items():
            found={}
            for target in TARGETS:
                need=math.ceil(target*int((in_a&y).sum()));rank=np.sort(sc[in_a&ok&y])[::-1]
                if len(rank)<need:
                    found[str(target)]={'status':'unreachable_after_cleanup'};continue
                threshold=float(rank[need-1]); record={'threshold':threshold}
                for split,mask in [('A',in_a),('B',in_b)]:
                    passed=mask&ok&(sc>=threshold);tp=int((passed&y).sum()); n=int(passed.sum());total=int((mask&y).sum())
                    record[split]={'sentences_passed':n,'true_positives':tp,'positive_total':total,
                                   'precision':tp/n if n else None,'recall':tp/total,'total_sentences':int(mask.sum())}
                found[str(target)]=record
            results[name]=found
        consistency={}
        for t in ('lim','fw','gap'):
            pa=P[in_b&ok,('lim','fw','gap').index(t)];na=N[in_b&ok,('lim','fw','gap').index(t)]
            consistency[t]={'pearson':float(np.corrcoef(pa,na)[0,1]),'opposite_binary_fraction':float(((pa>=.5)!=(na>=.5)).mean())}
        output[model]={'methods':results,'positive_vs_inverted_negative':consistency}
    out=DEST/'results.json'
    if out.exists():raise SystemExit('Preserve existing results')
    out.write_text(json.dumps({'plan_sha256':sha(PLAN),'models':output},indent=2)+'\n',encoding='utf8')
    for model,result in output.items():
        for name,targets in result['methods'].items():
            for t,row in targets.items():
                if row.get('status'):print(model,name,t,row['status']);continue
                b=row['B'];print(model,name,t,'threshold',round(row['threshold'],6),'passed',b['sentences_passed'],
                                  'P',round(b['precision']*100,1) if b['precision'] is not None else None,'R',round(b['recall']*100,1))


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':prepare()
    elif cmd=='export':export()
    elif cmd=='score':score(*sys.argv[2:])
    elif cmd=='analyze':analyze()
    else:raise SystemExit('Unknown command')
