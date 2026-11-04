# -*- coding: utf-8 -*-
"""Same procedure as Sci-Sentence, on FWS (future-work, ACL/CS). In-domain bge-small+logreg
head (paper-disjoint split), recall-first threshold, + validate_fws Stage C. Report A+B and
A+B+C with keepTP/dropFP. Pass --stage-c to run Stage C."""
import os, sys, json, random, argparse
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

ap = argparse.ArgumentParser(); ap.add_argument("--stage-c", action="store_true"); A = ap.parse_args()
ENC = SentenceTransformer("BAAI/bge-small-en-v1.5")
df = pd.read_csv(ROOT/"data"/"external"/"fws_corpus.csv")
papers = df.id.unique().tolist(); random.seed(0); random.shuffle(papers)
cut = int(0.8*len(papers)); trainP, testP = set(papers[:cut]), set(papers[cut:])
tr = df[df.id.isin(trainP)]; te = df[df.id.isin(testP)].reset_index(drop=True)
# cap train for speed, keep class balance-ish
tr = pd.concat([tr[tr.label==1], tr[tr.label==0].sample(min((tr.label==0).sum(), 12000), random_state=0)])
# sample a test subset (paper-contiguous) ~1500
tp = list(testP); random.seed(1); random.shuffle(tp)
sel=[]; n=0
for pid in tp:
    sub=te[te.id==pid]; sel.append(sub); n+=len(sub)
    if n>=1500: break
te = pd.concat(sel).reset_index(drop=True)
print(f"train {len(tr)} ({int(tr.label.sum())} fw) | test {len(te)} ({int(te.label.sum())} fw, {te.label.mean():.1%})\n", flush=True)

Xtr = ENC.encode(tr.text.astype(str).tolist(), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, tr.label.astype(int).values)
Xte = ENC.encode(te.text.astype(str).tolist(), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
sc = clf.predict_proba(Xte)[:, list(clf.classes_).index(1)]
yte = te.label.astype(int).values; sents = te.text.astype(str).tolist()
def ctx(i):
    return " ".join(str(te.iloc[j].text) for j in (i-1,i,i+1) if 0<=j<len(te) and te.iloc[j].id==te.iloc[i].id)

def prf(g,p):
    tp=int(((p==1)&(g==1)).sum()); fp=int(((p==1)&(g==0)).sum()); fn=int(((p==0)&(g==1)).sum())
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    return round(P,3),round(R,3),round(2*P*R/(P+R) if P+R else 0,3)

print(f"{'in-domain head @ thr':30}{'P':>7}{'R':>7}{'F1':>7}{'det+':>7}")
for thr in (0.5,0.4,0.3,0.2):
    P,R,Fv=prf(yte,(sc>=thr).astype(int)); print(f"{'thr='+str(thr):30}{P:>7.3f}{R:>7.3f}{Fv:>7.3f}{int((sc>=thr).sum()):>7}")

if A.stage_c:
    from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
    from gap2idea.pipeline.llm import get_llm_client
    F=os.environ["YANDEX_FOLDER_ID"].strip()
    def judge(idx):
        f=LLMGapFilter(backend="api", model=f"gpt://{F}/yandexgpt-5-lite/latest", mode="validate_fws",
                       client=get_llm_client("yandex"), context_style="inline")
        items=[{"sentence":sents[i],"context":ctx(i)} for i in idx]; keep=[]
        for k in range(0,len(items),20): keep.extend(f.judge_batch(items[k:k+20]))
        return {i:bool(x) for i,x in zip(idx,keep)}, f.n_calls
    print(f"\n{'A+B+C (validate_fws) @ thr':30}{'P':>7}{'R':>7}{'F1':>7}{'keepTP':>10}{'dropFP':>10}")
    for thr in (0.3,):
        pred=(sc>=thr).astype(int); idx=[i for i in range(len(yte)) if pred[i]]
        keep,nc=judge(idx); final=pred.copy()
        for i in idx:
            if not keep[i]: final[i]=0
        P,R,Fv=prf(yte,final)
        kt=sum(1 for i in idx if keep[i] and yte[i]); dt=sum(1 for i in idx if not keep[i] and not yte[i])
        dTP=int(((pred==1)&(yte==1)).sum()); dFP=int(((pred==1)&(yte==0)).sum())
        print(f"{'thr='+str(thr):30}{P:>7.3f}{R:>7.3f}{Fv:>7.3f}{str(kt)+'/'+str(dTP):>10}{str(dt)+'/'+str(dFP):>10}  (calls {nc})")
