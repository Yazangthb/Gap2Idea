# -*- coding: utf-8 -*-
"""Recall-first funnel on Sci-Sentence: the 0.575 end-to-end recall is a STAGE B limit,
not Stage C. Fix Stage B (in-domain head + recall-tuned threshold) and let Stage C (which
drops ~100% of FPs) restore precision. Show the operating points.

Pass --stage-c to run the GAP-matched Stage C at the recall-tuned point."""
import os, sys, json, argparse
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np, pandas as pd, joblib
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from gap2idea.pipeline.gap_funnel import cue_label

ap = argparse.ArgumentParser(); ap.add_argument("--stage-c", action="store_true"); A = ap.parse_args()
ENC = SentenceTransformer("BAAI/bge-small-en-v1.5")
POS = {"LIMITATION", "RESEARCH GAP"}
sx = pd.read_excel(ROOT/"data"/"external"/"scisentence_repo"/"datasets"/"Based_Version.xlsx")
tr = sx[sx.Classification == "TRAINING"].reset_index(drop=True)
te = sx[sx.Classification == "TEST"].reset_index(drop=True)
Xtr = ENC.encode(tr.Sentence.astype(str).tolist(), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
ytr = np.array([1 if c in POS else 0 for c in tr.Category])
Xte = ENC.encode(te.Sentence.astype(str).tolist(), normalize_embeddings=True, batch_size=256, show_progress_bar=False)
yte = np.array([1 if c in POS else 0 for c in te.Category])
sents = te.Sentence.astype(str).tolist()
print(f"train {len(ytr)} ({ytr.sum()} gap) | test {len(yte)} ({yte.sum()} gap)\n")

def prf(g, p):
    tp=int(((p==1)&(g==1)).sum()); fp=int(((p==1)&(g==0)).sum()); fn=int(((p==0)&(g==1)).sum())
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    return round(P,3), round(R,3), round(2*P*R/(P+R) if P+R else 0,3)

# ---- frozen production head (3-class): gap score = P(lim)+P(future_work) ----
clf0 = joblib.load(ROOT/"data"/"pipeline_head.joblib"); cls = list(clf0.classes_)
gi = [i for i,c in enumerate(cls) if c in ("limitation","future_work")]
s_frozen = clf0.predict_proba(Xte)[:, gi].sum(1)
# ---- in-domain head trained on Sci-Sentence train ----
clf1 = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
s_indom = clf1.predict_proba(Xte)[:, list(clf1.classes_).index(1)]

print(f"{'detector @ threshold':34}{'P':>7}{'R':>7}{'F1':>7}{'det+':>7}")
for name, sc in [("frozen head", s_frozen), ("in-domain head", s_indom)]:
    for thr in (0.5, 0.4, 0.3, 0.2):
        pred = (sc >= thr).astype(int)
        P,R,Fv = prf(yte, pred)
        print(f"{name+' thr='+str(thr):34}{P:>7.3f}{R:>7.3f}{Fv:>7.3f}{int(pred.sum()):>7}")
    print()

if A.stage_c:
    from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
    from gap2idea.pipeline.llm import get_llm_client
    F = os.environ["YANDEX_FOLDER_ID"].strip()
    from gap2idea.pipeline.stage_c_prompts import GAP_SYS as SYS_GAP, GAP_SHOTS as SHOTS_GAP
    def judge(idxs):
        f = LLMGapFilter(backend="api", model=f"gpt://{F}/yandexgpt-5-lite/latest", mode="validate",
                         client=get_llm_client("yandex"), context_style="inline")
        f._sys, f._shots = SYS_GAP, SHOTS_GAP
        items=[{"sentence":sents[i],"context":sents[i]} for i in idxs]; keep=[]
        for k in range(0,len(items),20): keep.extend(f.judge_batch(items[k:k+20]))
        return {i:bool(k) for i,k in zip(idxs,keep)}, f.n_calls
    print(f"{'A+B+C (GAP-matched Stage C)':34}{'P':>7}{'R':>7}{'F1':>7}{'keepTP':>9}{'dropFP':>9}")
    for name, sc, thr in [("in-domain head thr=0.3", s_indom, 0.3), ("in-domain head thr=0.2", s_indom, 0.2)]:
        pred = (sc >= thr).astype(int); idx=[i for i in range(len(yte)) if pred[i]]
        keep, nc = judge(idx)
        final = pred.copy()
        for i in idx:
            if not keep[i]: final[i]=0
        P,R,Fv = prf(yte, final)
        kt=sum(1 for i in idx if keep[i] and yte[i]); dt=sum(1 for i in idx if not keep[i] and not yte[i])
        print(f"{name:34}{P:>7.3f}{R:>7.3f}{Fv:>7.3f}"
              f"{str(kt)+'/'+str(int(((pred==1)&(yte==1)).sum())):>9}{str(dt)+'/'+str(int(((pred==1)&(yte==0)).sum())):>9}  (calls {nc})")
