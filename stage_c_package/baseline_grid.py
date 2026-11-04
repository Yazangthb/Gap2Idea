# -*- coding: utf-8 -*-
"""Tier-1 baseline grid (no API, CPU) for sentence-level gap/limitation/future-work detection.
Datasets: SAL (limitation), FWS (future-work), Sci-Sentence (limitation+research-gap).
Baselines per dataset: majority, TF-IDF+LogReg, CountBinary+BernoulliNB, cue-rules, bge-small frozen+logreg.
SAL also scores the 8 encoder models already predicted in sal_test_allmodels.json (incl. fine-tuned).
Metric: positive-class P/R/F1 at native prevalence (+ SAL also @20.7%)."""
import sys, json, random, ast
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sentence_transformers import SentenceTransformer
from gap2idea.pipeline.gap_funnel import cue_label
SAL_JSON = ROOT/"stage_c_package"/"sal_criteria"/"sal_test_allmodels.json"

BGE = SentenceTransformer("BAAI/bge-small-en-v1.5")

def prf(y, p):
    y, p = np.array(y), np.array(p)
    tp=int(((p==1)&(y==1)).sum()); fp=int(((p==1)&(y==0)).sum()); fn=int(((p==0)&(y==1)).sum())
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
    return round(P,3), round(R,3), round(2*P*R/(P+R) if P+R else 0,3)

def run(name, tr_s, tr_y, te_s, te_y, cue_target):
    tr_y=np.array(tr_y); te_y=np.array(te_y)
    print(f"\n### {name}  train={len(tr_y)} ({tr_y.mean():.1%} pos) | test={len(te_y)} ({te_y.mean():.1%} pos)")
    print(f"{'baseline':34}{'P':>7}{'R':>7}{'F1':>7}")
    rows={}
    def show(tag, pred): P,R,F=prf(te_y,pred); print(f"{tag:34}{P:>7.3f}{R:>7.3f}{F:>7.3f}"); rows[tag]=[P,R,F]
    # majority
    show("majority (all-neg)", np.zeros(len(te_y)))
    # TF-IDF + LogReg
    v=TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True); Xtr=v.fit_transform(tr_s); Xte=v.transform(te_s)
    show("TF-IDF + LogReg", LogisticRegression(C=4,max_iter=2000,class_weight="balanced").fit(Xtr,tr_y).predict(Xte))
    # CountBinary + BernoulliNB (the FWS SOTA method)
    cv=CountVectorizer(ngram_range=(1,4), binary=True, min_df=2); Ctr=cv.fit_transform(tr_s); Cte=cv.transform(te_s)
    show("CountBin(1-4) + BernoulliNB", BernoulliNB().fit(Ctr,tr_y).predict(Cte))
    # cue-rules only
    show("cue-rules only", [1 if cue_label(s) in cue_target else 0 for s in te_s])
    # bge-small frozen + logreg
    Etr=BGE.encode(tr_s, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    Ete=BGE.encode(te_s, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    show("bge-small frozen + logreg", LogisticRegression(C=2,max_iter=2000,class_weight="balanced").fit(Etr,tr_y).predict(Ete))
    return rows

# ---------- SAL ----------
def parse_lim(x):
    try: return 1 if any(bool(b) for b in ast.literal_eval(x)) else 0
    except Exception: return 1 if "True" in str(x) else 0
tr=pd.read_csv(ROOT/"data"/"external"/"sal_rct"/"train.csv")
tr_s=tr.sentences.astype(str).tolist(); tr_y=[parse_lim(x) for x in tr.limitation_annotation]
js=json.load(open(SAL_JSON,encoding="utf-8"))
te_s=[r["sent"] for r in js]; te_y=[r["label"] for r in js]
sal_rows=run("SAL (limitation)", tr_s, tr_y, te_s, te_y, {"limitation"})
print("  -- encoder models already predicted on SAL test (incl. fine-tuned) --")
for m in js[0]["preds"]:
    P,R,F=prf(te_y,[r["preds"][m] for r in js]); print(f"  {m:30}{P:>7.3f}{R:>7.3f}{F:>7.3f}"); sal_rows[m]=[P,R,F]

# ---------- FWS ----------
fw=pd.read_csv(ROOT/"data"/"external"/"fws_corpus.csv")
papers=fw.id.unique().tolist(); random.seed(0); random.shuffle(papers)
te_p=set(papers[:int(len(papers)*0.2)])
fw_te=fw[fw.id.isin(te_p)]; fw_tr=fw[~fw.id.isin(te_p)]
fws_rows=run("FWS (future-work)", fw_tr.text.astype(str).tolist(), fw_tr.label.astype(int).tolist(),
             fw_te.text.astype(str).tolist(), fw_te.label.astype(int).tolist(), {"future_work"})

# ---------- Sci-Sentence ----------
sx=pd.read_excel(ROOT/"data"/"external"/"scisentence_repo"/"datasets"/"Based_Version.xlsx")
POS={"LIMITATION","RESEARCH GAP"}
tr2=sx[sx.Classification=="TRAINING"]; te2=sx[sx.Classification=="TEST"]
ss_rows=run("Sci-Sentence (limitation+gap)",
            tr2.Sentence.astype(str).tolist(), [1 if c in POS else 0 for c in tr2.Category],
            te2.Sentence.astype(str).tolist(), [1 if c in POS else 0 for c in te2.Category],
            {"limitation","future_work"})

json.dump({"SAL":sal_rows,"FWS":fws_rows,"SciSentence":ss_rows},
          open(SCR.parent/"scratchpad"/"baseline_grid_results.json","w"), indent=1) if False else None
json.dump({"SAL":sal_rows,"FWS":fws_rows,"SciSentence":ss_rows}, open(Path(__file__).parent/"baseline_grid_results.json","w"), indent=1)
print("\nsaved -> baseline_grid_results.json")
