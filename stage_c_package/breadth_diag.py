# -*- coding: utf-8 -*-
"""Diagnose Stage C on the breadth sets and test a framing-MATCHED prompt.
  FWS: dump the kept-but-gold0 sentences (are they real future-work = gold noise?).
  Sci-Sentence: compare the self-ack `validate` prompt vs a GAP-matched prompt (gaps in
  prior/existing work), report keepTP / dropFP / P / R / F1 for the precision-filter claim.
"""
import os, sys, json, random, argparse
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np, pandas as pd, joblib
from sentence_transformers import SentenceTransformer
from gap2idea.pipeline.gap_funnel import cue_label
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
from gap2idea.pipeline.llm import get_llm_client

HERE = Path(__file__).parent
ENC = SentenceTransformer("BAAI/bge-small-en-v1.5")
clf = joblib.load(ROOT/"data"/"pipeline_head.joblib"); classes = list(clf.classes_)
TARGET = {"limitation", "future_work"}
CLIENT = get_llm_client("yandex"); F = os.environ["YANDEX_FOLDER_ID"].strip()
MODEL = f"gpt://{F}/yandexgpt-5-lite/latest"

def stage_b(sents, want):
    X = ENC.encode(sents, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    pr = clf.predict_proba(X); out = []
    for s, p in zip(sents, pr):
        rule = cue_label(s); j = int(np.argmax(p)); mt, mp = classes[j], float(p[j])
        out.append(1 if (rule in want or (mt in want and mp >= 0.5)) else 0)
    return out

def judge(sents, ctxs, sys_p=None, shots=None, mode="validate"):
    f = LLMGapFilter(backend="api", model=MODEL, mode=mode, client=CLIENT, context_style="inline")
    if sys_p: f._sys, f._shots = sys_p, shots
    items = [{"sentence": s, "context": c} for s, c in zip(sents, ctxs)]
    keep = []
    for k in range(0, len(items), 20): keep.extend(f.judge_batch(items[k:k+20]))
    return [bool(x) for x in keep], f.n_calls

def metrics(gold, b, keep_pos, idxs):
    """keep_pos: list[bool] aligned to idxs (the detector positives)."""
    final = b[:]
    for i, k in zip(idxs, keep_pos):
        if not k: final[i] = 0
    tp = sum(1 for g, p in zip(gold, final) if g and p)
    fp = sum(1 for g, p in zip(gold, final) if p and not g)
    fn = sum(1 for g, p in zip(gold, final) if g and not p)
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    detTP = [i for i in idxs if gold[i]]; detFP = [i for i in idxs if not gold[i]]
    keptTP = sum(1 for i, k in zip(idxs, keep_pos) if k and gold[i])
    dropFP = sum(1 for i, k in zip(idxs, keep_pos) if not k and not gold[i])
    return dict(P=round(P,3), R=round(R,3), F1=round(2*P*R/(P+R) if P+R else 0,3),
                keepTP=f"{keptTP}/{len(detTP)}", dropFP=f"{dropFP}/{len(detFP)}")

train = {json.loads(l)["sentence"].strip() for l in
         (ROOT/"data"/"scibert_prep"/"train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

# ---------------- FWS ----------------
df = pd.read_csv(ROOT/"data"/"external"/"fws_corpus.csv")
df = df[~df.text.str.strip().isin(train)].reset_index(drop=True)
papers = df.id.unique().tolist(); random.seed(0); random.shuffle(papers)
sel, n = [], 0
for pid in papers:
    sel.append(df[df.id == pid]); n += len(df[df.id == pid])
    if n >= 1500: break
fws = pd.concat(sel).reset_index(drop=True)
fs = fws.text.tolist(); fg = fws.label.astype(int).tolist()
def ctx(frame):
    out = []
    for i in range(len(frame)):
        out.append(" ".join(str(frame.iloc[j].text) for j in (i-1,i,i+1)
                   if 0<=j<len(frame) and frame.iloc[j].id==frame.iloc[i].id))
    return out
fc = ctx(fws)
fb = stage_b(fs, {"future_work"}); fidx = [i for i,x in enumerate(fb) if x]
fkeep, fnc = judge([fs[i] for i in fidx], [fc[i] for i in fidx], mode="validate_fws")
fm = metrics(fg, fb, fkeep, fidx)
print(f"\n### FWS  Stage C=validate_fws  (calls {fnc})"); print(" ", fm)
fp = [fidx[j] for j,k in enumerate(fkeep) if k and not fg[fidx[j]]]
print(f"  kept-but-gold0 (real future-work FWS missed, or LLM error?): {len(fp)}")
for i in fp[:12]: print("   -", fs[i][:150])

# ---------------- Sci-Sentence ----------------
sx = pd.read_excel(ROOT/"data"/"external"/"scisentence_repo"/"datasets"/"Based_Version.xlsx")
sx = sx[sx.Classification == "TEST"].reset_index(drop=True)
ss = sx.Sentence.astype(str).tolist()
sg = [1 if c in {"LIMITATION","RESEARCH GAP"} else 0 for c in sx.Category]
sb = stage_b(ss, TARGET); sidx = [i for i,x in enumerate(sb) if x]

from gap2idea.pipeline.stage_c_prompts import GAP_SYS as SYS_GAP, GAP_SHOTS as SHOTS_GAP

for tag, (sp, sh, md) in {"validate (self-ack)": (None,None,"validate"),
                          "GAP-matched prompt": (SYS_GAP,SHOTS_GAP,"validate")}.items():
    keep, nc = judge([ss[i] for i in sidx], [ss[i] for i in sidx], sp, sh, md)
    m = metrics(sg, sb, keep, sidx)
    print(f"\n### Sci-Sentence  Stage C={tag}  (calls {nc})"); print(" ", m)
    if "GAP" in tag:
        fn = [sidx[j] for j,k in enumerate(keep) if not k and sg[sidx[j]]]
        print(f"  still-dropped gold gaps: {len(fn)}")
        for i in fn[:8]: print("   -", ss[i][:150])
