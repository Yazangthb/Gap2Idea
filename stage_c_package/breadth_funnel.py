# -*- coding: utf-8 -*-
"""Cross-domain breadth: run the CURRENT funnel (Stage A section gate + Stage B cue+head
+ Stage C LLM) on FWS (future-work, ACL/CS) and Sci-Sentence (limitation+research-gap,
related-work/CS). Mirrors production Stage B exactly (cue_label wins on type; cached
bge-small+logreg head fills the rest). Stage C = yandexgpt-5-lite.

Usage: python breadth_funnel.py [--stage-c]   (omit to run A+B only, free)
"""
import os, sys, json, time, random, argparse
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np, pandas as pd, joblib
from sentence_transformers import SentenceTransformer
from gap2idea.pipeline.gap_funnel import cue_label

ap = argparse.ArgumentParser(); ap.add_argument("--stage-c", action="store_true"); A = ap.parse_args()
HERE = Path(__file__).parent
ENCODER = "BAAI/bge-small-en-v1.5"
TARGET = {"limitation", "future_work"}
enc = SentenceTransformer(ENCODER)
clf = joblib.load(ROOT / "data" / "pipeline_head.joblib")
classes = list(clf.classes_)
print(f"head classes: {classes}\n", flush=True)

def stage_b(sents, want):
    """production rule: cue_label wins on type; else head argmax if in `want` & >=0.5."""
    X = enc.encode(sents, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
    proba = clf.predict_proba(X)
    pos = []
    for s, p in zip(sents, proba):
        rule = cue_label(s)
        j = int(np.argmax(p)); mt, mp = classes[j], float(p[j])
        if rule in want:        pos.append(1)
        elif mt in want and mp >= 0.5: pos.append(1)
        else:                   pos.append(0)
    return pos

def prf(gold, pred):
    tp = sum(1 for g, p in zip(gold, pred) if g and p)
    fp = sum(1 for g, p in zip(gold, pred) if p and not g)
    fn = sum(1 for g, p in zip(gold, pred) if g and not p)
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    return round(P, 3), round(R, 3), round(2*P*R/(P+R) if P+R else 0, 3)

def stage_c(idxs, sents, ctxs, mode):
    from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
    from gap2idea.pipeline.llm import get_llm_client
    F = os.environ["YANDEX_FOLDER_ID"].strip()
    f = LLMGapFilter(backend="api", model=f"gpt://{F}/yandexgpt-5-lite/latest",
                     mode=mode, client=get_llm_client("yandex"), context_style="inline")
    keep = {}
    items = [{"sentence": sents[i], "context": ctxs[i]} for i in idxs]
    res = []
    for k in range(0, len(items), 20):
        res.extend(f.judge_batch(items[k:k+20]))
    for i, kp in zip(idxs, res):
        keep[i] = bool(kp)
    return keep, f.n_calls

# leakage guard: sentences the head was trained on
train = {json.loads(l)["sentence"].strip() for l in
         (ROOT/"data"/"scibert_prep"/"train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}

# ================= FWS (future-work, ACL/CS) =================
df = pd.read_csv(ROOT/"data"/"external"/"fws_corpus.csv")
df = df[~df.text.str.strip().isin(train)].reset_index(drop=True)      # drop train overlap
papers = df.id.unique().tolist(); random.seed(0); random.shuffle(papers)
sel, n = [], 0
for pid in papers:
    sub = df[df.id == pid]; sel.append(sub); n += len(sub)
    if n >= 1500: break
fws = pd.concat(sel).reset_index(drop=True)
fws_sents = fws.text.tolist()
fws_gold = fws.label.astype(int).tolist()
def ctx_series(frame, col, idkey):
    out = []
    for i in range(len(frame)):
        parts = []
        for j in (i-1, i, i+1):
            if 0 <= j < len(frame) and frame.iloc[j][idkey] == frame.iloc[i][idkey]:
                parts.append(str(frame.iloc[j][col]))
        out.append(" ".join(parts))
    return out
fws_ctx = ctx_series(fws, "text", "id")
# Stage A: FWS is already conclusion/future-work sections -> gate ~pass-through
fws_b = stage_b(fws_sents, want={"future_work"})

# ================= Sci-Sentence (limitation+research-gap, CS) =================
sx = pd.read_excel(ROOT/"data"/"external"/"scisentence_repo"/"datasets"/"Based_Version.xlsx")
sx = sx[sx.Classification == "TEST"].reset_index(drop=True)
ss_sents = sx.Sentence.astype(str).tolist()
ss_gold = [1 if c in {"LIMITATION", "RESEARCH GAP"} else 0 for c in sx.Category]
ss_ctx = ss_sents[:]   # related-work: no reliable neighbor order -> sentence as its own context
ss_b = stage_b(ss_sents, want=TARGET)

def report(name, gold, b, sents, ctx, mode, prev_note):
    gp = sum(gold); bp = sum(b)
    Pa, Ra, Fa = (1.0, 1.0, 1.0)  # Stage A retention (see notes)
    Pb, Rb, Fb = prf(gold, b)
    print(f"\n### {name}  (n={len(gold)}, pos={gp}, {gp/len(gold):.1%}; {prev_note})")
    print(f"  A+B (no Stage C):   P={Pb} R={Rb} F1={Fb}   | detector positives={bp}")
    out = {"n": len(gold), "pos": gp, "AB": [Pb, Rb, Fb], "det_pos": bp}
    if A.stage_c and bp:
        idxs = [i for i, x in enumerate(b) if x]
        keep, nc = stage_c(idxs, sents, ctx, mode)
        bc = [1 if (b[i] and keep.get(i, True)) else 0 for i in range(len(b))]
        Pc, Rc, Fc = prf(gold, bc)
        print(f"  A+B+C ({mode}): P={Pc} R={Rc} F1={Fc}   | Stage C calls={nc}")
        out["ABC"] = [Pc, Rc, Fc]; out["mode"] = mode
    return out

print("="*70); print("CROSS-DOMAIN BREADTH — current funnel"); print("="*70)
res = {}
res["FWS"] = report("FWS (future_work, ACL)", fws_gold, fws_b, fws_sents, fws_ctx,
                    "validate_fws", "Stage A ~pass-through: corpus already conclusion sections")
res["SciSentence"] = report("Sci-Sentence (limitation+research-gap, CS related-work)", ss_gold, ss_b,
                    ss_sents, ss_ctx, "validate", "no section gate: related-work sentences")
json.dump(res, open(HERE/"breadth_results.json", "w"), indent=1)
print(f"\nsaved -> breadth_results.json")
