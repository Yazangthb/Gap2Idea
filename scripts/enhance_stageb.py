"""Enhance Stage B on the LimGen benchmark: threshold tuning + stacking + more-data fine-tune.

Targets the recall gap vs fine-tuned DistilBERT (0.629) measured in bench_limgen.py.
All on LimGen's own train/test split, leakage-clean. Results are appended to the
report FILE as each method finishes (orphan-resilient), and printed unbuffered.

  baseline   bge+logreg @0.5            (the current Stage B)
  #2 thresh  bge+logreg, F1-tuned cut on val
  #3 stack   meta-logreg over [bge proba, tfidf proba, cue flag]
  #1 ft+data DistilBERT fine-tuned on MORE LimGen data

    python -u scripts/enhance_stageb.py --cap-pos 2000 --test-papers 150 --ft-epochs 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from gap2idea.pipeline.gap_funnel import cue_label  # noqa: E402
from bench_limgen import build_xy, fetch, prf, m_bert_ft  # noqa: E402

REPORT = ROOT / "docs/experiments/limgen_enhanced.md"


def best_threshold(proba, y):
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        _, _, f = prf(y, (proba >= t).astype(int))
        if f > best_f:
            best_f, best_t = f, t
    return round(float(best_t), 3)


def emit(name, P, R, F, note=""):
    line = f"| {name} | {P} | {R} | {F} | {note} |"
    print(f">>> {name:42} P={P} R={R} F1={F}  {note}", flush=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-pos", type=int, default=2000)
    ap.add_argument("--train-papers", type=int, default=2500)
    ap.add_argument("--test-papers", type=int, default=150)
    ap.add_argument("--ft-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    REPORT.write_text(
        "# LimGen Stage-B enhancements (limitation detection, held-out test)\n\n"
        "Targets the recall gap vs fine-tuned DistilBERT (0.629). LimGen own split, "
        "leakage-clean. Rows appended as computed.\n\n"
        "| method | precision | recall | F1 | note |\n|---|---|---|---|---|\n",
        encoding="utf-8")

    tr_s, tr_y = build_xy(fetch("train.jsonl"), args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(fetch("test.jsonl"), args.test_papers, None, args.seed + 1)
    # train/val split for tuning the cheap methods
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(tr_s))
    n_val = len(tr_s) // 5
    vi, ti = perm[:n_val], perm[n_val:]
    print(f"train={len(ti)} val={len(vi)} test={len(te_s)} "
          f"({int(te_y.sum())} limitation / {int((1-te_y).sum())} other)", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("encoding (bge) ...", flush=True)
    Xall = enc.encode(tr_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xte = enc.encode(te_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xtr, Xval = Xall[ti], Xall[vi]
    ytr, yval = tr_y[ti], tr_y[vi]

    # ---- baseline + #2 threshold ----
    bge = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    proba_te = bge.predict_proba(Xte)[:, 1]
    emit("baseline bge+logreg @0.5", *prf(te_y, (proba_te >= 0.5).astype(int)))
    thr = best_threshold(bge.predict_proba(Xval)[:, 1], yval)
    emit("#2 bge+logreg, tuned threshold", *prf(te_y, (proba_te >= thr).astype(int)), note=f"thr={thr}")

    # ---- #3 stacking [bge, tfidf, cue] ----
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    Xtr_tf = tfidf.fit_transform([tr_s[k] for k in ti])
    tf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_tf, ytr)
    def stack_feats(sents, Xbge):
        p_bge = bge.predict_proba(Xbge)[:, 1]
        p_tf = tf.predict_proba(tfidf.transform(sents))[:, 1]
        cue = np.array([1.0 if cue_label(s) == "limitation" else 0.0 for s in sents])
        return np.column_stack([p_bge, p_tf, cue])
    Fval = stack_feats([tr_s[k] for k in vi], Xval)
    meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Fval, yval)
    Fte = stack_feats(te_s, Xte)
    proba_stack = meta.predict_proba(Fte)[:, 1]
    emit("#3 stacking [bge+tfidf+cue] @0.5", *prf(te_y, (proba_stack >= 0.5).astype(int)))
    sthr = best_threshold(meta.predict_proba(Fval)[:, 1], yval)
    emit("#3 stacking, tuned threshold", *prf(te_y, (proba_stack >= sthr).astype(int)), note=f"thr={sthr}")

    # ---- #1 DistilBERT fine-tuned on MORE data (slowest, last) ----
    print("fine-tuning DistilBERT on more data ...", flush=True)
    pred = m_bert_ft(tr_s, tr_y, te_s, args.ft_epochs, args.seed)
    emit(f"#1 DistilBERT fine-tuned ({len(tr_s)} train)", *prf(te_y, pred), note=f"{args.ft_epochs} epoch(s)")

    print(f"\nsaved -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
