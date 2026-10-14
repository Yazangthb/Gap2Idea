# -*- coding: utf-8 -*-
"""Regenerate the FROZEN-encoder detector predictions in sal_test_allmodels.json.

The SAL experiments read precomputed per-sentence predictions from this JSON. Fine-tuned
columns (PubMedBERT-FT / SciBERT-FT / DistilBERT-FT) need their GPU checkpoints and are
PRESERVED as-is. The frozen-encoder + logreg columns (the production-style detectors) are
retrainable on CPU and are what this script regenerates — so when you change/retrain a head,
rerun this and the SAL funnel / baseline-grid / cost experiments pick up the new predictions.

    python scripts/bench/dump_sal_preds.py                       # regen bge-small, write new file
    python scripts/bench/dump_sal_preds.py --encoders all --inplace   # regen all frozen cols, overwrite

Removes the "SAL detector predictions are pinned" caveat in docs/experiments/REPRODUCIBILITY.md
for the frozen heads.
"""
from __future__ import annotations
import argparse, ast, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

# frozen column name -> sentence-transformers model id
ENCODERS = {
    "bge-small(frozen)": "BAAI/bge-small-en-v1.5",
    "bge-base(frozen)":  "BAAI/bge-base-en-v1.5",
    "e5-base(frozen)":   "intfloat/e5-base-v2",
    "mpnet(frozen)":     "sentence-transformers/all-mpnet-base-v2",
    "MiniLM-L6(frozen)": "sentence-transformers/all-MiniLM-L6-v2",
}

def parse_lim(x):
    try: return 1 if any(bool(b) for b in ast.literal_eval(x)) else 0
    except Exception: return 1 if "True" in str(x) else 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(ROOT/"stage_c_package"/"sal_criteria"/"sal_test_allmodels.json"))
    ap.add_argument("--train", default=str(ROOT/"data"/"external"/"sal_rct"/"train.csv"))
    ap.add_argument("--encoders", default="bge-small(frozen)",
                    help="'all' or comma list of frozen column names to regenerate")
    ap.add_argument("--threshold", type=float, default=0.5, help="P(limitation) cutoff (lower = recall-first)")
    ap.add_argument("--inplace", action="store_true", help="overwrite --json instead of writing *_regen.json")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    js = json.load(open(args.json, encoding="utf-8"))
    te_s = [r["sent"] for r in js]
    tr = pd.read_csv(args.train)
    tr_s = tr.sentences.astype(str).tolist(); tr_y = np.array([parse_lim(x) for x in tr.limitation_annotation])
    cols = list(ENCODERS) if args.encoders == "all" else [c.strip() for c in args.encoders.split(",")]
    present = [c for c in js[0]["preds"]]
    print(f"test sents={len(te_s)} | train={len(tr_y)} ({tr_y.mean():.1%} pos) | thr={args.threshold}")
    print(f"regenerating: {cols}\npreserving (need GPU checkpoints): {[c for c in present if c not in cols]}\n")

    for col in cols:
        if col not in ENCODERS:
            print(f"  skip {col}: not a known frozen encoder"); continue
        enc = SentenceTransformer(ENCODERS[col])
        Xtr = enc.encode(tr_s, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        Xte = enc.encode(te_s, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, tr_y)
        pos_i = list(clf.classes_).index(1)
        proba = clf.predict_proba(Xte)[:, pos_i]
        pred = (proba >= args.threshold).astype(int)
        for r, p in zip(js, pred): r["preds"][col] = int(p)
        gold = np.array([r["label"] for r in js])
        tp = int(((pred==1)&(gold==1)).sum()); fp=int(((pred==1)&(gold==0)).sum()); fn=int(((pred==0)&(gold==1)).sum())
        P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0
        print(f"  {col:20} P={P:.3f} R={R:.3f} F1={2*P*R/(P+R) if P+R else 0:.3f}  (pred+={int(pred.sum())})")

    out = Path(args.json) if args.inplace else Path(args.json).with_name(Path(args.json).stem + "_regen.json")
    json.dump(js, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nwrote {out}  ({'in place' if args.inplace else 'new file; use --inplace to overwrite'})")

if __name__ == "__main__":
    main()
