"""Worker: Stage B (bge+logreg + cue) on a LimGen test sample. Saves predictions
so the LLM (Stage C) can run in a SEPARATE process (avoids the torch-thread hang
when sentence-transformers + a generating LLM share one process)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from gap2idea.pipeline.gap_funnel import cue_label  # noqa: E402
from bench_limgen import build_xy, fetch  # noqa: E402

OUT = ROOT / "data" / "limgen" / "_sc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-pos", type=int, default=600)
    ap.add_argument("--train-papers", type=int, default=900)
    ap.add_argument("--test-papers", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tr_s, tr_y = build_xy(fetch("train.jsonl"), args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(fetch("test.jsonl"), args.test_papers, None, args.seed + 1)
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5")
    Xtr = enc.encode(tr_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xte = enc.encode(te_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, tr_y)

    bge_pos = clf.predict(Xte).astype(int)
    cue_pos = np.array([1 if cue_label(s) == "limitation" else 0 for s in te_s])
    (OUT / "te_s.json").write_text(json.dumps(te_s), encoding="utf-8")
    np.save(OUT / "te_y.npy", te_y)
    np.save(OUT / "bge_pos.npy", bge_pos)
    np.save(OUT / "cue_pos.npy", cue_pos)
    print(f"stage-B saved: test={len(te_s)} ({int(te_y.sum())} limitation) "
          f"stageB_pos={int((bge_pos | cue_pos).sum())} cue_pos={int(cue_pos.sum())}", flush=True)


if __name__ == "__main__":
    main()
