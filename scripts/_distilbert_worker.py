"""Isolated DistilBERT worker — trains in its own process (NO sentence-transformers
loaded), so it can't deadlock torch CPU threads against bge. Saves the fixed split
+ DistilBERT val/test probabilities for finish_stageb.py to fuse.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_limgen import build_xy, fetch  # noqa: E402  (does NOT import sentence-transformers)

OUT = ROOT / "data" / "limgen" / "_split"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-pos", type=int, default=1500)
    ap.add_argument("--train-papers", type=int, default=2000)
    ap.add_argument("--test-papers", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tr_s, tr_y = build_xy(fetch("train.jsonl"), args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(fetch("test.jsonl"), args.test_papers, None, args.seed + 1)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(tr_s))
    nv = len(tr_s) // 5; vi, ti = perm[:nv], perm[nv:]
    sub_s = [tr_s[k] for k in ti]; ytr = tr_y[ti]
    val_s = [tr_s[k] for k in vi]; yval = tr_y[vi]

    (OUT / "sub_s.json").write_text(json.dumps(sub_s), encoding="utf-8")
    (OUT / "val_s.json").write_text(json.dumps(val_s), encoding="utf-8")
    (OUT / "te_s.json").write_text(json.dumps(te_s), encoding="utf-8")
    np.save(OUT / "ytr.npy", ytr); np.save(OUT / "yval.npy", yval); np.save(OUT / "te_y.npy", te_y)
    print(f"split saved: train={len(sub_s)} val={len(val_s)} test={len(te_s)}", flush=True)

    import torch
    torch.set_num_threads(1)  # multi-threaded CPU backward deadlocks randomly on this box
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    counts = np.bincount(ytr, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5); bs = 16
    for ep in range(args.epochs):
        model.train(); order = rng.permutation(len(sub_s)); tot = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([sub_s[k] for k in idx], return_tensors="pt", padding=True, truncation=True, max_length=96)
            loss = lossf(model(**enc).logits, torch.tensor(ytr[idx], dtype=torch.long))
            loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss.detach())
        print(f"  distilbert epoch {ep+1}/{args.epochs} loss {tot/(len(order)//bs+1):.3f}", flush=True)

    def proba(sents):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(sents), 32):
                enc = tok(sents[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=96)
                out.extend(torch.softmax(model(**enc).logits, -1)[:, 1].tolist())
        return np.array(out)

    np.save(OUT / "pd_val.npy", proba(val_s))
    np.save(OUT / "pd_te.npy", proba(te_s))
    print("DistilBERT probabilities saved.", flush=True)


if __name__ == "__main__":
    main()
