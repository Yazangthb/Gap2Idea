"""Fine-tune Stage B on LimGen and chain with the LLM filter (Stage C).

Trains a transformer (SciBERT / DistilBERT / RoBERTa) on LimGen-train, evaluates
on test, then runs the LLM filter as Stage C on the model's positives and
evaluates the chain. Reports a single side-by-side table.

GPU-only (V100 32GB), uses fp16 and CUDA. Save predictions so the LLM step can be
re-run / swapped without retraining.

    python -u scripts/bench/finetune_and_chain.py --bert allenai/scibert_scivocab_uncased --llm Qwen/Qwen2.5-14B-Instruct
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_limgen import build_xy, fetch, prf  # noqa: E402


def train_bert(model_name, tr_s, tr_y, epochs, lr, bs, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(dev)
    counts = np.bincount(tr_y, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    print(f"  training {model_name} on {len(tr_s)} sents x {epochs} epochs (bs={bs}, lr={lr})", flush=True)
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(tr_s)); tot = 0.0; nb = 0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([tr_s[k] for k in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(dev)
            loss = lossf(model(**enc).logits, torch.tensor(tr_y[idx], dtype=torch.long).to(dev))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1
        print(f"    epoch {ep+1}/{epochs} loss {tot/max(1,nb):.4f}", flush=True)
    return model, tok, dev


def predict_bert(model, tok, te_s, dev, bs=64):
    import torch
    model.eval(); proba = []
    with torch.no_grad():
        for i in range(0, len(te_s), bs):
            enc = tok(te_s[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(dev)
            p = torch.softmax(model(**enc).logits, -1)[:, 1].cpu().numpy()
            proba.extend(p.tolist())
    return np.array(proba)


def best_thr(proba, y):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        _, _, f = prf(y, (proba >= t).astype(int))
        if f > bf:
            bf, bt = f, float(t)
    return bt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bert", default="allenai/scibert_scivocab_uncased")
    ap.add_argument("--llm", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--cap-pos", type=int, default=4000, help="cap positives in training")
    ap.add_argument("--train-papers", type=int, default=4000)
    ap.add_argument("--test-papers", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--llm-mode", default="validate", choices=["validate", "validate_cot", "validate_v5", "validate_v6", "validate_v7", "junk"])
    args = ap.parse_args()

    tr_s, tr_y = build_xy(fetch("train.jsonl"), args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(fetch("test.jsonl"), args.test_papers, None, args.seed + 1)
    # train/val split (val used for threshold tuning on the BERT only)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(tr_s))
    nv = len(tr_s) // 5; vi, ti = perm[:nv], perm[nv:]
    sub_s = [tr_s[k] for k in ti]; ytr = tr_y[ti]
    val_s = [tr_s[k] for k in vi]; yval = tr_y[vi]
    print(f"train={len(sub_s)} val={len(val_s)} test={len(te_s)}  "
          f"({int(te_y.sum())} lim / {int((1-te_y).sum())} other)", flush=True)

    # ---- fine-tune ----
    model, tok, dev = train_bert(args.bert, sub_s, ytr, args.epochs, args.lr, args.bs, args.seed)
    pte = predict_bert(model, tok, te_s, dev)
    pval = predict_bert(model, tok, val_s, dev)
    thr = best_thr(pval, yval)
    bert_pred = (pte >= thr).astype(int)
    Pb, Rb, Fb = prf(te_y, bert_pred)
    print(f"\n=== {args.bert} (fine-tuned) ===", flush=True)
    print(f"  thr={thr:.3f}  P={Pb} R={Rb} F1={Fb}", flush=True)
    # free GPU before loading LLM
    del model; import torch, gc; gc.collect(); torch.cuda.empty_cache()

    if args.skip_llm:
        return

    # ---- chain LLM filter over the BERT positives ----
    from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
    filt = LLMGapFilter(backend="local", model=args.llm, mode=args.llm_mode)
    keep = bert_pred.copy()
    judged = 0
    for i, s in enumerate(te_s):
        if bert_pred[i] == 1:
            judged += 1
            if not filt.judge(s):
                keep[i] = 0
    Pc, Rc, Fc = prf(te_y, keep)
    print(f"\n=== {args.bert} + {args.llm} (Stage C chain) ===", flush=True)
    print(f"  LLM judged {judged} BERT positives", flush=True)
    print(f"  P={Pc} R={Rc} F1={Fc}", flush=True)

    print(f"\nΔF1 (chain vs BERT alone): {Fc - Fb:+.3f}", flush=True)
    print(f"ΔF1 (chain vs Stage B bge+cue 0.598): {Fc - 0.598:+.3f}", flush=True)

    out = ROOT / "docs/experiments/finetune_chain.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Stage B fine-tune + Stage C chain (LimGen, leakage-clean)\n\n"
        f"BERT: {args.bert} (FT, {args.epochs}ep, lr={args.lr}, bs={args.bs}).  "
        f"LLM: {args.llm}.  test={len(te_s)} sents.\n\n"
        "| stage | precision | recall | F1 |\n|---|---|---|---|\n"
        f"| Stage B FT alone | {Pb} | {Rb} | {Fb} |\n"
        f"| **Stage B FT + Stage C LLM** | {Pc} | {Rc} | **{Fc}** |\n",
        encoding="utf-8")
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
