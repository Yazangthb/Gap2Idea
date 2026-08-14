"""Fine-tune our Stage B model on RCT/SAL (binary limitation detection).

Frozen bge-small + logreg plateaus ~0.25 F1 on RCT (heavy imbalance, biomedical).
This fine-tunes the transformer end-to-end (adapts the representation), then adds
the batched Stage C filter, and reports vs the frozen baseline and PubMedBERT 0.821.

CPU is slow (~1s/example) -> we undersample negatives. On a GPU set a larger
--neg-ratio and more --epochs; the same script matches their setup with
--model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext.

    python scripts/training/finetune_rct.py                       # bge-small, CPU
    python scripts/training/finetune_rct.py --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --epochs 3 --neg-ratio 6  # GPU
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from bench_rct import load, stage_c  # noqa: E402
from bench_limgen import prf  # noqa: E402
from gap2idea.pipeline.llm import active_provider  # noqa: E402


def train(model, tok, s, y, epochs, seed, device, bs=16):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    counts = np.bincount(y, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(s)); tot = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([s[k] for k in idx], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
            yb = torch.tensor(y[idx], dtype=torch.long, device=device)
            loss = lossf(model(**enc).logits, yb)
            loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss.detach())
        print(f"    epoch {ep+1}/{epochs} loss {tot/(len(order)//bs+1):.3f}", flush=True)


def proba(model, tok, s, device, bs=64):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(s), bs):
            enc = tok(s[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=96).to(device)
            out.extend(torch.softmax(model(**enc).logits, -1)[:, 1].cpu().tolist())
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--neg-ratio", type=int, default=3)
    ap.add_argument("--test-cap", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rec-target", type=float, default=0.90, help="dev recall floor for the recall-tuned threshold")
    ap.add_argument("--eval-prevalence", type=float, default=None,
                    help="also evaluate on a negative-subsampled pool at this positive rate (e.g. 0.207 to match their pool)")
    ap.add_argument("--pool-csv", default=None, help="section-filtered pool CSV (from build_rct_pool.py) to also evaluate on")
    ap.add_argument("--dump-preds", default=None, help="write pool sentences+gold+detector-pred JSON here (for Stage C tuning)")
    args = ap.parse_args()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)

    tr_s, tr_y = load("train"); dv_s, dv_y = load("dev"); te_s, te_y = load("test", args.test_cap)
    pos = np.where(tr_y == 1)[0]; neg = np.where(tr_y == 0)[0]
    rng = np.random.default_rng(args.seed)
    keep = np.concatenate([pos, rng.choice(neg, size=min(len(neg), len(pos) * args.neg_ratio), replace=False)])
    rng.shuffle(keep)
    trs = [tr_s[i] for i in keep]; trly = tr_y[keep]
    print(f"provider={active_provider()} device={device} model={args.model}", flush=True)
    print(f"train {len(trs)} ({int(trly.sum())} lim, undersampled 1:{args.neg_ratio}) / "
          f"dev {len(dv_s)} / test {len(te_s)} ({int(te_y.sum())} lim)", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    t0 = time.time()
    train(model, tok, trs, trly, args.epochs, args.seed, device)
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)

    grid = np.linspace(0.05, 0.95, 91)
    pd = proba(model, tok, dv_s, device)
    # (a) max-F1 threshold (the single-model operating point -- what a lone classifier picks)
    maxf_t, best_f = 0.5, -1
    for t in grid:
        f = prf(dv_y, (pd >= t).astype(int))[2]
        if f > best_f: best_f, maxf_t = f, t
    # (b) recall-tuned threshold: lowest threshold (=> highest precision) that still hits >=0.90 dev recall,
    #     leaving precision for Stage C to recover. This is the two-stage operating point.
    rec_t = maxf_t
    for t in grid:  # ascending; keep the largest t whose recall is still >= target
        if prf(dv_y, (pd >= t).astype(int))[1] >= args.rec_target:
            rec_t = t
    pt = proba(model, tok, te_s, device)
    name = args.model.split('/')[-1]

    def report(tag, thr, idx, with_c):
        s = [te_s[i] for i in idx]; y = te_y[idx]; p = pt[idx]
        sb = (p >= thr).astype(int)
        P, R, F = prf(y, sb)
        beat = "  >= PubMedBERT 0.821" if F >= 0.821 else ""
        print(f"\n{tag} {name}  (thr={thr:.2f})  P={P} R={R} F1={F}{beat}", flush=True)
        if with_c:
            fc, nc, nj, nd = stage_c(s, sb)
            Pc, Rc, Fc = prf(y, fc)
            beatc = "  >= PubMedBERT 0.821" if Fc >= 0.821 else ""
            print(f"  + Stage C (judged {nj} in {nc} calls, dropped {nd}):  P={Pc} R={Rc} F1={Fc}{beatc}", flush=True)

    full = np.arange(len(te_y))
    print(f"\n### FULL STREAM  (prevalence {te_y.mean():.1%}, {int(te_y.sum())}/{len(te_y)})")
    report("[max-F1]     ", maxf_t, full, True)
    report("[recall-tuned]", rec_t, full, True)

    if args.eval_prevalence:
        # Replicate their evaluation condition: a candidate pool at ~their prevalence.
        # Keep every positive, subsample negatives so positives == eval_prevalence. Recall is
        # unchanged by this; precision is corrected to their base rate -> the fair comparison.
        rng = np.random.default_rng(args.seed)
        pos = np.where(te_y == 1)[0]; neg = np.where(te_y == 0)[0]
        n_neg = min(len(neg), int(round(len(pos) * (1 - args.eval_prevalence) / args.eval_prevalence)))
        idx = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)]); rng.shuffle(idx)
        print(f"\n### PREVALENCE-MATCHED to their pool  ({te_y[idx].mean():.1%}, {len(pos)}/{len(idx)})")
        report("[max-F1]     ", maxf_t, idx, True)
        report("[recall-tuned]", rec_t, idx, True)

    if args.pool_csv:
        import csv as _csv
        norm = lambda x: " ".join(x.split())
        want = {}
        with open(args.pool_csv, encoding="utf-8") as f:
            for r in _csv.DictReader(f):
                want[norm(r["sentence"])] = int(r["label"])
        idx = np.array([i for i, s in enumerate(te_s) if norm(s) in want])
        yp = te_y[idx]
        print(f"\n### SECTION-FILTERED POOL  ({yp.mean():.1%}, {int(yp.sum())}/{len(idx)})  [their filter, our test articles]")
        report("[max-F1]     ", maxf_t, idx, True)
        report("[recall-tuned]", rec_t, idx, True)
        if args.dump_preds:
            sb = (pt[idx] >= rec_t).astype(int)
            rows = [{"sentence": te_s[i], "gold": int(te_y[i]), "pred": int(pt[i] >= rec_t)} for i in idx]
            import json as _json
            Path(args.dump_preds).write_text(_json.dumps(
                {"threshold": float(rec_t), "rows": rows}), encoding="utf-8")
            print(f"  dumped {len(rows)} pool preds ({int(sb.sum())} predicted-pos) -> {args.dump_preds}", flush=True)

    print(f"\nreference: frozen bge+logreg 0.25 (+C 0.55) | PubMedBERT fine-tuned 0.821 (their pool ~20.7% pos)")


if __name__ == "__main__":
    main()
