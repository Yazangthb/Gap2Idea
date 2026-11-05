# -*- coding: utf-8 -*-
"""Stage-B on the V100: fine-tune a transformer classifier (and/or zero-shot NLI)
on the CURRENT gold, evaluated with the SAME binary-gap metric as the frozen head
(train_gold_head.py) so numbers compare directly to its test F1 0.453.

Self-contained: reads data_v2/{train,val,test}.jsonl ({sentence,label,domain,...}).
Gap = any type; pgap = 1 - P(none); threshold tuned on VAL for binary F1; report TEST.

  python stageb_v100.py --mode ft --model BAAI/bge-small-en-v1.5 --tag bge-small
  python stageb_v100.py --mode ft --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext --tag biomedbert
  python stageb_v100.py --mode zeroshot --model MoritzLaurer/deberta-v3-base-zeroshot-v2.0 --tag zshot
"""
from __future__ import annotations
import os, sys, json, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data_v2"
RES = HERE / "results_v2"; RES.mkdir(exist_ok=True)
CLASSES = ["none", "limitation", "future_work", "research_gap"]
GAP = CLASSES[1:]
L2I = {c: i for i, c in enumerate(CLASSES)}


def load(part):
    S, Y, D = [], [], []
    for ln in (DATA / f"{part}.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); S.append(r["sentence"]); Y.append(r["label"]); D.append(r["domain"])
    return S, Y, D


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def binmetrics(pgap, ptype_idx, Y, D, tau):
    """pgap: array P(gap). ptype_idx: array argmax type-class idx into GAP (0..2) or None."""
    tp = fp = fn = 0
    by_dom = defaultdict(lambda: [0, 0, 0]); by_type = defaultdict(lambda: [0, 0])
    tcorr = tgp = 0
    for i, (y, d) in enumerate(zip(Y, D)):
        pg = pgap[i] >= tau; tg = y != "none"
        if pg and tg: tp += 1; by_dom[d][0] += 1
        elif pg and not tg: fp += 1; by_dom[d][1] += 1
        elif not pg and tg: fn += 1; by_dom[d][2] += 1
        if tg:
            by_type[y][1] += 1
            if pg:
                by_type[y][0] += 1
                if ptype_idx is not None:
                    tgp += 1; tcorr += int(GAP[ptype_idx[i]] == y)
    P, R, F = prf(tp, fp, fn)
    return dict(P=P, R=R, F=F, tp=tp, fp=fp, fn=fn,
                by_dom={k: prf(*v) for k, v in by_dom.items()},
                by_type={k: (v[0]/v[1] if v[1] else 0, v[1]) for k, v in by_type.items()},
                type_acc=(tcorr/tgp if tgp else 0.0))


def tune_tau(pgap, ptype, Y, D):
    best = (0.5, -1.0)
    for tau in np.arange(0.20, 0.96, 0.05):
        f = binmetrics(pgap, ptype, Y, D, float(tau))["F"]
        if f > best[1]:
            best = (round(float(tau), 2), f)
    return best[0]


def run_ft(args):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    trS, trY, trD = load("train"); vaS, vaY, vaD = load("val"); teS, teY, teD = load("test")

    # neg subsample 4x (match frozen protocol)
    rng = np.random.default_rng(args.seed)
    pos = [i for i, y in enumerate(trY) if y != "none"]
    neg = [i for i, y in enumerate(trY) if y == "none"]
    cap = int(len(pos) * args.neg_ratio)
    if len(neg) > cap:
        neg = list(rng.permutation(neg)[:cap])
    keep = sorted(pos + neg)
    trS = [trS[i] for i in keep]; trY = [trY[i] for i in keep]
    print(f"[{args.tag}] train rows {len(trS)} (pos {len(pos)}) | val {len(vaS)} | test {len(teS)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=4,
        id2label={i: c for c, i in L2I.items()}, label2id=L2I).to(dev)

    class DS(Dataset):
        def __init__(self, S, Y=None):
            self.S = S; self.Y = Y
        def __len__(self): return len(self.S)
        def __getitem__(self, i):
            return self.S[i], (L2I[self.Y[i]] if self.Y is not None else -1)

    def collate(batch):
        texts = [b[0] for b in batch]; labels = torch.tensor([b[1] for b in batch])
        enc = tok(texts, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
        return enc, labels

    # class-weighted loss (gap classes up-weighted)
    import collections as _c
    cnt = _c.Counter(trY)
    w = torch.tensor([1.0 / max(1, cnt.get(c, 1)) for c in CLASSES], dtype=torch.float)
    w = (w / w.sum() * len(CLASSES)).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    dl = DataLoader(DS(trS, trY), batch_size=args.batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = len(dl) * args.epochs
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    for ep in range(args.epochs):
        t0 = time.time(); tot = 0.0
        for enc, labels in dl:
            enc = {k: v.to(dev) for k, v in enc.items()}; labels = labels.to(dev)
            opt.zero_grad()
            out = model(**enc)
            loss = lossf(out.logits, labels)
            loss.backward(); opt.step(); sch.step()
            tot += loss.item()
        print(f"[{args.tag}] epoch {ep+1}/{args.epochs} loss {tot/len(dl):.4f} ({time.time()-t0:.0f}s)", flush=True)

    @torch.no_grad()
    def predict(S):
        model.eval(); out_p = []
        for i in range(0, len(S), args.batch):
            enc = tok(S[i:i+args.batch], padding=True, truncation=True,
                      max_length=args.max_length, return_tensors="pt").to(dev)
            logits = model(**enc).logits
            out_p.append(torch.softmax(logits, -1).cpu().numpy())
        return np.concatenate(out_p, 0)

    Pva = predict(vaS); Pte = predict(teS)
    none_j = L2I["none"]; gap_js = [L2I[c] for c in GAP]
    def gp(P): return 1.0 - P[:, none_j]
    def gt(P): return np.array([gap_js.index(gap_js[int(np.argmax(P[i, gap_js]))]) for i in range(len(P))])
    pva, pte = gp(Pva), gp(Pte)
    tva = np.array([int(np.argmax(Pva[i, gap_js])) for i in range(len(Pva))])
    tte = np.array([int(np.argmax(Pte[i, gap_js])) for i in range(len(Pte))])
    tau = tune_tau(pva, tva, vaY, vaD)
    vm = binmetrics(pva, tva, vaY, vaD, tau); tm = binmetrics(pte, tte, teY, teD, tau)
    res = {"tag": args.tag, "model": args.model, "mode": "ft", "tau": tau,
           "epochs": args.epochs, "lr": args.lr, "neg_ratio": args.neg_ratio,
           "val_F": vm["F"], "test": {k: tm[k] for k in ("P", "R", "F", "type_acc")},
           "test_by_dom": tm["by_dom"], "test_by_type": tm["by_type"]}
    (RES / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.tag}] tau={tau} VAL F1={vm['F']:.3f} | TEST P={tm['P']:.3f} R={tm['R']:.3f} "
          f"F1={tm['F']:.3f} type-acc={tm['type_acc']:.3f}", flush=True)


def run_zeroshot(args):
    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    vaS, vaY, vaD = load("val"); teS, teY, teD = load("test")
    GAP_HYP = ("This sentence points out a limitation, weakness, shortcoming, open research "
               "problem, or future-work direction of the research described.")
    zsc = pipeline("zero-shot-classification", model=args.model, device=dev,
                   batch_size=args.batch, truncation=True, max_length=args.max_length)

    def scores(S):
        out = []
        res = zsc(S, candidate_labels=[GAP_HYP], hypothesis_template="{}", multi_label=True)
        if isinstance(res, dict): res = [res]
        for r in res: out.append(float(r["scores"][0]))
        return np.array(out)
    print(f"[{args.tag}] scoring val {len(vaS)} ...", flush=True); pva = scores(vaS)
    print(f"[{args.tag}] scoring test {len(teS)} ...", flush=True); pte = scores(teS)
    tau = tune_tau(pva, None, vaY, vaD)
    vm = binmetrics(pva, None, vaY, vaD, tau); tm = binmetrics(pte, None, teY, teD, tau)
    res = {"tag": args.tag, "model": args.model, "mode": "zeroshot", "tau": tau,
           "val_F": vm["F"], "test": {k: tm[k] for k in ("P", "R", "F")},
           "test_by_dom": tm["by_dom"], "test_by_type": tm["by_type"]}
    (RES / f"{args.tag}.json").write_text(json.dumps(res, indent=2))
    print(f"[{args.tag}] tau={tau} VAL F1={vm['F']:.3f} | TEST P={tm['P']:.3f} R={tm['R']:.3f} "
          f"F1={tm['F']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ft", "zeroshot"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--neg-ratio", type=float, default=4.0)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    (run_ft if args.mode == "ft" else run_zeroshot)(args)


if __name__ == "__main__":
    main()
