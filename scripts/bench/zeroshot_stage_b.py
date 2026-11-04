# -*- coding: utf-8 -*-
"""Zero-shot NLI cross-encoder baseline for Stage B (no training).

Frames gap detection as entailment — "This sentence points out a limitation /
open problem / future-work direction of the research" — scored by an off-the-shelf
NLI cross-encoder. Compares, on the SAME paper-level split and the SAME per-sentence
test set as the trained head (train_gold_head.py), to isolate whether a cross-encoder
helps vs our frozen bi-encoder + logreg (test F1 0.453).

Protocol match: gap = any type (limitation/future_work/research_gap); threshold tuned
on VAL for binary F1; reported on the full TEST set at realistic prevalence.

    python scripts/bench/zeroshot_stage_b.py                 # default model, GPU if present
    python scripts/bench/zeroshot_stage_b.py --model <hf_id> --limit-test 3000

Writes docs/experiments/stage_b_zeroshot.md.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass
from gold_label_full import load_corpus

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
SPLITS = ROOT / "data" / "gold" / "splits.json"
OUTDOC = ROOT / "docs" / "experiments" / "stage_b_zeroshot.md"

GAP_HYP = ("This sentence points out a limitation, weakness, shortcoming, open research "
           "problem, or future-work direction of the research described.")
TYPE_LABELS = {
    "limitation": "a limitation, weakness, or shortcoming of the research",
    "future_work": "a future-work direction or something left for future research",
    "research_gap": "an open research problem or unresolved gap in knowledge",
}
HYP_TEMPLATE = "This sentence describes {}."


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def build(split_part, gold, domain, corp):
    S, Y, D = [], [], []
    for pid in split_part:
        p = corp[pid]
        for i, s in enumerate(p["sents"]):
            S.append(s); Y.append(gold.get((pid, i), "none")); D.append(domain[pid])
    return S, Y, D


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="MoritzLaurer/deberta-v3-base-zeroshot-v2.0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit-test", type=int, default=0, help="subsample test (prevalence-preserved)")
    ap.add_argument("--max-length", type=int, default=160)
    args = ap.parse_args()

    gold, domain = {}, {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            g = json.loads(ln)
            gold[(g["paper_id"], g["idx"])] = g["label"]
            domain[g["paper_id"]] = g["domain"]
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in domain}
    split = json.loads(SPLITS.read_text(encoding="utf-8"))

    vaS, vaY, vaD = build(split["val"], gold, domain, corp)
    teS, teY, teD = build(split["test"], gold, domain, corp)
    if args.limit_test and len(teS) > args.limit_test:
        import numpy as np
        rng = np.random.default_rng(42)
        idx = sorted(rng.permutation(len(teS))[:args.limit_test])
        teS = [teS[i] for i in idx]; teY = [teY[i] for i in idx]; teD = [teD[i] for i in idx]

    import torch
    from transformers import pipeline
    dev = 0 if torch.cuda.is_available() else -1
    print(f"model={args.model} device={'cuda' if dev == 0 else 'cpu'} "
          f"val={len(vaS)} test={len(teS)}", flush=True)
    zsc = pipeline("zero-shot-classification", model=args.model, device=dev,
                   batch_size=args.batch, truncation=True, max_length=args.max_length)

    def gap_scores(sents):
        out = []
        res = zsc(sents, candidate_labels=[GAP_HYP], hypothesis_template="{}", multi_label=True)
        if isinstance(res, dict):
            res = [res]
        for r in res:
            out.append(float(r["scores"][0]))
        return out

    print("scoring val ...", flush=True)
    va_scores = gap_scores(vaS)
    print("scoring test ...", flush=True)
    te_scores = gap_scores(teS)

    def binmetrics(scores, Y, D, tau):
        tp = fp = fn = 0
        by_dom = defaultdict(lambda: [0, 0, 0])
        by_type = defaultdict(lambda: [0, 0])
        for sc, y, d in zip(scores, Y, D):
            pg = sc >= tau; tg = y != "none"
            if pg and tg: tp += 1; by_dom[d][0] += 1
            elif pg and not tg: fp += 1; by_dom[d][1] += 1
            elif not pg and tg: fn += 1; by_dom[d][2] += 1
            if tg:
                by_type[y][1] += 1; by_type[y][0] += int(pg)
        P, R, F = prf(tp, fp, fn)
        return dict(P=P, R=R, F=F, tp=tp, fp=fp, fn=fn,
                    by_dom={k: prf(*v) for k, v in by_dom.items()},
                    by_type={k: (v[0] / v[1] if v[1] else 0, v[1]) for k, v in by_type.items()})

    import numpy as np
    best = (0.5, -1.0)
    for tau in np.arange(0.20, 0.96, 0.05):
        f = binmetrics(va_scores, vaY, vaD, float(tau))["F"]
        if f > best[1]:
            best = (round(float(tau), 2), f)
    tau = best[0]
    vm = binmetrics(va_scores, vaY, vaD, tau)
    tm = binmetrics(te_scores, teY, teD, tau)

    # type-acc on predicted positives only
    pos = [i for i, sc in enumerate(te_scores) if sc >= tau and teY[i] != "none"]
    type_acc = None
    if pos:
        labs = list(TYPE_LABELS)
        res = zsc([teS[i] for i in pos], candidate_labels=[TYPE_LABELS[k] for k in labs],
                  hypothesis_template=HYP_TEMPLATE, multi_label=False)
        if isinstance(res, dict):
            res = [res]
        lab_by_desc = {v: k for k, v in TYPE_LABELS.items()}
        correct = 0
        for i, r in zip(pos, res):
            pred = lab_by_desc[r["labels"][0]]
            correct += int(pred == teY[i])
        type_acc = correct / len(pos)

    L = ["# Stage B — zero-shot NLI cross-encoder baseline (no training)\n",
         f"Model: `{args.model}`. Same split/test set as the trained head "
         f"(test {len(teS)} sents, {sum(1 for y in teY if y!='none')} gaps). "
         "Gap = any type; τ tuned on VAL for binary F1; realistic prevalence.\n",
         "| detector | τ | TEST P | TEST R | TEST F1 |",
         "|---|---|---|---|---|",
         f"| **zero-shot NLI** | {tau} | {tm['P']:.3f} | {tm['R']:.3f} | **{tm['F']:.3f}** |",
         "| trained frozen bge-small + logreg (ref) | 0.85 | 0.328 | 0.733 | 0.453 |",
         "",
         f"VAL F1 at τ={tau}: {vm['F']:.3f}",
         "\nPer-domain (test): " + ", ".join(
             f"{d} P={p:.3f} R={r:.3f} F1={f:.3f}" for d, (p, r, f) in sorted(tm["by_dom"].items())),
         "\nPer-type recall (test): " + ", ".join(
             f"{k} {v[0]*100:.1f}% (n={v[1]})" for k, v in sorted(tm["by_type"].items())),
         f"\nType accuracy among detected gaps: {type_acc*100:.1f}%" if type_acc is not None else "",
         "\n### Notes\n"
         "- Zero-shot = no gold training; gold-independent. A reference point for whether a "
         "cross-encoder (joint attention over hypothesis+sentence) beats the frozen bi-encoder.\n"
         "- τ tuned on val (a hyperparameter, not training) to match the trained head's protocol.\n"
         "- Single sentence, no document context — a known handicap for this task."]
    doc = "\n".join(x for x in L if x is not None) + "\n"
    OUTDOC.parent.mkdir(parents=True, exist_ok=True)
    OUTDOC.write_text(doc, encoding="utf-8")
    print(doc)
    print("wrote", OUTDOC)


if __name__ == "__main__":
    main()
