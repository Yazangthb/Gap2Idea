# -*- coding: utf-8 -*-
"""Span/paragraph-level scorer for limitation detection (Phase 1 of PUBLICATION_PLAN.md).

Span-based limitation golds (SAL, Kilicoglu) tag ONE sentence per limitation (the minimal
cause-clause) and exclude consequence/elaboration sentences. A sentence-level detector that
correctly flags an *adjacent* sentence of the same limitation is then scored as a false
positive. This scorer fixes that: a prediction counts as correct if it is within +/-W
sentences of a gold positive in the SAME paper (W=0 = strict sentence match; W=1 = span-level).

    from span_eval import span_metrics
    P, R, F1 = span_metrics(rows, pred_key="pred", W=1)

`rows` is an ordered list (reading order per paper) of dicts with:
  pmcid/paper_id, label (1=gold limitation), and a boolean/0-1 prediction under `pred_key`.
"""
from __future__ import annotations
from collections import defaultdict


def _positions(rows, id_key):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r[id_key]].append(i)
    pos = {}
    for pid, idxs in by.items():
        for p, gi in enumerate(idxs):
            pos[gi] = (pid, p)
    return pos


def span_metrics(rows, pred_key, W=1, label_key="label", id_key="pmcid"):
    """Return (precision, recall, f1) with +/-W-sentence span credit (W=0 = strict)."""
    pos = _positions(rows, id_key)
    gold_idx = [i for i, r in enumerate(rows) if r.get(label_key) == 1]
    pred_idx = [i for i, r in enumerate(rows) if r.get(pred_key)]
    goldby, predby = defaultdict(set), defaultdict(set)
    for j in gold_idx:
        goldby[pos[j][0]].add(pos[j][1])
    for i in pred_idx:
        predby[pos[i][0]].add(pos[i][1])
    tp = sum(1 for i in pred_idx if any(abs(pos[i][1] - g) <= W for g in goldby[pos[i][0]]))
    cov = sum(1 for j in gold_idx if any(abs(pos[j][1] - p) <= W for p in predby[pos[j][0]]))
    P = tp / len(pred_idx) if pred_idx else 0.0
    R = cov / len(gold_idx) if gold_idx else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return round(P, 3), round(R, 3), round(F, 3)


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Span-level scoring of a per-sentence predictions file.")
    ap.add_argument("preds", help="JSON list of ordered rows with pmcid/label/<pred_key>")
    ap.add_argument("--pred-key", default="pred")
    ap.add_argument("--id-key", default="pmcid")
    ap.add_argument("--windows", default="0,1,2")
    args = ap.parse_args()
    rows = json.load(open(args.preds, encoding="utf-8"))
    for W in (int(x) for x in args.windows.split(",")):
        P, R, F = span_metrics(rows, args.pred_key, W=W, id_key=args.id_key)
        print(f"W={W}: P={P} R={R} F1={F}")
