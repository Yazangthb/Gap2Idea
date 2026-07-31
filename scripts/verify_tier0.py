"""Verify the Tier-0 benchmark: arithmetic, concrete TP/FP/FN cases, and the
effect of a context window (flag sentence i if itself OR a neighbor matches).

Run: python scripts/verify_tier0.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from gap2idea.pipeline.gap_prefilter import GapPrefilter
from eval_tier0 import effective_label, GAP_TYPES

BENCH = Path("data/bench")
labels = pd.read_csv(BENCH / "label_sheet.tsv", sep="\t", dtype={"paper_id": str, "sent_idx": int})
data = json.loads((Path("data/tier0_dictionary.json")).read_text(encoding="utf-8"))
max_n = int(data["max_n"])
pf = GapPrefilter(phrases=[p["phrase"] for p in data["dictionary"]], max_n=max_n)

labels["is_gap"] = labels.apply(lambda r: effective_label(r) in GAP_TYPES, axis=1)
labels["flagged"] = labels["sentence"].astype(str).apply(pf.is_candidate)

tp = labels[labels.is_gap & labels.flagged]
fp = labels[~labels.is_gap & labels.flagged]
fn = labels[labels.is_gap & ~labels.flagged]
tn = labels[~labels.is_gap & ~labels.flagged]

print("=== Arithmetic check (window=0, pool) ===")
print(f"TP={len(tp)} FP={len(fp)} FN={len(fn)} TN={len(tn)}  total={len(labels)}")
print(f"recall   = TP/(TP+FN) = {len(tp)}/{len(tp)+len(fn)} = {len(tp)/(len(tp)+len(fn)):.4f}")
print(f"survivor = (TP+FP)/N  = {len(tp)+len(fp)}/{len(labels)} = {(len(tp)+len(fp))/len(labels):.4f}")
print(f"specificity = TN/(TN+FP) = {len(tn)}/{len(tn)+len(fp)} = {len(tn)/(len(tn)+len(fp)):.4f}")

print("\n=== Sample TP (gap, flagged) — what phrase matched ===")
for _, r in tp.head(5).iterrows():
    print(f"  [{effective_label(r)}] match={sorted(pf.matched_phrases(r['sentence']))[:4]}")
    print(f"     {r['sentence'][:120]}")

print("\n=== FN (gap, NOT flagged) — genuinely cue-less? ===")
for _, r in fn.iterrows():
    print(f"  [{effective_label(r)}] match={sorted(pf.matched_phrases(r['sentence']))}")
    print(f"     {r['sentence'][:160]}")

print("\n=== Sample FP (non-gap, flagged) — cue-bearing but not a gap ===")
for _, r in fp.head(5).iterrows():
    print(f"  match={sorted(pf.matched_phrases(r['sentence']))[:4]}")
    print(f"     {r['sentence'][:120]}")

# ---- Context-window experiment: flag i if any of [i-w, i+w] matches ----
print("\n=== Context window: flag sentence i if itself OR a ±w neighbor matches ===")
print("(neighbor order = sent_idx within paper; approximate at gold/predicted boundary)")
for w in (0, 1, 2):
    fl = labels["flagged"].copy().to_numpy()
    new = fl.copy()
    for pid, sub in labels.groupby("paper_id"):
        idx = sub.sort_values("sent_idx").index.to_list()
        base = [bool(fl[labels.index.get_loc(i)]) for i in idx]
        for k in range(len(idx)):
            lo, hi = max(0, k - w), min(len(idx), k + w + 1)
            if any(base[lo:hi]):
                new[labels.index.get_loc(idx[k])] = True
    g = labels["is_gap"].to_numpy()
    tpn = int((g & new).sum()); fpn = int((~g & new).sum())
    fnn = int((g & ~new).sum()); tnn = int((~g & ~new).sum())
    rec = tpn / (tpn + fnn) if (tpn + fnn) else 0
    surv = (tpn + fpn) / len(labels)
    spec = tnn / (tnn + fpn) if (tnn + fpn) else 0
    print(f"  w={w}: recall={rec:.4f}  survivor={surv:.4f}  specificity={spec:.4f}  "
          f"(TP={tpn} FP={fpn} FN={fnn})")
