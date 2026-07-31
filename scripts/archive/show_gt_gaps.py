"""Show the ground-truth (silver) gap sentences for inspection / adjudication.

Prints a digest and writes data/bench/gt_gaps_review.tsv with an empty `verdict`
column (mark: keep / drop / retype:<type>) for human adjudication.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_tier0 import effective_label, GAP_TYPES  # noqa

BENCH = Path("data/bench")
df = pd.read_csv(BENCH / "label_sheet.tsv", sep="\t", dtype={"paper_id": str, "sent_idx": int})
df["eff"] = df.apply(effective_label, axis=1)
gaps = df[df["eff"].isin(GAP_TYPES)].copy()

print(f"=== {len(gaps)} GT gap sentences across {gaps['paper_id'].nunique()} papers ===")
print("\nBy type:")
print(gaps["eff"].value_counts().to_string())
print("\nBy silver confidence bucket:")
conf = pd.to_numeric(gaps["silver_confidence"], errors="coerce")
print(pd.cut(conf, [0, .5, .7, .9, 1.0]).value_counts().sort_index().to_string())

# --- the quantum sentence in question, with context ---
print("\n=== The sentence you flagged (quant-ph/0402095 #0) + context ===")
qp = df[df["paper_id"] == "quant-ph/0402095"].sort_values("sent_idx")
for _, r in qp.head(4).iterrows():
    tag = f"<<{r['eff']}>>" if r["eff"] in GAP_TYPES else "(none)"
    print(f"  #{r['sent_idx']} {tag} conf={r['silver_confidence']}")
    print(f"     {r['sentence']}")
    if r["eff"] in GAP_TYPES:
        print(f"     rationale: {r['silver_rationale']}")

# --- full list grouped by type ---
for t in ["limitation", "future_work", "open_problem"]:
    sub = gaps[gaps["eff"] == t]
    print(f"\n=== {t} ({len(sub)}) ===")
    for _, r in sub.iterrows():
        print(f"  [{r['paper_id']} #{r['sent_idx']}] conf={r['silver_confidence']}  {str(r['sentence'])[:120]}")

# --- write adjudication sheet ---
review = gaps[["paper_id", "sent_idx", "source", "eff", "silver_confidence",
               "sentence", "silver_rationale"]].rename(columns={"eff": "silver_label"})
review["verdict"] = ""   # keep / drop / retype:<type>
out = BENCH / "gt_gaps_review.tsv"
review.to_csv(out, sep="\t", index=False)
print(f"\nWrote {out} ({len(review)} rows) — fill `verdict` to adjudicate.")
