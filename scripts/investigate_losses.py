"""Investigate every gold v2 gap we lose: where does it die, and why?

For each gold gap:
  - Is it in Stage A v2 slice? (if no, Stage A is the bottleneck)
  - If in slice, did Stage B classify it as gap? (if no, Stage B is the bottleneck)
  - If predicted, did Stage C keep it? (if no, Stage C is over-rejecting)

Then groups losses by phrase pattern so we know what anchors/training to add.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import (
    slice_with_midpaper_anchors, token_containment,
)


def main():
    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences_v2.tsv", sep="\t", dtype={"paper_id": str})
    papers = {}
    for l in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l); papers[str(r["id"])] = r
    preds_v2 = pd.read_csv(ROOT / "data/scibert_prep/scibert_gold_gaps_v2slice.tsv", sep="\t").fillna("")

    # For each gold gap, find its fate
    results = []
    for _, g in gold.iterrows():
        pid = g["paper_id"]
        rec = papers.get(pid)
        if not rec:
            continue
        regs = slice_with_midpaper_anchors(rec["text"], blocks=rec.get("blocks"))
        slice_text = " ".join(s for r in regs for s in r.sentences)
        in_slice = token_containment(g["gap_sentence"], slice_text) >= 0.7
        # Did Stage B emit something matching?
        sub_pred = preds_v2[preds_v2["paper_id"] == pid]
        hit_b = False
        for p in sub_pred["gap_sentence"]:
            if token_containment(g["gap_sentence"], p) >= 0.7 or token_containment(p, g["gap_sentence"]) >= 0.7:
                hit_b = True; break
        # Where did it die?
        if not in_slice:
            stage = "DIES_AT_STAGE_A"
        elif not hit_b:
            stage = "DIES_AT_STAGE_B"
        else:
            stage = "SURVIVES_A+B"
        results.append({
            "gap_id": g["gap_id"], "type": g["gap_type"], "stage": stage,
            "sentence": g["gap_sentence"],
        })
    df = pd.DataFrame(results)

    # Summary
    print("=" * 80)
    print("WHERE EACH GOLD v2 GAP DIES (Stage A v2 + SciBERT-FT)")
    print("=" * 80)
    print(df["stage"].value_counts().to_string())
    print()

    # Show losses with content patterns
    for stage_name in ["DIES_AT_STAGE_A", "DIES_AT_STAGE_B"]:
        sub = df[df["stage"] == stage_name]
        if sub.empty:
            continue
        print(f"\n--- {stage_name} ({len(sub)} gaps) ---")
        # Categorize by first 4 words
        firsts = Counter(" ".join(s["sentence"].split()[:4]).lower() for _, s in sub.iterrows())
        print("Phrase-start histogram (top 10):")
        for p, n in firsts.most_common(10):
            print(f"  {n:>2}  \"{p}...\"")
        # Show sample sentences
        print(f"\nSample losses ({stage_name}):")
        for _, r in sub.head(8).iterrows():
            print(f"  [{r['type'][:5]}] {r['sentence'][:120]}")

    # What patterns Stage A could add
    print("\n" + "=" * 80)
    print("CANDIDATE NEW STAGE A ANCHOR PATTERNS (from missed gaps)")
    print("=" * 80)
    patterns_to_consider = [
        r"\bwe restrict\b", r"\blimited to\b", r"\bonly considers?\b",
        r"\bin (?:our|this) (?:setting|regime|setup)\b", r"\bdoes not\b",
        r"\bdo not\b", r"\bnot directly apply\b", r"\bonly requires?\b",
        r"\bin contrast\b", r"\bunfortunately\b", r"\bdo not experiment\b",
        r"\bsetting is limited\b", r"\binitial step on\b", r"\bsuffers? from\b",
        r"\beven if\b", r"\bif we further restrict\b", r"\bcontinues to face\b",
    ]
    a_loss = df[df["stage"] == "DIES_AT_STAGE_A"]
    for pat in patterns_to_consider:
        rx = re.compile(pat, re.IGNORECASE)
        n = sum(1 for _, r in a_loss.iterrows() if rx.search(r["sentence"]))
        if n > 0:
            print(f"  {n:>2}  pattern: {pat}")


if __name__ == "__main__":
    main()
