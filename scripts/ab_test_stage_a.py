"""A/B test Stage A v1 (slice_terminal_regions) vs v2 (slice_with_midpaper_anchors)
on gold v2 (49 gaps / 10 papers). Reports localization recall + slice size for
each version. Does NOT modify or replace v1 — purely additive comparison.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    slice_terminal_regions, slice_with_midpaper_anchors, token_containment,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402


def measure(papers, gold, slice_fn, name):
    full_total = slice_total = 0
    loc_70 = loc_80 = loc_90 = 0
    n_regions = 0
    for pid, rec in papers.items():
        regs = slice_fn(rec["text"], blocks=rec.get("blocks"))
        slice_text = " ".join(s for r in regs for s in r.sentences)
        full = split_sentences(_cut_before_references(rec["text"]))
        full_total += len(full)
        slice_total += sum(len(r.sentences) for r in regs)
        n_regions += len(regs)
        for _, g in gold[gold["paper_id"] == pid].iterrows():
            c = token_containment(g["gap_sentence"], slice_text)
            if c >= 0.7: loc_70 += 1
            if c >= 0.8: loc_80 += 1
            if c >= 0.9: loc_90 += 1
    print(f"\n=== {name} ===")
    print(f"  total full sentences:  {full_total}")
    print(f"  total slice sentences: {slice_total}")
    print(f"  drop rate:             {100*(1-slice_total/full_total):.1f}%")
    print(f"  regions per paper:     {n_regions/len(papers):.1f}")
    print(f"  localization @ 0.90:   {loc_90}/{len(gold)} = {loc_90/len(gold):.3f}")
    print(f"  localization @ 0.80:   {loc_80}/{len(gold)} = {loc_80/len(gold):.3f}")
    print(f"  localization @ 0.70:   {loc_70}/{len(gold)} = {loc_70/len(gold):.3f}")
    return slice_total, loc_70, loc_80, loc_90


def main():
    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences_v2.tsv", sep="\t", dtype={"paper_id": str})
    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec
    print(f"Gold v2: {len(gold)} gaps over {gold['paper_id'].nunique()} papers")
    print(f"Papers:  {len(papers)}")

    v1_s, v1_70, v1_80, v1_90 = measure(papers, gold, slice_terminal_regions, "v1 — slice_terminal_regions (FROZEN)")
    v2_s, v2_70, v2_80, v2_90 = measure(papers, gold, slice_with_midpaper_anchors, "v2 — slice_with_midpaper_anchors")

    print(f"\n=== DELTA v2 vs v1 (on gold v2) ===")
    print(f"  slice size:     +{v2_s - v1_s} sentences ({100*(v2_s-v1_s)/v1_s:+.1f}%)")
    print(f"  loc @ 0.70:     {v1_70/len(gold):.3f} → {v2_70/len(gold):.3f}  (+{(v2_70-v1_70)/len(gold):.3f})")
    print(f"  loc @ 0.80:     {v1_80/len(gold):.3f} → {v2_80/len(gold):.3f}  (+{(v2_80-v1_80)/len(gold):.3f})")


if __name__ == "__main__":
    main()
