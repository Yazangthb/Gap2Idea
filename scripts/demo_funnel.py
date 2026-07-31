"""Demo: run Stage A + Stage B on a few papers, show the output and the reduction.

Prints, per paper: the funnel (full sentences -> Stage A slice -> Stage B gaps)
with drop %, the emitted gap sentences (type/source/section, flagged vs gold),
and a couple of slice sentences Stage B rejected. Then an aggregate reduction.

    python scripts/demo_funnel.py --head data/gap_head.joblib --detail 4
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    slice_terminal_regions, extract_gaps, token_containment,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402
from bench_gap_recall import load as load_gold  # noqa: E402


def short(s, n=88):
    return textwrap.shorten(" ".join(str(s).split()), n)


def gold_match(sent, gold_rows):
    for _, g in gold_rows.iterrows():
        if max(token_containment(g["gap_sentence"], sent), token_containment(sent, g["gap_sentence"])) >= 0.8:
            return g["gap_id"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=Path("data/gap_head.joblib"))
    ap.add_argument("--report", type=Path, default=Path("docs/experiments/funnel_demo_output.md"))
    ap.add_argument("--tsv", type=Path, default=Path("data/demo/funnel_gaps.tsv"))
    args = ap.parse_args()

    head = None
    if args.head.exists():
        from gap2idea.pipeline.gap_funnel import EmbeddingGapHead
        head = EmbeddingGapHead.load(args.head)

    gold, texts = load_gold()
    items = list(texts.items())
    n = len(items)
    tot_full = tot_slice = tot_gaps = 0
    R = ["# Funnel demo — Stage A + Stage B output on the gold papers", "",
         f"Head: `{args.head}` (mode=hybrid). Each paper shows the reduction "
         "(full → Stage A slice → Stage B gaps), every emitted gap (flagged vs "
         "gold), and a sample of slice sentences Stage B rejected.", ""]
    all_rows = []

    for pid, rec in items:
        full = split_sentences(_cut_before_references(rec["text"]))
        regions = slice_terminal_regions(rec["text"], blocks=rec["blocks"])
        slice_sents = [s for r in regions for s in r.sentences]
        gaps = extract_gaps(pid, rec["text"], blocks=rec["blocks"], head=head, mode="hybrid")
        tot_full += len(full); tot_slice += len(slice_sents); tot_gaps += len(gaps)
        gp = gold[gold["paper_id"] == pid]
        dropA = 100 * (1 - len(slice_sents) / max(1, len(full)))
        dropB = 100 * (1 - len(gaps) / max(1, len(slice_sents)))
        regstr = ", ".join(f"{r.section_type}({len(r.sentences)})" for r in regions)
        R += [f"## {pid}  ·  gold gaps: {len(gp)}", "",
              f"- **Stage A:** {len(full)} sentences → **{len(slice_sents)}** in slice "
              f"(−{dropA:.0f}%, free) · regions: {regstr}",
              f"- **Stage B:** {len(slice_sents)} slice → **{len(gaps)}** gaps "
              f"(−{dropB:.0f}% of slice)", "", "| type | source | section | gold? | gap sentence |",
              "|---|---|---|---|---|"]
        emitted_keys = set()
        for g in gaps:
            m = gold_match(g["gap_sentence"], gp)
            emitted_keys.add(m)
            all_rows.append({"paper_id": pid, "gap_type": g["gap_type"], "source": g["source"],
                             "section_type": g["section_type"], "gold_match": m or "",
                             "gap_sentence": g["gap_sentence"]})
            R.append(f"| {g['gap_type']} | {g['source']} | {g['section_type']} | "
                     f"{('✅ '+str(m)) if m else '— extra'} | {short(g['gap_sentence'], 120)} |")
        missed = [g["gap_id"] for _, g in gp.iterrows() if g["gap_id"] not in emitted_keys]
        if missed:
            R += ["", f"**Gold missed:** {missed}"]
        gap_norm = {short(g["gap_sentence"], 60) for g in gaps}
        rejected = [s for s in slice_sents if short(s, 60) not in gap_norm][:4]
        if rejected:
            R += ["", "_Stage B rejected (sample non-gaps Stage A kept):_"]
            R += [f"- {short(s, 110)}" for s in rejected]
        R += [""]

    R += ["---", "## Aggregate reduction", "",
          f"| stage | total | per paper | drop |", "|---|---|---|---|",
          f"| full body sentences | {tot_full} | {tot_full/n:.0f} | — |",
          f"| → Stage A slice | {tot_slice} | {tot_slice/n:.0f} | −{100*(1-tot_slice/tot_full):.0f}% (free) |",
          f"| → Stage B gaps | {tot_gaps} | {tot_gaps/n:.1f} | "
          f"−{100*(1-tot_gaps/tot_slice):.0f}% of slice, −{100*(1-tot_gaps/tot_full):.1f}% of full |"]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.tsv.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(R), encoding="utf-8")
    import pandas as pd
    pd.DataFrame(all_rows).to_csv(args.tsv, sep="\t", index=False)
    print(f"papers={n}  full={tot_full}  slice={tot_slice}  gaps={tot_gaps}")
    print(f"readable report -> {args.report}")
    print(f"all gaps (tsv)  -> {args.tsv}")


if __name__ == "__main__":
    main()
