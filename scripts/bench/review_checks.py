# -*- coding: utf-8 -*-
"""Two checks requested in review, from saved verdicts only (no LLM calls).

1. Paper-level exclusion: gold test F1 of the published funnel without every paper that contributed any of the
   196 development candidates.
2. Length split: on the 28 arXiv papers, F1 of whole-paper prompting and of the published funnel for papers at or
   below the median length and above it, per model, high-precision mode.
Writes docs/experiments/review_checks.json.
"""
import json
import statistics

import funnel_nli_ft as f
import llm_size_compare as L
from stage_c_heldout import selection_keys

ROOT = f.ROOT


def main():
    out = {}
    f.STAGE_B = "zero_shot"
    rows = f.SETS["gold"]()["rows"]
    seen_papers = {p for p, _ in selection_keys()}
    keep_rows = [r for r in rows if r["pid"] not in seen_papers]
    out["paper_exclusion"] = {"papers_excluded": len(seen_papers & {r["pid"] for r in rows}),
                              "papers_left": len({r["pid"] for r in keep_rows}),
                              "gaps_left": sum(r["label"] != "none" for r in keep_rows)}
    for mode in f.MODES:
        kept = f.published("gold", rows, mode)
        out["paper_exclusion"][mode] = {"all": f.prf(rows, kept), "without_dev_papers": f.prf(keep_rows, kept)}

    arx = [r for r in rows if r["dom"] == "arxiv_cs"]
    size = {}
    for r in arx:
        size[r["pid"]] = size.get(r["pid"], 0) + 1
    med = statistics.median(size.values())
    bins = {"short": {p for p, n in size.items() if n <= med}, "long": {p for p, n in size.items() if n > med}}
    out["length_split"] = {"median_sentences": med, "papers": {k: len(v) for k, v in bins.items()}}
    for model in ("27b", "4b", "9b"):
        try:
            base, _ = L.whole(model, "gold", "high_precision")
        except Exception:
            continue
        if model == "27b":
            fun = f.published("gold", rows, "high_precision")
        else:
            folder = L.SMALL[model] / "funnel_zs_batch20" if isinstance(L.SMALL, dict) else L.SMALL / "funnel_zs_batch20"
            if not (folder / "gold_high_precision.jsonl").exists():
                continue
            try:
                fun, _ = L.batch20(folder, "zero_shot", "gold", "high_precision")
            except AssertionError:  # run not finished
                continue
            f.STAGE_B = "zero_shot"
        res = {}
        for k, papers in bins.items():
            R = [r for r in arx if r["pid"] in papers]
            res[k] = {"whole": f.prf(R, base), "funnel": f.prf(R, fun)}
        out["length_split"][model] = res
    (ROOT / "docs/experiments/review_checks.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
