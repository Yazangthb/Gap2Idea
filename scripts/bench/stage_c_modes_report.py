# -*- coding: utf-8 -*-
"""Paper-ready numbers for the three Stage C modes, from the saved test verdicts.

No model calls. Reads stage_c_rubric_test_*.jsonl and writes stage_c_modes.json (gitignored).
CIs resample papers (2,000 draws, seed 0). Biomedical has only 6 papers, so its CIs are wide.
"""
import json
import random

from stage_c_nli_qwen import ROOT, TAUS, candidates, prf, setup

MODES = {"high_recall": "keep_biased", "balanced": "neutral", "high_precision": "strict"}
OUT = ROOT / "docs/experiments/stage_c_modes.json"


def kept_set(pool, name):
    kept = set()
    for raw in (ROOT / f"docs/experiments/stage_c_rubric_test_{name}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        if r["keep"]:
            kept.add((pool[r["i"]]["id"], pool[r["i"]]["idx"]))
    return kept


def scores(rows, kept, n=2000):
    table = {}
    for r in rows:
        c = table.setdefault(r["pid"], [0, 0, 0])
        gap, k = r["label"] != "none", (r["pid"], r["idx"]) in kept
        c[0] += k and gap
        c[1] += k and not gap
        c[2] += gap
    papers, rng, draws = list(table), random.Random(0), []
    for _ in range(n):
        pick = rng.choices(papers, k=len(papers))
        draws.append(prf(*[sum(table[p][j] for p in pick) for j in range(3)]))
    totals = [sum(table[p][j] for p in papers) for j in range(3)]
    point = prf(*totals)
    out = {"papers": len(papers), "kept": totals[0] + totals[1], "tp": totals[0], "fp": totals[1], "gaps": totals[2]}
    for j, m in enumerate(("P", "R", "F1")):
        vals = sorted(d[j] for d in draws)
        out[m] = {"value": round(point[j], 3), "ci95": [round(vals[int(0.025 * n)], 3), round(vals[int(0.975 * n)], 3)]}
    return out


def main():
    rows, typed, _ = setup()
    pool = candidates(rows, typed, TAUS["r90"])
    result = {"tau": TAUS["r90"], "candidates": len(pool)}
    for mode, rubric in MODES.items():
        kept = kept_set(pool, rubric)
        result[mode] = {
            "rubric": rubric,
            "all": scores(rows, kept),
            "arxiv": scores([r for r in rows if r["dom"] == "arxiv_cs"], kept),
            "biomedical": scores([r for r in rows if r["dom"] == "biomed_sal"], kept),
            "no_research_gap": scores([r for r in rows if r["label"] != "research_gap"], kept),
            "recall_by_type": {t: round(sum((r["pid"], r["idx"]) in kept for r in rows if r["label"] == t)
                                        / sum(r["label"] == t for r in rows), 3)
                               for t in ("limitation", "future_work", "research_gap")},
        }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for mode in MODES:
        r = result[mode]
        for part in ("all", "arxiv", "biomedical", "no_research_gap"):
            s = r[part]
            print(mode, part, s["papers"], s["kept"], s["tp"], s["fp"], s["gaps"],
                  *[f'{m} {s[m]["value"]} {s[m]["ci95"]}' for m in ("P", "R", "F1")])


if __name__ == "__main__":
    main()
