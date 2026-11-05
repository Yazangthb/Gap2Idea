# -*- coding: utf-8 -*-
"""Score the full Qwen Stage C run on candidates that played no part in choosing the setup.

Selection used: the first 120 candidates in file order (Yandex, 2026-10-26) and the 80 from
random.seed(11) (Sonnet, then Yandex and Qwen). Makes no API calls.
  sentence level: drop those 200-odd sentences, keep the rest of their papers
  paper level:    drop every paper that contributed any of those sentences
End-to-end recall assumes the A+B share of gaps reaching Stage C (480/664) also holds here.
"""
import json
import random
from collections import defaultdict

from stage_c_full_qwen import CANDS, CKPT, load_in_reading_order

A_B_RECALL = 480 / 664


def scores(rows):
    gaps = sum(c["label"] != "none" for c, _ in rows)
    tp = sum(k and c["label"] != "none" for c, k in rows)
    fp = sum(k and c["label"] == "none" for c, k in rows)
    prec, kept = tp / (tp + fp), tp / gaps
    rec = kept * A_B_RECALL
    return {"papers": len({c["id"] for c, _ in rows}), "candidates": len(rows), "gaps": gaps, "tp": tp, "fp": fp,
            "precision_before": round(gaps / len(rows), 3), "precision": round(prec, 3),
            "gaps_kept": round(kept, 3), "recall": round(rec, 3), "f1": round(2 * prec * rec / (prec + rec), 3)}


def with_ci(rows):
    by_paper = defaultdict(list)
    for c, k in rows:
        by_paper[c["id"]].append((c, k))
    rng, papers, stats, n = random.Random(0), list(by_paper), [], 2000
    for _ in range(n):
        picked = [row for p in rng.choices(papers, k=len(papers)) for row in by_paper[p]]
        gaps = sum(c["label"] != "none" for c, _ in picked)
        tp = sum(k and c["label"] != "none" for c, k in picked)
        fp = sum(k and c["label"] == "none" for c, k in picked)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / gaps * A_B_RECALL if gaps else 0.0
        stats.append((prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0))
    ci = lambda j: [round(sorted(s[j] for s in stats)[int(q * n)], 3) for q in (0.025, 0.975)]
    biomed = sum(p.startswith("PMC") for p in by_paper)
    return {**scores(rows), "biomed_papers": biomed,
            "ci95": {"precision": ci(0), "recall": ci(1), "f1": ci(2)}}


def selection_keys():
    """(paper, sentence index) of every candidate used to choose the setup."""
    file_order = json.loads(CANDS.read_text(encoding="utf-8"))
    first_120 = {(c["id"], c["idx"]) for c in file_order[:120]}
    shuffled = list(file_order)
    random.seed(11)
    random.shuffle(shuffled)
    return first_120 | {(c["id"], c["idx"]) for c in shuffled[:80]}


def main():
    seen = selection_keys()
    seen_papers = {p for p, _ in seen}

    pool = load_in_reading_order()
    verdicts = {}
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        verdicts[r["i"]] = r["keep"]
    rows = [(c, verdicts[i]) for i, c in enumerate(pool)]

    print(f"selection sentences: {len(seen)}, from {len(seen_papers)} papers")
    print("all         ", with_ci(rows))
    print("seen only   ", scores([(c, k) for c, k in rows if (c["id"], c["idx"]) in seen]))
    print("sentence-out", with_ci([(c, k) for c, k in rows if (c["id"], c["idx"]) not in seen]))
    clean = [(c, k) for c, k in rows if c["id"] not in seen_papers]
    print("paper-out   ", with_ci(clean) if clean else "no papers left")


if __name__ == "__main__":
    main()
