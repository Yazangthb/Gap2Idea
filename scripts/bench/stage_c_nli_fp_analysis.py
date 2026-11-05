# -*- coding: utf-8 -*-
"""Per-stage numbers and false-positive breakdown for short NLI (tau 0.354) + Qwen. No API calls."""
import json
import random
import sys
from collections import Counter, defaultdict

import numpy as np

from nli_variants import ROOT
from stage_c_nli_qwen import TAUS, candidates, kept_set, setup

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def line(name, kept, tp, gaps=664):
    print(f"{name:28} kept {kept:6}  gaps {tp:4}  P {tp / kept:.3f}  R {tp / gaps:.3f}  F1 {2 * tp / (kept + gaps):.3f}")


def main():
    rows, typed, _ = setup()
    n, gold_gaps = len(rows), sum(r["label"] != "none" for r in rows)
    line("All sentences", n, gold_gaps)
    a = [r for r in rows if not r["junk"]]
    line("A: junk filter", len(a), sum(r["label"] != "none" for r in a))
    pool = candidates(rows, typed, TAUS["r90"])
    line("B: NLI tau 0.354", len(pool), sum(c["label"] != "none" for c in pool))
    kept = kept_set(rows, typed, "r90")
    fps = [c for c in pool if (c["id"], c["idx"]) in kept and c["label"] == "none"]
    tps = [c for c in pool if (c["id"], c["idx"]) in kept and c["label"] != "none"]
    line("C: Qwen", len(kept), len(tps))
    print("kept share of all sentences:", round(len(kept) / n, 3), "| gold prevalence:", round(gold_gaps / n, 3))

    dom = {r["pid"]: r["dom"] for r in rows}
    print("\nFP by domain:", dict(Counter(dom[c["id"]] for c in fps)), "| TP:", dict(Counter(dom[c["id"]] for c in tps)))
    for label, items in (("FP", fps), ("TP", tps)):
        print(f"{label} by predicted type:", dict(Counter(c["gap_type"] for c in items)))
    per_paper = Counter(c["id"] for c in fps)
    top = per_paper.most_common()
    print("FP per paper: median", int(np.median(list(per_paper.values()))), "| top 5 share",
          round(sum(v for _, v in top[:5]) / len(fps), 3), "| top 5", top[:5])
    prec = []
    by_paper = defaultdict(lambda: [0, 0])
    for c in tps:
        by_paper[c["id"]][0] += 1
    for c in fps:
        by_paper[c["id"]][1] += 1
    prec = sorted(t / (t + f) for t, f in by_paper.values())
    print("per-paper precision: min", round(prec[0], 3), "quartiles", [round(prec[int(len(prec) * q)], 3) for q in (.25, .5, .75)], "max", round(prec[-1], 3))

    # Did either labelling judge flag this sentence before adjudication?
    judged = {}
    for raw in (ROOT / "data/gold/gap_gold_sentences.jsonl").read_text(encoding="utf-8").splitlines():
        if raw.strip():
            g = json.loads(raw)
            judged[(g.get("paper_id"), g.get("idx"))] = g
    sample_key = next(iter(judged.values()))
    print("\njudge file keys:", sorted(sample_key))
    fp_flagged = [c for c in fps if (c["id"], c["idx"]) in judged]
    print("FP flagged by at least one judge, then rejected:", len(fp_flagged), "of", len(fps))
    cand_neg = [c for c in pool if c["label"] == "none"]
    flagged_neg = sum((c["id"], c["idx"]) in judged for c in cand_neg)
    print("Same share among all negatives reaching Stage C:", flagged_neg, "of", len(cand_neg))

    rng = random.Random(7)
    arxiv = [c for c in fps if dom[c["id"]] == "arxiv_cs"]
    print("\n--- random sample of 40 arXiv false positives")
    for c in rng.sample(arxiv, 40):
        print(f"[{c['gap_type'][:3]}] {c['gap_sentence'][:190]}")
    print("\n--- all biomedical false positives")
    for c in [c for c in fps if dom[c["id"]] != "arxiv_cs"]:
        print(f"[{c['gap_type'][:3]}] {c['gap_sentence'][:150]}")


if __name__ == "__main__":
    main()
