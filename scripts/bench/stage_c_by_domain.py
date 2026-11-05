# -*- coding: utf-8 -*-
"""Split the full Qwen Stage C run by domain, with exact per-domain gap totals, and
report how the gold test papers relate to the human-labelled SAL-40 benchmark.
Makes no API calls. Writes docs/experiments/stage_c_by_domain.json (gitignored)."""
import json
import random
import re
from collections import Counter, defaultdict

from stage_c_full_qwen import CKPT, ROOT, load_in_reading_order
from stage_c_heldout import selection_keys

GOLD = ROOT / "data/gold/gap_gold_final.jsonl"
SPLITS = ROOT / "data/gold/splits.json"
SAL = ROOT / "stage_c_package/sal_criteria/sal_test_allmodels.json"
OUT = ROOT / "docs/experiments/stage_c_by_domain.json"
TYPES = ("limitation", "future_work", "research_gap")


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def norm(s):
    return re.sub(r"\W+", " ", s).strip().lower()


def main():
    splits = json.loads(SPLITS.read_text(encoding="utf-8"))
    test = set(splits["test"])
    domain, totals, type_totals = {}, Counter(), defaultdict(Counter)
    split_of = {p: name for name in ("train", "val", "test") for p in splits[name]}
    for line in GOLD.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        domain[g["paper_id"]] = g["domain"]
        if g["paper_id"] in test and g["label"] != "none":
            totals[g["paper_id"]] += 1
            type_totals[g["domain"]][g["label"]] += 1

    pool = load_in_reading_order()
    verdict = {}
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        verdict[r["i"]] = r["keep"]
    rows = [(c, verdict[i]) for i, c in enumerate(pool)]
    seen = selection_keys()

    result = {}
    for dom in ("arxiv_cs", "biomed_sal"):
        dom_rows = [(c, k) for c, k in rows if domain[c["id"]] == dom]
        papers = sorted(p for p in test if domain[p] == dom)
        by_paper = defaultdict(list)
        for c, k in dom_rows:
            by_paper[c["id"]].append((c, k))

        def score(paper_list, keep_filter=lambda c: True):
            tp = fp = before_tp = before_fp = gaps = 0
            for p in paper_list:
                gaps += totals[p]
                for c, k in by_paper[p]:
                    if not keep_filter(c):
                        continue
                    gap = c["label"] != "none"
                    before_tp += gap
                    before_fp += not gap
                    tp += k and gap
                    fp += k and not gap
            return tp, fp, before_tp, before_fp, gaps

        tp, fp, btp, bfp, gaps = score(papers)
        p_b, r_b = btp / (btp + bfp), btp / gaps
        p_a, r_a = tp / (tp + fp), tp / gaps
        rng, stats = random.Random(0), []
        for _ in range(2000):
            t, f, _bt, _bf, g = score(rng.choices(papers, k=len(papers)))
            pr, re_ = t / (t + f) if t + f else 0.0, t / g if g else 0.0
            stats.append((pr, re_, f1(pr, re_)))
        ci = lambda j: [round(sorted(s[j] for s in stats)[q], 3) for q in (50, 1949)]

        h_tp, h_fp, h_btp, _h, _g = score(papers, lambda c: (c["id"], c["idx"]) not in seen)
        h_p, h_kept = h_tp / (h_tp + h_fp), h_tp / h_btp
        kept_type = Counter(c["label"] for c, k in dom_rows if k and c["label"] != "none")
        cand_type = Counter(c["label"] for c, _ in dom_rows if c["label"] != "none")
        result[dom] = {
            "papers": len(papers), "gaps": gaps, "candidates": btp + bfp, "reach_stage_c": btp,
            "before": {"P": round(p_b, 3), "R": round(r_b, 3), "F1": round(f1(p_b, r_b), 3)},
            "after": {"P": round(p_a, 3), "R": round(r_a, 3), "F1": round(f1(p_a, r_a), 3), "tp": tp, "fp": fp},
            "gaps_kept": round(tp / btp, 3),
            "ci95": {"P": ci(0), "R": ci(1), "F1": ci(2)},
            "held_out": {"tp": h_tp, "fp": h_fp, "reach": h_btp, "P": round(h_p, 3), "gaps_kept": round(h_kept, 3),
                         "R_assuming_same_reach": round(h_kept * r_b, 3),
                         "F1": round(f1(h_p, h_kept * r_b), 3)},
            "recall_by_type": {t: {"n": type_totals[dom][t], "before": round(cand_type[t] / type_totals[dom][t], 3),
                                   "after": round(kept_type[t] / type_totals[dom][t], 3)}
                               for t in TYPES if type_totals[dom][t]},
        }

    # Relation to the human-labelled SAL-40 test set.
    sal = json.loads(SAL.read_text(encoding="utf-8"))
    sal_papers = {r["pmcid"] for r in sal}
    biomed = {p for p, d in domain.items() if d == "biomed_sal"}
    sal_sent = {(r["pmcid"], norm(r["sent"])): r["label"] for r in sal}
    matched = [(c, k, sal_sent[(c["id"], norm(c["gap_sentence"]))]) for c, k in rows
               if (c["id"], norm(c["gap_sentence"])) in sal_sent]
    h_tp = sum(k and s == 1 for _, k, s in matched)
    h_fp = sum(k and s != 1 for _, k, s in matched)
    result["sal_relation"] = {
        "sal40_papers": len(sal_papers),
        "biomed_gold_papers": len(biomed),
        "sal40_in_gold_by_split": dict(Counter(split_of.get(p, "not_in_gold") for p in sal_papers)),
        "biomed_test_papers_in_sal40": len([p for p in biomed & test if p in sal_papers]),
        "candidates_matched_to_sal_labels": len(matched),
        "sal_human_limitations_among_matched": sum(s == 1 for *_, s in matched),
        "qwen_kept_with_sal_label": {"tp": h_tp, "fp": h_fp},
        "llm_gold_vs_sal_on_matched": dict(Counter(f"{c['label']}|sal={s}" for c, _, s in matched)),
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
