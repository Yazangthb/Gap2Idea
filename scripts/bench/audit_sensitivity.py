# -*- coding: utf-8 -*-
"""How much do the headline numbers depend on questionable choices? Saved verdicts only, no API calls.

Gold corpus (Qwen, 1,438 candidates):
  1. drop research_gap: its sentences are removed from scoring entirely
  2. high-confidence gold only: adjudicated gaps are removed from scoring entirely
SAL-40 (Qwen, rct_loose):
  3. Lan's Stage A filter: abstracts also pass, and each flagged abstract sentence counts as
     an unfiltered false positive (a worst case, since Stage C never judged them)
"""
import json
from collections import Counter

from stage_c_full_qwen import CKPT, ROOT, load_in_reading_order


def f1(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def line(name, tp, fp, gaps, before_tp=None, before_fp=None):
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / gaps
    out = f"{name:44} gaps {gaps:4}  TP {tp:4} FP {fp:4}  P {p:.3f} R {r:.3f} F1 {f1(p, r):.3f}"
    if before_tp is not None:
        pb, rb = before_tp / (before_tp + before_fp), before_tp / gaps
        out += f"  | before P {pb:.3f} R {rb:.3f} F1 {f1(pb, rb):.3f}"
    print(out)


def main():
    conf, domain = {}, {}
    splits = json.loads((ROOT / "data/gold/splits.json").read_text(encoding="utf-8"))
    test = set(splits["test"])
    totals = Counter()
    for raw in (ROOT / "data/gold/gap_gold_final.jsonl").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        g = json.loads(raw)
        domain[g["paper_id"]] = g["domain"]
        if g["paper_id"] in test and g["label"] != "none":
            conf[(g["paper_id"], g["idx"])] = g.get("confidence")
            totals[(g["domain"], g["label"], g.get("confidence"))] += 1

    pool = load_in_reading_order()
    verdict = {}
    for raw in CKPT.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        verdict[r["i"]] = r["keep"]
    rows = [(c, verdict[i], domain[c["id"]]) for i, c in enumerate(pool)]
    print("test gaps by confidence:", dict(Counter(k[2] for k in totals.elements())))

    def scored(dom, keep_row, count_total):
        chosen = [(c, k) for c, k, d in rows if (dom is None or d == dom) and keep_row(c)]
        gaps = sum(n for key, n in totals.items() if (dom is None or key[0] == dom) and count_total(key))
        tp = sum(k and c["label"] != "none" for c, k in chosen)
        fp = sum(k and c["label"] == "none" for c, k in chosen)
        btp = sum(c["label"] != "none" for c, _ in chosen)
        bfp = sum(c["label"] == "none" for c, _ in chosen)
        return tp, fp, gaps, btp, bfp

    for dom, name in ((None, "all"), ("arxiv_cs", "arXiv"), ("biomed_sal", "biomedical")):
        print(f"\n--- {name}")
        line("baseline: all three types, all labels", *scored(dom, lambda c: True, lambda k: True))
        line("1. research_gap removed", *scored(dom, lambda c: c["label"] != "research_gap", lambda k: k[1] != "research_gap"))
        line("2. high-confidence labels only",
             *scored(dom, lambda c: c["label"] == "none" or conf.get((c["id"], c["idx"])) == "high", lambda k: k[2] == "high"))

    print("\n--- SAL-40")
    fx = json.load(open(ROOT / "stage_c_package/sal_criteria/sal_test_allmodels.json", encoding="utf-8"))
    qwen = json.loads((ROOT / "docs/experiments/stage_c_sal40_qwen.json").read_text(encoding="utf-8"))
    v = {int(k): x for k, x in qwen["qwen_verdicts"].items()}
    gold = sum(r["label"] for r in fx)
    tp = sum(keep and fx[i]["label"] == 1 for i, keep in v.items())
    fp = sum(keep and fx[i]["label"] != 1 for i, keep in v.items())
    line("Qwen, DISCUSS+CONCL", tp, fp, gold)
    flagged = [r for r in fx if r["sec"] == "ABSTRACT" and r["preds"]["PubMedBERT-FT"] == 1]
    print("abstract sentences:", sum(r["sec"] == "ABSTRACT" for r in fx), "| flagged by PubMedBERT:", len(flagged),
          "| of which gold:", sum(r["label"] for r in flagged))
    line("3. + flagged abstracts, unfiltered (worst case)", tp + sum(r["label"] for r in flagged),
         fp + sum(1 - r["label"] for r in flagged), gold)


if __name__ == "__main__":
    main()
