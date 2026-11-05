# -*- coding: utf-8 -*-
"""The three Stage C modes on SAL-40, scored against SAL's human limitation labels.

Candidates: the 391 PubMedBERT A+B positives, in document order, +/-1 sentence context.
Rubrics are the frozen ones from stage_c_rubric_val.py, used in two scopes fixed before any run:
  limitation   primary: rubric narrowed to limitations, because SAL labels nothing else
  all_gaps     sensitivity: the gold rubric unchanged, so kept future work counts against it
Secondary scoring uses the LLM gold on the same sentences, because SAL is known to under-label.
"""
import json
import random
import sys
import time

from stage_c_prompt_vs_model import ROOT, qwen_client
from stage_c_rubric_val import BATCH, KEEP_BIAS, RUBRICS, judge

ROWS = ROOT / "stage_c_package/sal_criteria/sal_test_allmodels.json"
GOLD = ROOT / "data/gold/gap_gold_final.jsonl"
RCT_LOOSE = ROOT / "docs/experiments/stage_c_sal40_qwen.json"
OUT = ROOT / "docs/experiments/stage_c_sal40_modes.json"
MODES = {"high_recall": "keep_biased", "balanced": "neutral", "high_precision": "strict"}
KEEP_SECTIONS = {"DISCUSS", "CONCL"}


def swap(text, old, new):
    if old not in text:
        sys.exit(f"Rubric text changed; not found: {old[:50]}")
    return text.replace(old, new)


def limitation_scope(rubric):
    start = rubric.index("  (b) an explicit FUTURE-WORK")
    end = rubric.index("Answer NO (drop)")
    text = rubric[:start].rstrip() + "\n\n" + rubric[end:]
    text = swap(text, "does it genuinely state a research gap?", "does it genuinely state a limitation of the authors' own study?")
    text = swap(text, "if the sentence states ANY of:", "if the sentence states:")
    text = swap(text, "if the sentence is instead: ", "if the sentence is instead: a future-work direction, an open problem in the field or prior work, ")
    if KEEP_BIAS in rubric:
        text = swap(text, "whether a sentence is a real gap", "whether a sentence is a real limitation")
        text = swap(text, "clearly is NOT a gap", "clearly is NOT a limitation")
    if "Be STRICT" in rubric:
        text = swap(text, "explicitly states the limitation, the proposed direction, or the unsolved problem",
                    "explicitly states the limitation")
        text = swap(text, "names no specific gap", "names no specific limitation")
    return text


def run_one(tag, prompt, pool, client):
    ckpt = ROOT / f"docs/experiments/stage_c_sal40_modes_{tag}.jsonl"
    done = {}
    if ckpt.exists():
        for raw in ckpt.read_text(encoding="utf-8").splitlines():
            r = json.loads(raw)
            done[r["i"]] = r["keep"]
    failed, t0 = [], time.time()
    with ckpt.open("a", encoding="utf-8") as fh:
        for start in range(0, len(pool), BATCH):
            idx = range(start, min(start + BATCH, len(pool)))
            if all(i in done for i in idx):
                continue
            got = {}
            for _attempt in range(2):
                try:
                    got = judge(client, pool, start, prompt)
                    if len(got) == len(idx):
                        break
                except Exception as e:  # reported; never counted as a verdict
                    print(f"{tag} batch {start}: {type(e).__name__}: {e}", flush=True)
            if len(got) < len(idx):
                failed.append(start)
            for i, keep in sorted(got.items()):
                done[i] = keep
                fh.write(json.dumps({"i": i, "keep": keep}) + "\n")
            fh.flush()
    print(f"{tag} {len(done)}/{len(pool)} [{time.time() - t0:.0f}s]", flush=True)
    return done, failed


def prf(tp, fp, gaps):
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / gaps if gaps else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def table(rows, kept, label):
    out = {}
    for i, r in enumerate(rows):
        c = out.setdefault(r["pmcid"], [0, 0, 0])
        c[0] += i in kept and label(i)
        c[1] += i in kept and not label(i)
        c[2] += label(i)
    return out


def scored(rows, kept, label, n=2000):
    t = table(rows, kept, label)
    papers, rng, draws = list(t), random.Random(0), []
    for _ in range(n):
        pick = rng.choices(papers, k=len(papers))
        draws.append(prf(*[sum(t[p][j] for p in pick) for j in range(3)]))
    totals = [sum(t[p][j] for p in papers) for j in range(3)]
    res = {"kept": totals[0] + totals[1], "tp": totals[0], "fp": totals[1], "gaps": totals[2]}
    for j, m in enumerate(("P", "R", "F1")):
        vals = sorted(d[j] for d in draws)
        res[m] = {"value": round(prf(*totals)[j], 3), "ci95": [round(vals[int(0.025 * n)], 3), round(vals[int(0.975 * n)], 3)]}
    return res


def paired(rows, a, b, label, n=2000):
    ta, tb = table(rows, a, label), table(rows, b, label)
    papers, rng, diffs = list(ta), random.Random(0), []
    for _ in range(n):
        pick = rng.choices(papers, k=len(papers))
        diffs.append(tuple(prf(*[sum(t[p][j] for p in pick) for j in range(3)])[m] for m in (0, 2) for t in (ta, tb)))
    pd = sorted(d[0] - d[1] for d in diffs)
    fd = sorted(d[2] - d[3] for d in diffs)
    point_a, point_b = [prf(*[sum(t[p][j] for p in papers) for j in range(3)]) for t in (ta, tb)]
    ci = lambda v: [round(v[int(0.025 * n)], 3), round(v[int(0.975 * n)], 3)]
    return {"precision_diff": round(point_a[0] - point_b[0], 3), "precision_ci95": ci(pd),
            "f1_diff": round(point_a[2] - point_b[2], 3), "f1_ci95": ci(fd)}


def main():
    rows = json.load(open(ROWS, encoding="utf-8"))
    idx_in_paper, seen = [], {}
    for r in rows:
        idx_in_paper.append(seen.get(r["pmcid"], 0))
        seen[r["pmcid"]] = idx_in_paper[-1] + 1
    gold = {}
    for raw in GOLD.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            g = json.loads(raw)
            gold[(g["paper_id"], g["idx"])] = g["label"]
    llm = [gold.get((r["pmcid"], k), "none") for r, k in zip(rows, idx_in_paper)]

    def window(i):
        return " ".join(rows[j]["sent"] for j in range(i - 1, i + 2)
                        if 0 <= j < len(rows) and rows[j]["pmcid"] == rows[i]["pmcid"])

    ids = sorted(i for i, r in enumerate(rows) if r["sec"] in KEEP_SECTIONS and r["preds"]["PubMedBERT-FT"] == 1)
    if len(ids) != 391 or sum(r["label"] for r in rows) != 205:
        sys.exit("SAL-40 does not match the 391 candidates and 205 limitations used elsewhere")
    pool = [{"gap_type": "limitation", "gap_sentence": rows[i]["sent"], "paragraph_text": window(i)} for i in ids]

    sal = lambda i: rows[i]["label"] == 1
    llm_lim = lambda i: llm[i] == "limitation"
    llm_any = lambda i: llm[i] != "none"
    before = set(ids)
    loose = {int(k) for k, v in json.loads(RCT_LOOSE.read_text(encoding="utf-8"))["qwen_verdicts"].items() if v}
    result = {"candidates": len(ids), "before_stage_c": scored(rows, before, sal), "rct_loose_qwen": scored(rows, loose, sal)}
    client = qwen_client()
    for scope in ("limitation", "all_gaps"):
        for mode, rubric_name in MODES.items():
            prompt = RUBRICS[rubric_name]
            if scope == "limitation":
                prompt = limitation_scope(prompt)
            done, failed = run_one(f"{scope}_{mode}", prompt, pool, client)
            if failed or len(done) < len(pool):
                sys.exit(f"{scope} {mode}: incomplete, failed batches {failed}. Rerun to resume.")
            kept = {ids[i] for i, k in done.items() if k}
            result[f"{scope}_{mode}"] = {
                "sal_human": scored(rows, kept, sal),
                "llm_limitation": scored(rows, kept, llm_lim),
                "llm_any_gap": scored(rows, kept, llm_any),
                "vs_before_stage_c": paired(rows, kept, before, sal),
                "vs_rct_loose_qwen": paired(rows, kept, loose, sal),
            }
            OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
