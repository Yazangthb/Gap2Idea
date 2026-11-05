# -*- coding: utf-8 -*-
"""SAL-40 Stage C with only the model changed: Yandex (0.658 headline) -> Qwen3.8-27B.

Same 391 PubMedBERT A+B positives in document order, rct_loose prompt, +/-1 inline
context, batch 20, same LLMGapFilter code path as stagec_faithful_pubmed.py. Labels are
SAL human gold. Stops unless the saved Yandex verdicts reproduce 269 kept, 156 TP.
Writes docs/experiments/stage_c_sal40_qwen.json (gitignored).
"""
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from stage_c_prompt_vs_model import qwen_client
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
from bench_stagec_opt import SHOTS_RCT, SYSTEM_RCT_LOOSE

PKG = ROOT / "stage_c_package" / "sal_criteria"
OUT = ROOT / "docs" / "experiments" / "stage_c_sal40_qwen.json"
EXTRA = {"seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
KEEP_SECTIONS = {"DISCUSS", "CONCL"}


def prf(tp, fp, gaps):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gaps if gaps else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rounded(values):
    return [round(v, 3) for v in values]


def main():
    rows = json.load(open(PKG / "sal_test_allmodels.json", encoding="utf-8"))

    def window(i):
        r = rows[i]
        return " ".join(rows[j]["sent"] for j in range(i - 1, i + 2)
                        if 0 <= j < len(rows) and rows[j]["pmcid"] == r["pmcid"])

    gold_all = sum(r["label"] for r in rows)
    pb = sorted(i for i, r in enumerate(rows)
                if r["sec"] in KEEP_SECTIONS and r["preds"]["PubMedBERT-FT"] == 1)

    saved = json.load(open(PKG / "faithful_pubmed_verdicts.json", encoding="utf-8"))
    yandex = {int(k): bool(v) for k, v in saved["rct"].items()}
    y_tp = sum(yandex[i] and rows[i]["label"] == 1 for i in pb)
    y_fp = sum(yandex[i] and rows[i]["label"] != 1 for i in pb)
    if (len(pb), y_tp, y_tp + y_fp) != (391, 156, 269):
        sys.exit(f"Baseline not reproduced: {len(pb)} candidates, {y_tp} TP, {y_tp + y_fp} kept")
    print(f"baseline OK: 391 candidates, Yandex kept 269, 156 TP, F1 {prf(y_tp, y_fp, gold_all)[2]:.3f}", flush=True)

    filt = LLMGapFilter(backend="api", model="qwen3.8-27b", mode="validate_rct",
                        client=qwen_client(), context_style="inline")
    filt._sys, filt._shots = SYSTEM_RCT_LOOSE, SHOTS_RCT
    filt.extra_body = EXTRA
    qwen, batches, t0 = {}, 0, time.time()
    for k in range(0, len(pb), 20):
        chunk = pb[k:k + 20]
        items = [{"sentence": rows[i]["sent"], "context": window(i)} for i in chunk]
        for i, keep in zip(chunk, filt.judge_batch(items)):
            qwen[i] = bool(keep)
        batches += 1
        print(f"{min(k + 20, len(pb))}/{len(pb)} [{time.time() - t0:.0f}s]", flush=True)
    fallbacks = filt.n_calls - batches

    gaps_per_paper = defaultdict(int)
    for r in rows:
        gaps_per_paper[r["pmcid"]] += r["label"]
    by_paper = defaultdict(list)
    for i in pb:
        by_paper[rows[i]["pmcid"]].append((rows[i]["label"] == 1, yandex[i], qwen[i]))

    def tally(entries, column):
        tp = sum(gap and e[column] for e in entries for gap in [e[0]])
        fp = sum((not e[0]) and e[column] for e in entries)
        return tp, fp

    every = [e for v in by_paper.values() for e in v]
    q_tp, q_fp = tally(every, 2)
    pos = [e for e in every if e[0]]
    neg = [e for e in every if not e[0]]

    # Papers are resampled because sentences within a paper are not independent.
    rng, papers, stats = random.Random(0), list(gaps_per_paper), []
    for _ in range(2000):
        pick = rng.choices(papers, k=len(papers))
        entries = [e for p in pick for e in by_paper[p]]
        gaps = sum(gaps_per_paper[p] for p in pick)
        fy = prf(*tally(entries, 1), gaps)[2]
        fq = prf(*tally(entries, 2), gaps)[2]
        stats.append((fq, fy, fq - fy))
    ci = lambda j: [round(sorted(s[j] for s in stats)[q], 3) for q in (50, 1949)]

    result = {
        "config": {"model": "qwen3.8-27b", "prompt": "rct_loose", "batch": 20, **EXTRA,
                   "batches": batches, "fallback_calls": fallbacks, "seconds": round(time.time() - t0, 1)},
        "gold_limitations": gold_all, "candidates": len(pb), "candidate_gaps": len(pos),
        "A+B": rounded(prf(len(pos), len(neg), gold_all)),
        "yandex": {"kept": y_tp + y_fp, "tp": y_tp, "fp": y_fp, "PRF": rounded(prf(y_tp, y_fp, gold_all)),
                   "f1_ci95": ci(1)},
        "qwen": {"kept": q_tp + q_fp, "tp": q_tp, "fp": q_fp, "PRF": rounded(prf(q_tp, q_fp, gold_all)),
                 "f1_ci95": ci(0)},
        "paired": {"f1_diff_qwen_minus_yandex": round(prf(q_tp, q_fp, gold_all)[2] - prf(y_tp, y_fp, gold_all)[2], 3),
                   "diff_ci95": ci(2), "verdicts_agree": sum(e[1] == e[2] for e in every),
                   "gaps_only_yandex_kept": sum(e[1] and not e[2] for e in pos),
                   "gaps_only_qwen_kept": sum(e[2] and not e[1] for e in pos),
                   "negatives_only_yandex_kept": sum(e[1] and not e[2] for e in neg),
                   "negatives_only_qwen_kept": sum(e[2] and not e[1] for e in neg)},
        "qwen_verdicts": {str(i): v for i, v in qwen.items()},
    }
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "qwen_verdicts"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
