"""Benchmark Stage C (LLM precision filter) on the gold papers.

Runs the funnel (Stage A+B) then the LLM filter, and reports precision/recall vs
gold BEFORE and AFTER, plus exactly which predictions the LLM dropped (so we can
see it kills false positives — acknowledgments, formulas, math exposition — not
real gaps). Writes a readable report.

    python scripts/bench/bench_stage_c.py --backend local
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap2idea.pipeline.gap_funnel import extract_gaps, token_containment  # noqa: E402
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter  # noqa: E402
from bench_gap_recall import load as load_gold  # noqa: E402


def short(s, n=84):
    return textwrap.shorten(" ".join(str(s).split()), n)


def matches_gold(sent, gp):
    for _, g in gp.iterrows():
        if max(token_containment(g["gap_sentence"], sent), token_containment(sent, g["gap_sentence"])) >= 0.8:
            return g["gap_id"]
    return None


def stats(preds_by_paper, gold):
    n_pred = sum(len(v) for v in preds_by_paper.values())
    matched, tp = set(), 0
    for pid, prs in preds_by_paper.items():
        gp = gold[gold["paper_id"] == pid]
        for pr in prs:
            m = matches_gold(pr["gap_sentence"], gp)
            if m:
                matched.add(m); tp += 1
    recall = len(matched) / max(1, len(gold))
    prec_floor = tp / max(1, n_pred)
    return n_pred, len(matched), round(recall, 3), round(prec_floor, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="local")
    ap.add_argument("--model", default=None)
    ap.add_argument("--mode", default="validate", choices=["validate", "junk"])
    args = ap.parse_args()

    gold, texts = load_gold()
    from gap2idea.pipeline.gap_funnel import EmbeddingGapHead
    head = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")
    filt = LLMGapFilter(backend=args.backend, model=args.model, mode=args.mode)

    before, after, dropped = {}, {}, []
    for pid, rec in texts.items():
        gp = gold[gold["paper_id"] == pid]
        gaps = extract_gaps(pid, rec["text"], blocks=rec["blocks"], head=head, mode="hybrid")
        kept = filt.filter_gaps(gaps)
        before[pid], after[pid] = gaps, kept
        kept_keys = {short(k["gap_sentence"], 50) for k in kept}
        for g in gaps:
            if short(g["gap_sentence"], 50) not in kept_keys:
                dropped.append((pid, matches_gold(g["gap_sentence"], gp), g))

    nb, mb, rb, pb = stats(before, gold)
    na, ma, ra, pa = stats(after, gold)
    R = ["# Stage C — LLM precision filter (results)", "",
         f"Backend: {args.backend} ({filt.model}).  {filt.n_judged} LLM judgments "
         f"over {len(texts)} papers (~{filt.n_judged/len(texts):.1f}/paper).", "",
         "| | predictions | gold matched | recall | precision (floor) |",
         "|---|---|---|---|---|",
         f"| **before** (Stage A+B) | {nb} | {mb} | {rb} | {pb} |",
         f"| **after** (+Stage C) | {na} | {ma} | {ra} | {pa} |", "",
         f"Stage C dropped **{nb-na}** predictions; recall {rb}→{ra}, precision-floor {pb}→{pa}.", "",
         "## What Stage C dropped (✗ = real gold gap lost; rest = false positives removed)"]
    real_lost = 0
    for pid, m, g in dropped:
        flag = f"✗ LOST gold {m}" if m else "✓ FP removed"
        if m:
            real_lost += 1
        R.append(f"- [{flag}] ({g['gap_type'][:4]}/{g['section_type'][:4]}) {short(g['gap_sentence'], 90)}")
    R += ["", f"Dropped {nb-na}: {nb-na-real_lost} false positives removed, {real_lost} real gold gaps lost."]

    out = ROOT / "docs/experiments/stage_c_output.md"
    out.write_text("\n".join(R), encoding="utf-8")
    print(f"\nBEFORE A+B:  preds={nb} recall={rb} prec_floor={pb}")
    print(f"AFTER +C:    preds={na} recall={ra} prec_floor={pa}")
    print(f"dropped {nb-na}: {nb-na-real_lost} FPs removed, {real_lost} real gaps lost")
    print(f"readable report -> {out}")


if __name__ == "__main__":
    main()
