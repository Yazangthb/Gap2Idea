# -*- coding: utf-8 -*-
"""Run Stage C (LLM precision filter) on the saved funnel-v2 candidates and compute
end-to-end A->B->C precision/recall. Reads data/gold/funnel_v2_candidates.json."""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter

CAND = ROOT / "data" / "gold" / "funnel_v2_candidates.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yandexgpt-5-lite")
    ap.add_argument("--mode", default="validate")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--file", default=str(CAND), help="candidates json (recall-first or balanced)")
    args = ap.parse_args()
    cands = json.loads(Path(args.file).read_text(encoding="utf-8"))
    allpos = sum(1 for c in cands if c["label"] != "none")
    pool = cands[:args.limit] if args.limit else cands
    print(f"candidates: {len(cands)} (gold gaps among them: {allpos}); judging {len(pool)} "
          f"with {args.model}/{args.mode}", flush=True)
    filt = LLMGapFilter(backend="api", model=args.model, mode=args.mode)
    rows_in = [{"id": c["id"], "gap_type": c["gap_type"], "gap_sentence": c["gap_sentence"],
                "paragraph_text": c["paragraph_text"]} for c in pool]
    out = filt.filter_gaps(rows_in, protect_rules=False)
    keptset = {(d["id"], d["gap_sentence"]) for d in out}
    tp = sum(1 for c in pool if (c["id"], c["gap_sentence"]) in keptset and c["label"] != "none")
    fp = sum(1 for c in pool if (c["id"], c["gap_sentence"]) in keptset and c["label"] == "none")
    before_pos = sum(1 for c in pool if c["label"] != "none")   # gold gaps in the judged pool
    before_P = before_pos / len(pool) if pool else 0.0
    P = tp / (tp + fp) if tp + fp else 0.0                       # precision after Stage C
    R_pool = tp / before_pos if before_pos else 0.0             # gaps retained within the judged pool
    R = tp / allpos if allpos else 0.0                          # global recall (only valid if limit=0)
    F = 2 * P * R_pool / (P + R_pool) if P + R_pool else 0.0
    subsample = bool(args.limit)
    print(f"\nStage C ({args.model}): judged {filt.n_judged} in {filt.n_calls} calls; "
          f"kept {len(out)}/{len(pool)} | gold gaps in pool: {before_pos}")
    print(f"BEFORE (detector): precision={before_P*100:.1f}%  (pool size {len(pool)})")
    print(f"AFTER  (+Stage C): precision={P*100:.1f}%  gaps-retained={R_pool*100:.1f}% (tp={tp} fp={fp})")
    if not subsample:
        print(f"   global recall vs all file gaps: {R*100:.1f}%  F1={F*100:.1f}%")
    else:
        print(f"   (subsample of {len(pool)} — precision + within-pool gap retention are the valid metrics)")
    scope = f"subsample of {len(pool)}" if subsample else f"all {len(pool)}"
    (ROOT / "docs" / "experiments" / "funnel_v2_stagec.md").write_text(
        f"# Funnel v2 — Stage C ({args.model}, mode={args.mode})\n\n"
        f"Judged {scope} candidates from `{Path(args.file).name}` in {filt.n_calls} calls; "
        f"kept {len(out)}/{len(pool)}. Gold gaps in pool: {before_pos}.\n\n"
        f"| | detector (before) | + Stage C |\n|---|---|---|\n"
        f"| precision | {before_P*100:.1f}% | **{P*100:.1f}%** |\n"
        f"| gaps retained (within pool) | 100% | {R_pool*100:.1f}% |\n\n"
        f"Stage C lifts precision {before_P*100:.1f}% -> {P*100:.1f}% while retaining {R_pool*100:.1f}% "
        f"of the true gaps in the judged pool"
        + ("" if subsample else f"; global recall {R*100:.1f}%, F1 {F*100:.1f}%")
        + ". Labels are LLM-gold.\n", encoding="utf-8")
    print("wrote docs/experiments/funnel_v2_stagec.md")


if __name__ == "__main__":
    main()
