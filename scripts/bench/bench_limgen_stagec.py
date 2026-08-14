"""LimGen Stage-C lift at scale (leakage-clean).

Reuses bench_limgen's LimGen loaders. Trains Stage B (bge-small + logreg) fresh on
LimGen-TRAIN, predicts on held-out LimGen-TEST, then applies the batched Stage-C
LLM precision filter over Stage-B's positives. Reports limitation-class P/R/F1
BEFORE and AFTER Stage C, and the split of what Stage C drops (false positives
removed vs true limitations lost).

The shipped head is NOT used (it saw LimGen test+valid). Stage C is a zero-shot
LLM judge -> no leakage.

    python scripts/bench/bench_limgen_stagec.py --test-papers 60 --train-papers 300   # sample
    python scripts/bench/bench_limgen_stagec.py                                         # full
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_limgen import fetch, build_xy, prf, m_bge  # noqa: E402
from gap2idea.pipeline.gap_funnel import cue_label  # noqa: E402
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter  # noqa: E402
from gap2idea.pipeline.llm import active_provider  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-papers", type=int, default=1500)
    ap.add_argument("--cap-pos", type=int, default=1800)
    ap.add_argument("--test-papers", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--protect-cues", action=argparse.BooleanOptionalAction, default=True,
                    help="keep cue-hit limitation positives unjudged (shipped protect_rules behavior)")
    ap.add_argument("--show-drops", action="store_true", help="print sample dropped sentences")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr_path, te_path = fetch("train.jsonl"), fetch("test.jsonl")
    tr_s, tr_y = build_xy(tr_path, args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(te_path, args.test_papers, None, args.seed + 1)
    print(f"provider={active_provider()}  TRAIN {len(tr_s)} sents ({int(tr_y.sum())} lim)  "
          f"TEST {len(te_s)} sents ({int(te_y.sum())} lim)", flush=True)

    # Stage B (fresh, leakage-clean)
    sb = m_bge(tr_s, tr_y, te_s)
    Pb, Rb, Fb = prf(te_y, sb)
    print(f"Stage B  (bge+logreg):  P={Pb} R={Rb} F1={Fb}", flush=True)

    # Stage C over Stage-B positives (batched). With --protect-cues, cue-hit
    # limitation positives are kept unjudged (the shipped protect_rules behavior);
    # only the cue-less positives are sent to the LLM.
    filt = LLMGapFilter(backend="api", mode="validate")
    all_pos = [i for i, v in enumerate(sb) if v == 1]
    if args.protect_cues:
        judge_idx = [i for i in all_pos if cue_label(te_s[i]) != "limitation"]
        n_prot = len(all_pos) - len(judge_idx)
    else:
        judge_idx = all_pos; n_prot = 0
    judge_sents = [te_s[i] for i in judge_idx]
    keep: list = []
    for k in range(0, len(judge_sents), args.chunk):
        keep.extend(filt.judge_batch(judge_sents[k:k + args.chunk]))
    final = sb.copy()
    dropped = fp_removed = lim_lost = 0
    for i, kp in zip(judge_idx, keep):
        if not kp:
            final[i] = 0; dropped += 1
            if te_y[i] == 1: lim_lost += 1
            else: fp_removed += 1
    pos_sents = judge_sents  # for the count print below
    print(f"protect-cues={args.protect_cues}: {n_prot} cue-hit positives protected, "
          f"{len(judge_idx)} judged", flush=True)
    if args.show_drops:
        lost = [te_s[i] for i, kp in zip(judge_idx, keep) if not kp and te_y[i] == 1][:12]
        fps = [te_s[i] for i, kp in zip(judge_idx, keep) if not kp and te_y[i] == 0][:8]
        print("\n-- dropped TRUE limitations (LimGen-labelled; recall loss) --")
        for s in lost: print("   x", s.strip()[:105])
        print("-- dropped FALSE positives (precision gain) --")
        for s in fps: print("   .", s.strip()[:105])
        print()
    Pc, Rc, Fc = prf(te_y, final)
    print(f"Stage C  judged {len(pos_sents)} positives in {filt.n_calls} calls; "
          f"dropped {dropped} ({fp_removed} FPs removed, {lim_lost} true limitations lost)", flush=True)
    print(f"Stage B + Stage C:      P={Pc} R={Rc} F1={Fc}")
    print(f"\nStage C lift: F1 {Fb} -> {Fc}   (P {Pb}->{Pc}, R {Rb}->{Rc})")

    out = ROOT / "docs/experiments/limgen_stagec.md"
    out.write_text(
        "# LimGen Stage-C lift (limitation detection, held-out test, leakage-clean)\n\n"
        f"Stage B = bge-small+logreg trained fresh on LimGen-TRAIN ({len(tr_s)} sents); "
        f"test {len(te_s)} sents ({int(te_y.sum())} limitations). Stage C = batched LLM "
        f"precision filter ({active_provider()}) over Stage-B positives.\n\n"
        "| stage | precision | recall | F1 |\n|---|---|---|---|\n"
        f"| Stage B | {Pb} | {Rb} | {Fb} |\n"
        f"| + Stage C | {Pc} | {Rc} | {Fc} |\n\n"
        f"Stage C judged {len(pos_sents)} positives in {filt.n_calls} calls; dropped {dropped} "
        f"({fp_removed} false positives removed, {lim_lost} true limitations lost).\n",
        encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
