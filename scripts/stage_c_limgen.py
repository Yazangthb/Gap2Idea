"""Stage C on the LimGen yardstick — before vs after (LLM part, isolated process).

Run `_stageb_limgen_worker.py` FIRST (saves Stage-B predictions). This script then
loads them and applies the Stage C LLM filter (rule-protected) — it does NOT load
sentence-transformers, so the LLM can't deadlock against it. Same task/metric as
the prior-art table, so it's directly comparable to bge+logreg 0.61 etc.

    python -u scripts/_stageb_limgen_worker.py --test-papers 15
    python -u scripts/stage_c_limgen.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gap2idea.pipeline.gap_llm_filter import LLMGapFilter  # noqa: E402 (transformers only)
from bench_limgen import prf  # noqa: E402

SC = ROOT / "data" / "limgen" / "_sc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--mode", default="validate")
    args = ap.parse_args()

    te_s = json.loads((SC / "te_s.json").read_text(encoding="utf-8"))
    te_y = np.load(SC / "te_y.npy")
    bge_pos = np.load(SC / "bge_pos.npy").astype(bool)
    cue_pos = np.load(SC / "cue_pos.npy").astype(bool)
    stage_b = bge_pos | cue_pos
    print(f"test={len(te_s)} ({int(te_y.sum())} limitation)  stageB positives={int(stage_b.sum())}", flush=True)

    filt = LLMGapFilter(backend="local", model=args.model, mode=args.mode)
    stage_c = stage_b.copy()
    n = 0
    for i, s in enumerate(te_s):
        if stage_b[i] and not cue_pos[i]:        # model-only positive -> LLM judges
            n += 1
            if not filt.judge(s):
                stage_c[i] = False
    print(f"LLM judged {n} model-only positives", flush=True)

    Pb, Rb, Fb = prf(te_y, stage_b.astype(int))
    Pc, Rc, Fc = prf(te_y, stage_c.astype(int))
    print(f"\n=== Stage C on LimGen (same metric as research table) ===", flush=True)
    print(f"  Stage B (before)   P={Pb} R={Rb} F1={Fb}", flush=True)
    print(f"  Stage B + C (after) P={Pc} R={Rc} F1={Fc}", flush=True)
    print(f"  ΔF1 = {Fc-Fb:+.3f}", flush=True)

    out = ROOT / "docs/experiments/stage_c_limgen.md"
    out.write_text(
        "# Stage C on the LimGen yardstick (small sample)\n\n"
        f"LimGen limitation detection, held-out test sample ({len(te_s)} sents, "
        f"{int(te_y.sum())} limitation). Same task/metric as the prior-art table. "
        f"Stage C = LLM filter ({args.model}), rule-protected.\n\n"
        "| method | precision | recall | F1 |\n|---|---|---|---|\n"
        f"| Stage B (bge+logreg+cue) | {Pb} | {Rb} | {Fb} |\n"
        f"| **Stage B + Stage C** | {Pc} | {Rc} | **{Fc}** |\n\n"
        f"ΔF1 {Fc-Fb:+.3f}; precision {Pb}→{Pc}, recall {Rb}→{Rc}. {n} LLM judgments.\n",
        encoding="utf-8")
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
