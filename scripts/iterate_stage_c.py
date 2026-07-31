"""Fast prompt iteration on a 20-sample LimGen subset (10 TPs LLM mis-rejected
+ 10 FPs LLM correctly killed). Each variant: ~1 min instead of ~25 min.

Goal: find a prompt that keeps the 10 TPs AND rejects the 10 FPs.
Inputs are the diagnostic JSONs from analyze_stage_c_misses.py.

    python -u scripts/iterate_stage_c.py --variants v5,v6,v7,v8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_samples():
    """20 LimGen samples: 10 real-positives V4 wrongly rejected + 10 real-negatives V4 correctly killed."""
    losses = json.load(open(ROOT / "data/scibert_prep/stage_c_recall_losses.json"))
    kills = json.load(open(ROOT / "data/scibert_prep/stage_c_correct_kills.json"))
    rng = np.random.default_rng(0)
    tps = [losses[i] for i in rng.permutation(len(losses))[:10]]  # gold YES, LLM said NO -> we want YES
    fps = [kills[i] for i in rng.permutation(len(kills))[:10]]    # gold NO,  LLM said NO -> we want NO
    return [(t["sentence"], "GAP") for t in tps] + [(f["sentence"], "NOT_GAP") for f in fps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--variants", default="validate,validate_cot,validate_v5,validate_v6")
    args = ap.parse_args()

    samples = load_samples()
    print(f"Loaded {len(samples)} samples: {sum(1 for _, e in samples if e == 'GAP')} gaps + "
          f"{sum(1 for _, e in samples if e == 'NOT_GAP')} non-gaps", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {args.model} on GPU...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    lm = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="auto")
    lm.eval()

    from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
    for mode in args.variants.split(","):
        mode = mode.strip()
        filt = LLMGapFilter(backend="local", mode=mode, model=args.model)
        filt._tok = tok; filt._lm = lm; filt._device = next(lm.parameters()).device

        tp_kept = fp_rejected = 0
        tp_total = fp_total = 0
        wrong_predictions = []
        for sent, expected in samples:
            kept = filt.judge(sent)
            pred = "GAP" if kept else "NOT_GAP"
            if expected == "GAP":
                tp_total += 1
                tp_kept += int(kept)
                if not kept:
                    wrong_predictions.append((sent, expected, pred))
            else:
                fp_total += 1
                fp_rejected += int(not kept)
                if kept:
                    wrong_predictions.append((sent, expected, pred))

        acc = (tp_kept + fp_rejected) / len(samples)
        print(f"\n=== {mode} ===  acc={acc:.2f}  TP-kept={tp_kept}/{tp_total}  "
              f"FP-rejected={fp_rejected}/{fp_total}", flush=True)
        if wrong_predictions:
            print(f"  WRONG ({len(wrong_predictions)}):")
            for s, e, p in wrong_predictions[:6]:
                print(f"    exp={e} got={p} :: {s[:90]}", flush=True)


if __name__ == "__main__":
    main()
