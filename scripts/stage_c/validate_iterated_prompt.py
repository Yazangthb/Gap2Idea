"""Validate the iterated gpt-4o prompt on the FULL 179-sentence diagnostic set
(held out from the iteration loop). Reports real Stage C generalization on
SciBERT-positives.

The iteration ran on 20 of these (10 TPs + 10 FPs). The remaining 159 sentences
are held out — they tell us if the iterated prompt actually generalizes or just
memorized the iteration set.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap2idea.pipeline.llm import get_llm_client  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-file", default="data/scibert_prep/best_openrouter_prompt.json")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    cfg = json.loads((ROOT / args.prompt_file).read_text(encoding="utf-8"))
    print(f"Model: {cfg['model']}", flush=True)
    print(f"System: {cfg['system'][:160]}...", flush=True)
    print(f"Shots: {len(cfg['shots'])}", flush=True)

    # Build dataset: all SciBERT positives we have, with gold labels.
    # gold=YES (limitation): recall_losses
    # gold=NO  (non-limitation): correct_kills + wrong_keeps
    losses = json.load(open(ROOT / "data/scibert_prep/stage_c_recall_losses.json"))
    kills = json.load(open(ROOT / "data/scibert_prep/stage_c_correct_kills.json"))
    keeps = json.load(open(ROOT / "data/scibert_prep/stage_c_wrong_keeps.json"))

    # Held-out: skip the first 10 of losses (TPs used in iteration) and first 10 of kills (FPs used)
    rng = np.random.default_rng(0)
    iter_tp_idx = set(rng.permutation(len(losses))[:10])
    iter_fp_idx = set(rng.permutation(len(kills))[:10])

    samples = []
    for i, r in enumerate(losses):
        if i not in iter_tp_idx:
            samples.append((r["sentence"], 1))    # gold = limitation
    for i, r in enumerate(kills):
        if i not in iter_fp_idx:
            samples.append((r["sentence"], 0))    # gold = non-limitation
    for r in keeps:
        samples.append((r["sentence"], 0))         # gold = non-limitation (all of these are held out)
    print(f"Held-out samples: {len(samples)} "
          f"({sum(y for _, y in samples)} gold-positives + "
          f"{sum(1-y for _, y in samples)} gold-negatives)", flush=True)

    # Build prompt shots
    shot_block = "\n".join(f"Example {k+1}: {s}\nLabel {k+1}: {a}"
                            for k, (s, a) in enumerate(cfg["shots"]))

    client = get_llm_client()
    preds = [None] * len(samples)
    sentences = [s for s, _ in samples]
    t0 = time.time()
    for i in range(0, len(sentences), args.batch):
        chunk = sentences[i:i + args.batch]
        test_block = "\n".join(f"{k+1}. {s}" for k, s in enumerate(chunk))
        user = f"Here are reference examples:\n{shot_block}\n\nNow classify these sentences:\n{test_block}"
        try:
            resp = client.chat.completions.create(
                model=cfg["model"],
                messages=[{"role": "system", "content": cfg["system"]},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=8 * len(chunk))
            content = resp.choices[0].message.content
            ans = {}
            for ln in content.splitlines():
                m = re.match(r"\s*(\d+)\s*[.):\-]\s*(YES|NO)", ln.strip(), re.IGNORECASE)
                if m:
                    ans[int(m.group(1))] = m.group(2).upper() == "YES"
            for k in range(len(chunk)):
                preds[i + k] = ans.get(k + 1, True)
            if (i // args.batch) % 3 == 0:
                print(f"  {i+len(chunk)}/{len(sentences)}", flush=True)
        except Exception as e:
            print(f"  chunk {i}: error {e}", flush=True)
            for k in range(len(chunk)):
                preds[i + k] = True
    dt = time.time() - t0
    n_calls = (len(sentences) + args.batch - 1) // args.batch
    print(f"\nDone: {n_calls} API calls, {len(sentences)} sentences in {dt:.1f}s", flush=True)

    # Stage C metrics on this held-out set
    y_true = np.array([y for _, y in samples])
    y_pred = np.array([1 if p else 0 for p in preds])

    # Confusion: SciBERT picked all of these as positive. Stage C decides keep/reject.
    sb_tp = int(y_true.sum())                            # = 57 (gold = limitation)
    sb_fp = int(len(y_true) - sb_tp)                     # = 102 (gold = non-limitation)
    kept_gold_pos = int(((y_pred == 1) & (y_true == 1)).sum())  # TPs kept
    kept_gold_neg = int(((y_pred == 1) & (y_true == 0)).sum())  # FPs not caught
    dropped_gold_pos = sb_tp - kept_gold_pos              # TPs lost (BAD)
    dropped_gold_neg = sb_fp - kept_gold_neg              # FPs killed (GOOD)

    print(f"\n=== Stage C on held-out diagnostic set ({cfg['model']}) ===")
    print(f"  SciBERT-positives in set: {len(samples)} "
          f"({sb_tp} real limits, {sb_fp} FPs)")
    print(f"  Stage C kept {(y_pred==1).sum()} sentences")
    print(f"    -- kept {kept_gold_pos}/{sb_tp} real limits "
          f"({kept_gold_pos/sb_tp:.2f} TP retention)")
    print(f"    -- killed {dropped_gold_neg}/{sb_fp} FPs "
          f"({dropped_gold_neg/sb_fp:.2f} FP kill rate)")
    print(f"  Equivalent: dropped {dropped_gold_pos} real limits + killed {dropped_gold_neg} FPs")

    # If we extrapolate the TP-retention / FP-kill ratios to full LimGen
    tp_ret = kept_gold_pos / max(1, sb_tp)
    fp_kill = dropped_gold_neg / max(1, sb_fp)
    # SciBERT alone: P=0.809 R=0.687 F1=0.743 on 13319 sents, 3338 limits
    R = 3338 * 0.687
    new_tp = R * tp_ret
    new_fp = (2832 - R) * (1 - fp_kill)
    new_p = new_tp / max(1, (new_tp + new_fp))
    new_r = new_tp / 3338
    new_f1 = 2 * new_p * new_r / max(1e-9, (new_p + new_r))
    print(f"\n  Extrapolated to full LimGen test:")
    print(f"    P = {new_p:.3f} (was 0.809)")
    print(f"    R = {new_r:.3f} (was 0.687)")
    print(f"    F1 = {new_f1:.3f} (was 0.743)  ΔF1 = {new_f1 - 0.743:+.3f}")


if __name__ == "__main__":
    main()
