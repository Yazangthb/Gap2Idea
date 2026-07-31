"""Stage C via OpenRouter with BATCHED requests + minimal prompt.

Sends N sentences per API call (default 20). Parses the response as
"1. YES\n2. NO\n3. YES..." to keep tokens cheap. Saves ~20× on request count.

Tests on the 20-sample LimGen subset first, then optionally full LimGen.

    python -u scripts/stage_c/test_openrouter_stage_c.py --model openai/gpt-4o-mini --batch 20
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

SYSTEM = (
    "Each input is a sentence from a scientific paper that a classifier flagged as "
    "possibly being from the Limitations section. For each, output YES if it plausibly "
    "appears in a Limitations section — including limitation acknowledgments, scope "
    "restrictions, future-work plans, assumptions, ACL Responsible-Research checklist "
    "Q&A, ethical concerns, and hedged statements. Output NO ONLY if it is clearly: "
    "gratitude/thanks, a numbered performance result ('95% accuracy'), pure "
    "hyperparameter listing, prior-work citation ('Smith (2024) shows...'), pure "
    "method/dataset description ('we use PaddleOCR'), or truncated fragment. "
    "When uncertain, default to YES. Output ONLY '<idx>. YES' or '<idx>. NO' in order."
)


def batched_judge(client, model, sentences, batch_size, max_retries=2):
    """Returns list of bool (True = keep, False = reject) per sentence."""
    out = [None] * len(sentences)
    for i in range(0, len(sentences), batch_size):
        chunk = sentences[i:i + batch_size]
        user = "\n".join(f"{k+1}. {s}" for k, s in enumerate(chunk))
        for attempt in range(max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=8 * len(chunk),
                )
                content = resp.choices[0].message.content
                # parse: lines like "3. YES" or "3. NO"
                ans = {}
                for ln in content.splitlines():
                    m = re.match(r"\s*(\d+)\s*[.):\-]\s*(YES|NO)", ln.strip(), re.IGNORECASE)
                    if m:
                        ans[int(m.group(1))] = m.group(2).upper() == "YES"
                missing = [k for k in range(1, len(chunk) + 1) if k not in ans]
                if missing and attempt < max_retries:
                    print(f"  retry chunk {i}: missing {len(missing)} answers", flush=True)
                    continue
                for k in range(1, len(chunk) + 1):
                    out[i + k - 1] = ans.get(k, True)  # default keep on parse failure
                break
            except Exception as e:
                print(f"  chunk {i} attempt {attempt + 1}: {e}", flush=True)
                if attempt == max_retries:
                    for k in range(len(chunk)):
                        out[i + k] = True   # default keep on hard failure
                time.sleep(1)
    return out


def load_20_subset():
    losses = json.load(open(ROOT / "data/scibert_prep/stage_c_recall_losses.json"))
    kills = json.load(open(ROOT / "data/scibert_prep/stage_c_correct_kills.json"))
    rng = np.random.default_rng(0)
    tps = [losses[i] for i in rng.permutation(len(losses))[:10]]
    fps = [kills[i] for i in rng.permutation(len(kills))[:10]]
    return [(t["sentence"], "GAP") for t in tps] + [(f["sentence"], "NOT_GAP") for f in fps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()

    samples = load_20_subset()
    print(f"Loaded {len(samples)} samples: {sum(1 for _, e in samples if e == 'GAP')} gaps + "
          f"{sum(1 for _, e in samples if e == 'NOT_GAP')} non-gaps", flush=True)

    client = get_llm_client()
    t0 = time.time()
    sents = [s for s, _ in samples]
    preds = batched_judge(client, args.model, sents, args.batch)
    dt = time.time() - t0
    n_calls = (len(sents) + args.batch - 1) // args.batch
    print(f"\n{n_calls} API calls for {len(sents)} sentences ({args.batch}/call) in {dt:.1f}s", flush=True)

    tp_kept = fp_rejected = 0
    tp_total = fp_total = 0
    wrong = []
    for (s, exp), pred in zip(samples, preds):
        if exp == "GAP":
            tp_total += 1
            if pred:
                tp_kept += 1
            else:
                wrong.append((s, exp))
        else:
            fp_total += 1
            if not pred:
                fp_rejected += 1
            else:
                wrong.append((s, exp))

    acc = (tp_kept + fp_rejected) / len(samples)
    print(f"\n=== {args.model} (batched, minimal prompt) ===", flush=True)
    print(f"  acc={acc:.2f}  TP-kept={tp_kept}/{tp_total}  FP-rejected={fp_rejected}/{fp_total}", flush=True)
    if wrong:
        print(f"  WRONG ({len(wrong)}):")
        for s, e in wrong[:10]:
            print(f"    exp={e} :: {s[:90]}", flush=True)


if __name__ == "__main__":
    main()
