"""LimGen-tuned Stage C: only narrow categories that empirically never overlap
with real positives. Tested honestly with train/test split (no leakage).

Strategy: instead of trying to filter many FP types and risking recall, only
reject the 3-4 categories that ZERO real Limitations-section sentences fall into:

  - GRATITUDE      : explicit thanks/gratitude
  - NUMBERED_RESULT: a performance number ('95% accuracy')
  - BROKEN_LINE    : truncated fragment (mid-clause, broken hyphenation)
  - PURE_CITATION  : citation-only line ('See X', 'Smith et al., 2024.')

Everything else is kept. Tests on the held-out 90-sample TEST split (gpt-4o
seen samples are not used). Reports projected full-LimGen F1.

    python -u scripts/stage_c/test_safe_limgen_prompt.py --model openai/gpt-4o
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

from gap2idea.pipeline.llm import get_llm_client  # noqa: E402

# === Safe LimGen-tuned prompt ===
SYSTEM_SAFE = (
    "Each input is a sentence a classifier flagged as possibly from a paper's "
    "Limitations section. Default to YES (keep). Reply NO ONLY if the sentence "
    "is clearly one of these four narrow categories — NO OTHER REASONS:\n"
    "  1. GRATITUDE: explicit thanks, gratitude, or acknowledgment of people "
    "(e.g. 'We thank...', 'We are grateful to...').\n"
    "  2. NUMBERED_RESULT: states a numeric performance result "
    "(e.g. '95.2% accuracy', 'F1 of 0.83', 'outperforms baselines by 3.1 points').\n"
    "  3. BROKEN_LINE: sentence is clearly truncated or fragmented mid-clause "
    "(starts with lowercase mid-word, ends abandoned at a hyphen).\n"
    "  4. PURE_CITATION: sentence is ONLY a citation/cross-reference with no content "
    "of its own (e.g. 'See Appendix L.3.', 'Smith et al., 2024.', 'Equation (3).').\n"
    "If the sentence might be ANYTHING ELSE — including method descriptions, "
    "encouragements, speculation, paper-intro contributions, ACL checklist questions, "
    "or anything else — reply YES.\n"
    "Output ONLY '<idx>. YES' or '<idx>. NO' per line."
)

SHOTS_SAFE = [
    # Universal accepts (real limitations / scope / future work)
    ("We leave multilingual evaluation for future work.", "YES"),
    ("This work focuses on English-language datasets.", "YES"),
    ("We did not study risks that may arise in other scenarios.", "YES"),
    # Universal rejects (one per safe category)
    ("We thank the anonymous reviewers for their feedback.", "NO"),
    ("Our method achieves 95.2% accuracy, outperforming baselines by 3.1 points.", "NO"),
    ("of the proposed framework with the additional", "NO"),
    ("See Appendix L.3 for further details.", "NO"),
]


def load_split():
    losses = json.load(open(ROOT / "data/scibert_prep/stage_c_recall_losses.json"))
    kills = json.load(open(ROOT / "data/scibert_prep/stage_c_correct_kills.json"))
    keeps = json.load(open(ROOT / "data/scibert_prep/stage_c_wrong_keeps.json"))
    samples = [(r["sentence"], 1) for r in losses] \
            + [(r["sentence"], 0) for r in kills] \
            + [(r["sentence"], 0) for r in keeps]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    split = len(samples) // 2
    return [samples[i] for i in idx[:split]], [samples[i] for i in idx[split:]]


def predict_batched(client, model, system, shots, sentences, batch=20):
    shot_block = "\n".join(f"Example {k+1}: {s}\nLabel {k+1}: {a}"
                            for k, (s, a) in enumerate(shots))
    preds = [None] * len(sentences)
    for i in range(0, len(sentences), batch):
        chunk = sentences[i:i + batch]
        test_block = "\n".join(f"{k+1}. {s}" for k, s in enumerate(chunk))
        user = f"Reference examples:\n{shot_block}\n\nClassify:\n{test_block}"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
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
        except Exception as e:
            print(f"  batch error: {e}", flush=True)
            for k in range(len(chunk)):
                preds[i + k] = True
    return preds


def metrics(samples, preds):
    tp = sum(y for _, y in samples)
    fp = sum(1 - y for _, y in samples)
    tp_kept = sum(1 for (_, y), p in zip(samples, preds) if y == 1 and p)
    fp_kill = sum(1 for (_, y), p in zip(samples, preds) if y == 0 and not p)
    return tp_kept / max(1, tp), fp_kill / max(1, fp), tp_kept, tp, fp_kill, fp


def project_f1(tp_ret, fp_kill):
    """Project to full LimGen (P=0.809, R=0.687, F1=0.743)."""
    sb_tp, sb_fp = 2293, 539
    new_tp = sb_tp * tp_ret
    new_fp = sb_fp * (1 - fp_kill)
    p = new_tp / max(1, new_tp + new_fp)
    r = new_tp / 3338
    return p, r, 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    args = ap.parse_args()

    train, test = load_split()
    print(f"TRAIN: {len(train)}   TEST: {len(test)}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Prompt length: {len(SYSTEM_SAFE)} chars, {len(SHOTS_SAFE)} fixed shots", flush=True)

    client = get_llm_client()
    t0 = time.time()
    test_sents = [s for s, _ in test]
    preds = predict_batched(client, args.model, SYSTEM_SAFE, SHOTS_SAFE, test_sents)
    dt = time.time() - t0

    tp_ret, fp_kill, tp_k, tp, fp_k, fp = metrics(test, preds)
    p, r, f1 = project_f1(tp_ret, fp_kill)

    print(f"\n=== SAFE LimGen prompt on held-out TEST ({len(test)} samples, {dt:.1f}s) ===")
    print(f"  TP retention: {tp_k}/{tp} = {tp_ret:.3f}")
    print(f"  FP kill rate: {fp_k}/{fp} = {fp_kill:.3f}")
    print(f"\n  Projected full-LimGen F1: {f1:.3f}  (was 0.743)  Δ = {f1 - 0.743:+.3f}")
    print(f"  Projected P: {p:.3f} (was 0.809)")
    print(f"  Projected R: {r:.3f} (was 0.687)")

    # What did it kill on test?
    print(f"\n  Stage C killed on TEST ({fp + tp - tp_k - fp_k} sentences total):")
    correct_kills = [(s, y) for (s, y), p in zip(test, preds) if not p and y == 0]
    wrong_kills = [(s, y) for (s, y), p in zip(test, preds) if not p and y == 1]
    for s, _ in correct_kills[:6]:
        print(f"    [GOOD kill] {s[:90]}")
    for s, _ in wrong_kills[:6]:
        print(f"    [BAD kill — was a real limitation] {s[:90]}")


if __name__ == "__main__":
    main()
