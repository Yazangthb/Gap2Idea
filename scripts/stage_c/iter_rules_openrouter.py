"""Categorical-rule iteration (no sentence memorization).

The previous iterator added wrong test sentences VERBATIM as shots — that's just
memorization. This one:
  1. Splits the 179-sentence diagnostic set 50/50 into TRAIN / TEST.
  2. Starts with a minimal system prompt + 5 fixed SYNTHETIC shots.
  3. Each round: looks at wrong TRAIN predictions, asks an LLM to classify each
     into an abstract CATEGORY (e.g. "encouragement", "method-recipe"), and adds
     the CATEGORY (with a synthesized description) to the system message. NEVER
     adds the failing sentence verbatim.
  4. Reports accuracy on the HELD-OUT TEST split — the honest generalization.

    python -u scripts/stage_c/iter_rules_openrouter.py --model openai/gpt-4o --rounds 5
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

# Minimal starting system, with empty category list.
SYSTEM_HEAD = (
    "Each input is a sentence a classifier flagged as possibly being from a paper's "
    "Limitations section. Reply YES if it plausibly belongs in a Limitations section. "
    "Reply NO ONLY if it clearly falls into one of these categories:"
)
SYSTEM_TAIL = (
    "\nWhen uncertain, reply YES (default-accept). "
    "Output ONLY '<idx>. YES' or '<idx>. NO' per line."
)

# Initial categories (the obvious universal ones, abstract descriptions only).
INITIAL_CATEGORIES = [
    ("GRATITUDE", "explicit thanks or acknowledgments to people"),
    ("NUMBERED_RESULT", "a performance result with specific numbers, e.g. '95% accuracy'"),
    ("CITATION_ONLY", "a pure citation reference with no content of its own"),
]

# 4 fixed synthetic shots — NEVER changed by the iteration.
FIXED_SHOTS = [
    ("We leave multilingual evaluation for future work.", "YES"),
    ("Our method assumes the availability of a knowledge base.", "YES"),
    ("We thank the anonymous reviewers.", "NO"),
    ("Our method achieves 95.2% accuracy, outperforming baselines by 3 points.", "NO"),
]


def build_system(categories):
    rules = "\n".join(f"  - {name}: {desc}" for name, desc in categories)
    return SYSTEM_HEAD + "\n" + rules + SYSTEM_TAIL


def load_split():
    """Load 179 SciBERT positives, split 50/50 train/test by seed."""
    losses = json.load(open(ROOT / "data/scibert_prep/stage_c_recall_losses.json"))
    kills = json.load(open(ROOT / "data/scibert_prep/stage_c_correct_kills.json"))
    keeps = json.load(open(ROOT / "data/scibert_prep/stage_c_wrong_keeps.json"))
    samples = [(r["sentence"], 1) for r in losses] \
            + [(r["sentence"], 0) for r in kills] \
            + [(r["sentence"], 0) for r in keeps]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    split = len(samples) // 2
    train = [samples[i] for i in idx[:split]]
    test = [samples[i] for i in idx[split:]]
    return train, test


def predict_batched(client, model, system, shots, samples, batch=20):
    sentences = [s for s, _ in samples]
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


def classify_pattern(client, model, sentence, categories):
    """Ask the LLM: which existing category does this sentence belong to, or
    what NEW category should be added? Returns (name, description)."""
    cat_list = "\n".join(f"  - {n}: {d}" for n, d in categories)
    msg = (
        f"The following sentence should be REJECTED (not a real research gap):\n"
        f"  \"{sentence[:200]}\"\n\n"
        f"Existing reject categories:\n{cat_list}\n\n"
        f"Does it fit one of the existing categories? If yes, reply: "
        f"EXISTING: <CATEGORY_NAME>\n"
        f"If it needs a NEW abstract category (not a phrase, a general type), reply:\n"
        f"NEW: <CATEGORY_NAME_UPPERCASE>: <one-sentence general description of what sentences in this category look like>\n"
        f"Output only one line."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": msg}],
            temperature=0.0, max_tokens=80)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def evaluate(samples, preds):
    tp_kept = fp_kill = tp = fp = 0
    for (_, y), p in zip(samples, preds):
        if y == 1:
            tp += 1
            if p: tp_kept += 1
        else:
            fp += 1
            if not p: fp_kill += 1
    acc = (tp_kept + fp_kill) / len(samples)
    return acc, tp_kept, tp, fp_kill, fp


def extrapolate_f1(tp_ret, fp_kill):
    """Roughly project to full LimGen using SciBERT baseline."""
    R0, P0 = 0.687, 0.809
    total_gold = 3338
    sb_pos = 2832
    sb_tp = total_gold * R0
    sb_fp = sb_pos - sb_tp
    new_tp = sb_tp * tp_ret
    new_fp = sb_fp * (1 - fp_kill)
    new_p = new_tp / max(1, (new_tp + new_fp))
    new_r = new_tp / total_gold
    new_f1 = 2 * new_p * new_r / max(1e-9, (new_p + new_r))
    return new_p, new_r, new_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--max-categories", type=int, default=10)
    args = ap.parse_args()

    train, test = load_split()
    print(f"TRAIN: {len(train)} ({sum(y for _,y in train)} TP / {sum(1-y for _,y in train)} FP)", flush=True)
    print(f"TEST : {len(test)} ({sum(y for _,y in test)} TP / {sum(1-y for _,y in test)} FP)", flush=True)

    client = get_llm_client()
    categories = list(INITIAL_CATEGORIES)

    for rd in range(args.rounds):
        system = build_system(categories)
        print(f"\n=== Round {rd+1} ===  categories={len(categories)}  "
              f"system_len={len(system)} chars", flush=True)

        # evaluate on both train and test
        train_pred = predict_batched(client, args.model, system, FIXED_SHOTS, train)
        test_pred = predict_batched(client, args.model, system, FIXED_SHOTS, test)
        tr_acc, tr_tp, tr_tp_t, tr_fp, tr_fp_t = evaluate(train, train_pred)
        te_acc, te_tp, te_tp_t, te_fp, te_fp_t = evaluate(test, test_pred)
        te_tp_ret = te_tp / max(1, te_tp_t)
        te_fp_kill = te_fp / max(1, te_fp_t)
        p, r, f1 = extrapolate_f1(te_tp_ret, te_fp_kill)
        print(f"  TRAIN acc={tr_acc:.3f}  TPret={tr_tp}/{tr_tp_t}  FPkill={tr_fp}/{tr_fp_t}", flush=True)
        print(f"  TEST  acc={te_acc:.3f}  TPret={te_tp}/{te_tp_t} ({te_tp_ret:.2f})  "
              f"FPkill={te_fp}/{te_fp_t} ({te_fp_kill:.2f})", flush=True)
        print(f"  EXTRAPOLATED LimGen F1: {f1:.3f} (was 0.743) Δ={f1-0.743:+.3f}", flush=True)

        if rd == args.rounds - 1 or len(categories) >= args.max_categories:
            break

        # Find one wrong FP-kept on training set (a sentence we should kill but didn't)
        wrong_keeps = [s for (s, y), p in zip(train, train_pred) if y == 0 and p]
        if not wrong_keeps:
            print("  no wrong FPs left on training — stopping", flush=True)
            break
        # Sample up to 3 wrong-keeps to discover new categories
        rng = np.random.default_rng(rd + 1)
        sample_wrong = [wrong_keeps[i] for i in rng.permutation(len(wrong_keeps))[:3]]
        new_cats_added = 0
        for sent in sample_wrong:
            answer = classify_pattern(client, args.model, sent, categories)
            print(f"\n  Classifier on \"{sent[:60]}...\":\n    {answer}", flush=True)
            if answer.upper().startswith("NEW:"):
                parts = answer[4:].strip().split(":", 1)
                if len(parts) == 2:
                    name, desc = parts[0].strip().upper(), parts[1].strip()
                    # avoid duplicates
                    if name and not any(n == name for n, _ in categories):
                        categories.append((name, desc))
                        new_cats_added += 1
                        print(f"    -> added category {name}", flush=True)
                        if len(categories) >= args.max_categories:
                            break
        if new_cats_added == 0:
            print("  no new categories discovered — stopping", flush=True)
            break

    print(f"\n=== FINAL ===")
    print(f"Categories ({len(categories)}):")
    for n, d in categories:
        print(f"  {n}: {d}")


if __name__ == "__main__":
    main()
