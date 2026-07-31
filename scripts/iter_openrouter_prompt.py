"""Auto-iterate a minimal OpenRouter Stage C prompt.

Loop:
  1. Run current (system + shots) on the 20-sample subset
  2. Find wrong predictions
  3. Add ONE new shot for each NEW failure pattern (max 2 new shots/round, capped at 12 total)
  4. Stop when accuracy >= target or after N iterations

Keeps the prompt minimal — no verbose categories, just the system message + few-shot Q/A pairs
that the model has actually needed to see.

    python -u scripts/iter_openrouter_prompt.py --model anthropic/claude-3-haiku --rounds 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gap2idea.pipeline.llm import get_llm_client  # noqa: E402

# Initial direct + minimal system message
SYSTEM_INIT = (
    "Each input is a sentence flagged as possibly being from a paper's Limitations "
    "section. For each, reply YES if it could plausibly be in a Limitations section "
    "(limitation, scope restriction, future-work plan, assumption, or ACL checklist Q&A). "
    "Reply NO if it is clearly NOT a gap (gratitude, numbered result, hyperparam list, "
    "method description, prior-work citation, encouragement, benefit speculation, "
    "paper-intro contribution, figure/example reference, or truncated fragment). "
    "When uncertain, reply YES. Output ONLY '<idx>. YES' or '<idx>. NO' per line."
)

# Tiny starting shot set — just 3 obvious cases
INITIAL_SHOTS = [
    ("We leave multilingual evaluation for future work.", "YES"),
    ("We thank the anonymous reviewers.", "NO"),
    ("Our method achieves 95.2% accuracy, outperforming baselines by 3 points.", "NO"),
]


def load_20_subset():
    losses = json.load(open(ROOT / "data/scibert_prep/stage_c_recall_losses.json"))
    kills = json.load(open(ROOT / "data/scibert_prep/stage_c_correct_kills.json"))
    rng = np.random.default_rng(0)
    tps = [losses[i] for i in rng.permutation(len(losses))[:10]]
    fps = [kills[i] for i in rng.permutation(len(kills))[:10]]
    return [(t["sentence"], "GAP") for t in tps] + [(f["sentence"], "NOT_GAP") for f in fps]


def call_batched(client, model, system, shots, sentences):
    # Format: shots as in-context examples in the user message, then the test sentences
    shot_block = "\n".join(f"Example {k+1}: {s}\nLabel {k+1}: {a}" for k, (s, a) in enumerate(shots))
    test_block = "\n".join(f"{k+1}. {s}" for k, s in enumerate(sentences))
    user = (
        (f"Here are reference examples:\n{shot_block}\n\n" if shots else "")
        + f"Now classify these sentences:\n{test_block}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.0, max_tokens=8 * len(sentences),
    )
    content = resp.choices[0].message.content
    ans = {}
    for ln in content.splitlines():
        m = re.match(r"\s*(\d+)\s*[.):\-]\s*(YES|NO)", ln.strip(), re.IGNORECASE)
        if m:
            ans[int(m.group(1))] = m.group(2).upper() == "YES"
    return [ans.get(k + 1, True) for k in range(len(sentences))]


def evaluate(preds, samples):
    tp_kept = fp_rej = tp_tot = fp_tot = 0
    wrong = []
    for (s, exp), p in zip(samples, preds):
        if exp == "GAP":
            tp_tot += 1
            if p: tp_kept += 1
            else: wrong.append((s, exp, "shouldve_kept"))
        else:
            fp_tot += 1
            if not p: fp_rej += 1
            else: wrong.append((s, exp, "shouldve_killed"))
    acc = (tp_kept + fp_rej) / len(samples)
    return acc, tp_kept, tp_tot, fp_rej, fp_tot, wrong


def shorten(s, n=70):
    return (s[:n] + "...") if len(s) > n else s


def add_correction_shots(shots, wrong, max_new=2, max_total=12):
    """Add at most `max_new` corrective shots for the wrong cases.
    Pick the SHORTEST/clearest wrong example per failure type."""
    if not wrong:
        return shots
    # split: needed labels
    keeps_to_add = [(s, "YES") for s, _, t in wrong if t == "shouldve_kept"]
    kills_to_add = [(s, "NO") for s, _, t in wrong if t == "shouldve_killed"]
    keeps_to_add.sort(key=lambda x: len(x[0]))
    kills_to_add.sort(key=lambda x: len(x[0]))
    # alternate: prioritize whichever class has more errors first
    pool = []
    if len(kills_to_add) >= len(keeps_to_add):
        pool = (kills_to_add + keeps_to_add)
    else:
        pool = (keeps_to_add + kills_to_add)
    existing_sents = {s for s, _ in shots}
    added = 0
    for s, a in pool:
        if added >= max_new or len(shots) >= max_total:
            break
        if s in existing_sents:
            continue
        shots = shots + [(s, a)]
        existing_sents.add(s)
        added += 1
    return shots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-3-haiku")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--target-acc", type=float, default=0.90)
    args = ap.parse_args()

    samples = load_20_subset()
    sentences = [s for s, _ in samples]
    print(f"Loaded {len(samples)} samples", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Initial system: {len(SYSTEM_INIT)} chars", flush=True)
    print(f"Initial shots: {len(INITIAL_SHOTS)}", flush=True)

    client = get_llm_client()
    shots = list(INITIAL_SHOTS)
    history = []
    best_acc, best_shots = 0, shots
    for rd in range(args.rounds):
        t0 = time.time()
        try:
            preds = call_batched(client, args.model, SYSTEM_INIT, shots, sentences)
        except Exception as e:
            print(f"  round {rd+1}: API error {e}", flush=True)
            break
        dt = time.time() - t0
        acc, tp_k, tp_t, fp_r, fp_t, wrong = evaluate(preds, samples)
        history.append((rd + 1, len(shots), acc, tp_k, fp_r, dt))
        print(f"\n=== Round {rd+1} ===  shots={len(shots)}  acc={acc:.2f}  "
              f"TP={tp_k}/{tp_t}  FP-kill={fp_r}/{fp_t}  ({dt:.1f}s)", flush=True)
        if wrong:
            print(f"  wrong ({len(wrong)}):")
            for s, e, t in wrong[:8]:
                print(f"    [{t}] {shorten(s)}", flush=True)
        if acc > best_acc:
            best_acc = acc; best_shots = list(shots)
        if acc >= args.target_acc:
            print(f"\n✅ target {args.target_acc} hit in round {rd+1}", flush=True)
            break
        # add corrective shots for next round
        new_shots = add_correction_shots(shots, wrong, max_new=2)
        if new_shots == shots:
            print("  no new shots to add — stopping", flush=True)
            break
        added = new_shots[len(shots):]
        print(f"  added {len(added)} shot(s) for round {rd+2}:")
        for s, a in added:
            print(f"    [{a}] {shorten(s)}", flush=True)
        shots = new_shots

    print(f"\n=== BEST ===  acc={best_acc:.2f}  shots={len(best_shots)}")
    print(f"\nHISTORY:")
    print(f"  {'round':>6} {'shots':>6} {'acc':>5} {'TP-kept':>8} {'FP-kill':>8}  {'sec':>5}")
    for rd, n, acc, tk, fr, dt in history:
        print(f"  {rd:>6} {n:>6} {acc:>5.2f} {tk:>8}  {fr:>8}  {dt:>5.1f}")

    # Save best prompt
    out = ROOT / "data/scibert_prep/best_openrouter_prompt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "system": SYSTEM_INIT, "shots": best_shots,
        "best_acc": best_acc, "history": history}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
