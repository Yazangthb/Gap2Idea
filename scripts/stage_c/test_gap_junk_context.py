"""GAP vs JUNK prompt with surrounding context. Honest held-out evaluation.

For each test sentence, fetch ~30 words of context BEFORE and AFTER from LimGen
data (search in 'limitations' field, fall back to 'content'). Feed the LLM the
sentence with >>> markers and its context, then classify GAP/JUNK.

    python -u scripts/stage_c/test_gap_junk_context.py --model openai/gpt-4o --window 30
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
    "For each numbered input, classify the highlighted sentence (between >>> <<<) "
    "as GAP or JUNK, using the surrounding context to disambiguate.\n\n"
    "GAP = the authors mention something NOT YET DONE in their own work:\n"
    "  - a limitation, weakness, or assumption of their method\n"
    "  - a scope restriction ('focuses only on English')\n"
    "  - a future-work direction ('we leave X for future work')\n"
    "  - an open problem or unanswered question\n\n"
    "JUNK = anything else, including:\n"
    "  - gratitude / acknowledgments\n"
    "  - numbered performance results ('95% accuracy', 'outperforms by 3 points')\n"
    "  - method or hyperparameter descriptions\n"
    "  - prior-work citations\n"
    "  - truncated fragments\n\n"
    "Output ONLY '<idx>. GAP' or '<idx>. JUNK' per line, nothing else."
)

SHOTS = [
    ("...we evaluate on six benchmarks. >>> Our method achieves 95.2% accuracy, "
     "outperforming baselines by 3.1 points. <<< Detailed tables are reported in "
     "Appendix A.", "JUNK"),
    ("...We acknowledge two main limitations. >>> First, our data augmentation "
     "strategy relies on the reconstruction ability of cycle adversarial nets. <<< "
     "This dependence may limit applicability when synthetic data is scarce.", "GAP"),
    ("...The authors thank our funding agency. >>> We would like to express our "
     "gratitude to Dr. Smith for sharing the data. <<< Their feedback helped shape "
     "the manuscript.", "JUNK"),
    ("...we use a single GPU. >>> We leave the multi-GPU implementation for "
     "future work. <<< This restricts our experiments to mid-sized models.", "GAP"),
    ("...Following standard preprocessing pipelines, >>> we use the commonly-used "
     "PaddleOCR to handle our dataset and obtain recognized texts. <<< The "
     "recognition quality is reported in Section 3.", "JUNK"),
]


def load_split():
    losses = json.load(open(ROOT / "data/scibert_prep/stage_c_recall_losses.json"))
    kills = json.load(open(ROOT / "data/scibert_prep/stage_c_correct_kills.json"))
    keeps = json.load(open(ROOT / "data/scibert_prep/stage_c_wrong_keeps.json"))
    samples = [(r["sentence"], 1, "limitations") for r in losses] \
            + [(r["sentence"], 0, "content") for r in kills] \
            + [(r["sentence"], 0, "content") for r in keeps]
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(samples))
    split = len(samples) // 2
    return [samples[i] for i in idx[split:]]  # held-out TEST split only


def load_limgen():
    """Load LimGen test+train .jsonl and concatenate (we need all papers to search)."""
    papers = []
    for fn in ["test.jsonl", "train.jsonl", "valid.jsonl"]:
        p = ROOT / "data" / "limgen" / fn
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    papers.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return papers


def find_context(sentence, papers, source_field, window=30):
    """Find sentence in papers, return (before, after) as strings of words.
    Searches the specified source field first, then the other."""
    fields = [source_field, "content" if source_field == "limitations" else "limitations"]
    needle = sentence.strip()
    # Truncate the needle for search: use first 80 chars for speed
    short = needle[:80] if len(needle) > 80 else needle
    for rec in papers:
        for field in fields:
            text = str(rec.get(field, "") or "")
            if not text:
                continue
            idx = text.find(short)
            if idx < 0:
                continue
            end = idx + len(needle)
            before_chars = text[max(0, idx - 400):idx].strip()
            after_chars = text[end:end + 400].strip()
            before = " ".join(before_chars.split()[-window:])
            after = " ".join(after_chars.split()[:window])
            return before, after
    return "", ""


def predict_batched(client, model, samples_with_ctx, batch_size=10):
    """samples_with_ctx is list of (sentence, before, after) tuples."""
    shot_lines = []
    for k, (s, a) in enumerate(SHOTS):
        shot_lines.append(f"Example {k+1}: {s}\nLabel {k+1}: {a}")
    shot_block = "\n".join(shot_lines)

    preds = [None] * len(samples_with_ctx)
    for i in range(0, len(samples_with_ctx), batch_size):
        chunk = samples_with_ctx[i:i + batch_size]
        # Build context-augmented input lines
        input_lines = []
        for k, (sent, before, after) in enumerate(chunk):
            ctx = f"{before} >>> {sent} <<< {after}".strip()
            input_lines.append(f"{k+1}. {ctx}")
        input_block = "\n".join(input_lines)
        user = f"Reference examples:\n{shot_block}\n\nClassify:\n{input_block}"
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=10 * len(chunk))
            content = resp.choices[0].message.content
            ans = {}
            for ln in content.splitlines():
                m = re.match(r"\s*(\d+)\s*[.):\-]\s*(GAP|JUNK)", ln.strip(), re.IGNORECASE)
                if m:
                    ans[int(m.group(1))] = m.group(2).upper() == "GAP"
            for k in range(len(chunk)):
                preds[i + k] = ans.get(k + 1, True)
        except Exception as e:
            print(f"  batch error: {e}", flush=True)
            for k in range(len(chunk)):
                preds[i + k] = True
    return preds


def metrics(samples, preds):
    tp = sum(1 for _, y, _ in samples if y == 1)
    fp = sum(1 for _, y, _ in samples if y == 0)
    tp_kept = sum(1 for (_, y, _), p in zip(samples, preds) if y == 1 and p)
    fp_kill = sum(1 for (_, y, _), p in zip(samples, preds) if y == 0 and not p)
    return tp_kept / max(1, tp), fp_kill / max(1, fp), tp_kept, tp, fp_kill, fp


def project_f1(tp_ret, fp_kill):
    sb_tp, sb_fp = 2293, 539
    new_tp = sb_tp * tp_ret
    new_fp = sb_fp * (1 - fp_kill)
    p = new_tp / max(1, new_tp + new_fp)
    r = new_tp / 3338
    return p, r, 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    test = load_split()
    print(f"Held-out TEST: {len(test)} samples "
          f"({sum(1 for _,y,_ in test if y==1)} TPs / "
          f"{sum(1 for _,y,_ in test if y==0)} FPs)", flush=True)

    print("Loading LimGen papers for context lookup ...", flush=True)
    papers = load_limgen()
    print(f"  {len(papers)} papers loaded", flush=True)

    print(f"Fetching {args.window}-word context for each test sentence ...", flush=True)
    t0 = time.time()
    samples_with_ctx = []
    found = 0
    for sent, gold, source in test:
        before, after = find_context(sent, papers, source, args.window)
        samples_with_ctx.append((sent, before, after))
        if before or after:
            found += 1
    print(f"  context found for {found}/{len(test)} sentences ({time.time()-t0:.1f}s)", flush=True)

    client = get_llm_client()
    print(f"\nRunning gpt batched (model={args.model}, batch={args.batch}) ...", flush=True)
    t0 = time.time()
    preds = predict_batched(client, args.model, samples_with_ctx, args.batch)
    dt = time.time() - t0
    print(f"  done in {dt:.1f}s", flush=True)

    tp_ret, fp_kill, tp_k, tp, fp_k, fp = metrics(
        [(s, g, src) for (s, g, src), _ in zip(test, range(len(test)))], preds)
    p, r, f1 = project_f1(tp_ret, fp_kill)

    print(f"\n=== GAP/JUNK with context ({args.window}w window, {args.model}) ===")
    print(f"  TP retention: {tp_k}/{tp} = {tp_ret:.3f}")
    print(f"  FP kill rate: {fp_k}/{fp} = {fp_kill:.3f}")
    print(f"\n  Projected full-LimGen F1: {f1:.3f}  (was 0.743)  Δ = {f1 - 0.743:+.3f}")
    print(f"  Projected P: {p:.3f} (was 0.809)")
    print(f"  Projected R: {r:.3f} (was 0.687)")

    # Show some examples of wrong predictions
    wrong_kills = [(s, b, a) for (s, b, a), (sent, g, src), pr in
                    zip(samples_with_ctx, test, preds) if not pr and g == 1][:5]
    wrong_keeps = [(s, b, a) for (s, b, a), (sent, g, src), pr in
                    zip(samples_with_ctx, test, preds) if pr and g == 0][:5]
    if wrong_kills:
        print(f"\n  BAD kills (real gaps the LLM said were junk):")
        for s, b, a in wrong_kills:
            print(f"    ...{b[-60:]} >>> {s[:80]} <<< {a[:60]}...")
    if wrong_keeps:
        print(f"\n  MISSED junk (LLM kept these but they're SciBERT FPs):")
        for s, b, a in wrong_keeps:
            print(f"    ...{b[-60:]} >>> {s[:80]} <<< {a[:60]}...")


if __name__ == "__main__":
    main()
