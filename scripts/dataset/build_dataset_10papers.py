"""Build a small gap-dataset on 10 fresh AI papers using the precision-first
pipeline. Picks 10 papers from runs/ai that are NOT in the gold set.

Pipeline:
  1. Stage A v2 — slice_with_midpaper_anchors (local, free)
  2. Stage C strict — gpt-4o GAP/JUNK with default-reject (skips SciBERT since
     we don't need a heavy classifier — Stage A v2 keeps ~70 sents/paper and
     strict-LLM is precise enough for a small dataset)
  3. Stage C confirm — second gpt-4o pass, both must agree

Output: data/dataset_v1/paper_gaps.tsv (paper_id, gap_sentence, context_before, context_after)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    slice_with_midpaper_anchors, _looks_like_sentence,
)
from gap2idea.pipeline.llm import get_llm_client  # noqa: E402
from test_strict_precision import SYS_STRICT, SYS_CONFIRM, call_batched  # noqa: E402


def find_context(sent, text, window=30):
    s = sent.strip()[:80]
    idx = text.find(s)
    if idx < 0:
        return "", ""
    end = idx + len(sent)
    before = " ".join(text[max(0, idx-400):idx].split()[-window:])
    after = " ".join(text[end:end+400].split()[:window])
    return before, after


def pick_10_papers():
    """Pick 10 fresh AI papers from runs/ai that are NOT in the gold set."""
    gold_ids = set()
    mani = ROOT / "data/bench_gold/papers_manifest.tsv"
    if mani.exists():
        gold_ids = set(pd.read_csv(mani, sep="\t", dtype=str)["id"])
    src = ROOT / "runs/ai/data/paper_texts.jsonl"
    if not src.exists():
        print(f"ERROR: {src} not found")
        return []
    candidates = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = str(rec.get("id"))
        if pid in gold_ids:
            continue
        if not rec.get("text") or len(rec.get("text", "")) < 5000:
            continue
        candidates.append(rec)
    # Pick the 10 LONGEST (more likely to be real research papers)
    candidates.sort(key=lambda r: -len(r.get("text", "")))
    return candidates[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    args = ap.parse_args()

    papers = pick_10_papers()
    if not papers:
        print("No papers found.")
        return
    print(f"Selected {len(papers)} papers:")
    for r in papers:
        print(f"  {r['id']}  ({len(r.get('text', ''))} chars)")

    # === Stage A v2 — slice ===
    print(f"\n=== Stage A v2 ===", flush=True)
    items = []
    total_slice = 0
    for rec in papers:
        regs = slice_with_midpaper_anchors(rec["text"], blocks=rec.get("blocks"))
        sents = [(s, r.section_type) for r in regs for s in r.sentences]
        for sent, sec in sents:
            if not _looks_like_sentence(sent):
                continue
            b, a = find_context(sent, rec["text"])
            items.append((sent, b, a, rec["id"]))
        total_slice += len(sents)
    print(f"  slice: {total_slice} sentences from {len(papers)} papers ({total_slice/len(papers):.0f}/paper)")
    # dedup
    seen = set(); deduped = []
    for s, b, a, pid in items:
        key = (pid, " ".join(s.lower().split())[:60])
        if key in seen:
            continue
        seen.add(key); deduped.append((s, b, a, pid))
    print(f"  after dedup: {len(deduped)} sentences")
    items = deduped

    client = get_llm_client()

    # === Stage C strict ===
    print(f"\n=== Stage C strict ({args.model}) ===", flush=True)
    t0 = time.time()
    pass1 = call_batched(client, args.model, SYS_STRICT, items, label_yes="GAP", label_no="JUNK")
    p1 = [it for it, k in zip(items, pass1) if k]
    print(f"  pass 1 kept {len(p1)}/{len(items)} ({time.time()-t0:.1f}s)")

    # === Stage C confirm ===
    print(f"\n=== Stage C confirm ===", flush=True)
    t0 = time.time()
    pass2 = call_batched(client, args.model, SYS_CONFIRM, p1, label_yes="YES", label_no="NO")
    final = [it for it, k in zip(p1, pass2) if k]
    print(f"  pass 2 confirmed {len(final)}/{len(p1)} ({time.time()-t0:.1f}s)")

    # === Save dataset ===
    out_dir = ROOT / "data/dataset_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"paper_id": pid, "gap_sentence": s, "context_before": b, "context_after": a}
             for s, b, a, pid in final]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paper_gaps.tsv", sep="\t", index=False)

    # Stats
    per_paper = df.groupby("paper_id").size().sort_values(ascending=False)
    print(f"\n=== DATASET SUMMARY ===")
    print(f"  total gaps: {len(df)} from {df['paper_id'].nunique()} papers")
    print(f"  mean: {len(df)/len(papers):.1f} gaps/paper")
    print(f"\n  Per-paper:")
    for pid, n in per_paper.items():
        print(f"    {pid}: {n} gaps")

    # Show output for validation
    print(f"\n=== ALL GAPS (for validation) ===")
    for pid, sub in df.groupby("paper_id"):
        print(f"\n--- {pid} ---")
        for i, (_, row) in enumerate(sub.iterrows(), 1):
            ctx = f"...{row['context_before'][-40:]} **{row['gap_sentence'][:120]}** {row['context_after'][:40]}..."
            print(f"  {i}. {ctx}")

    print(f"\nsaved -> {out_dir}/paper_gaps.tsv")


if __name__ == "__main__":
    main()
