"""Re-extract the 10 dataset papers with the column-aware v2 PDF extractor,
then re-run the precision-first dataset pipeline. Compares v1 vs v2 quality.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2  # noqa: E402
from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    slice_with_midpaper_anchors, _looks_like_sentence,
)
from gap2idea.pipeline.llm import get_llm_client  # noqa: E402
from test_strict_precision import SYS_STRICT, SYS_CONFIRM, call_batched  # noqa: E402


# The 10 papers we tested before
PAPER_IDS = ["2505.18658", "2309.06553", "2310.0565", "2402.12261", "2502.20968",
             "2402.15309", "2512.11573", "2406.13331", "2601.09515", "2509.24328"]


def find_context(sent, text, window=30):
    s = sent.strip()[:80]
    idx = text.find(s)
    if idx < 0:
        return "", ""
    end = idx + len(sent)
    before = " ".join(text[max(0, idx-400):idx].split()[-window:])
    after = " ".join(text[end:end+400].split()[:window])
    return before, after


def main():
    pdf_dir = ROOT / "runs/ai/data/pdfs"
    out_dir = ROOT / "data/dataset_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Phase 1: re-extract PDFs with v2 ===
    print("=" * 70)
    print("PHASE 1: re-extract PDFs with column-aware v2 extractor")
    print("=" * 70)
    papers = []
    for pid in PAPER_IDS:
        pdf = pdf_dir / f"{pid}.pdf"
        if not pdf.exists():
            print(f"  {pid}: PDF missing — skipping")
            continue
        t0 = time.time()
        blocks = extract_pdf_blocks_v2(pdf)
        text = blocks_to_text_v2(blocks)
        rec = {"id": pid, "text": text, "n_chars": len(text),
               "n_blocks": len(blocks),
               "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
               "blocks": blocks}
        papers.append(rec)
        print(f"  {pid}: {len(text)} chars, {len(blocks)} blocks, {rec['n_headings']} headings ({time.time()-t0:.1f}s)")

    # Save v2-extracted papers
    v2_jsonl = out_dir / "papers_v2.jsonl"
    with v2_jsonl.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {v2_jsonl}")

    # === Phase 2: compare v1 vs v2 on a known-bad sentence ===
    print("\n" + "=" * 70)
    print("PHASE 2: spot-check fix on previously corrupted paper 2502.20968")
    print("=" * 70)
    bad_paper = next((p for p in papers if p["id"] == "2502.20968"), None)
    if bad_paper:
        # Search for "One limitation of this study" — was bleeding into author names in v1
        snippet = bad_paper["text"]
        idx = snippet.find("One limitation of this study")
        if idx >= 0:
            print(f"Found 'One limitation of this study' at offset {idx}")
            print(f"  +400 chars context (v2):")
            print(f"  {snippet[idx:idx+400]!r}")
        else:
            print("  (not found in v2 extraction — checking alternative location)")
        # Check if "Tom Brown" (author name from references) appears mixed in mid-paper
        author_idx = snippet.find("Tom Brown, Benjamin Mann")
        if author_idx >= 0:
            # Check how far this is from References section
            ref_idx = snippet.lower().find("references")
            print(f"  'Tom Brown, Benjamin Mann' at offset {author_idx}, References at {ref_idx}")
            if ref_idx >= 0 and author_idx > ref_idx:
                print(f"  ✓ Author names ARE inside References section (fixed!)")
            else:
                print(f"  ✗ Author names still bleed into body text")

    # === Phase 3: re-run pipeline on v2 papers ===
    print("\n" + "=" * 70)
    print("PHASE 3: re-run precision-first pipeline on v2-extracted papers")
    print("=" * 70)
    items = []
    total_slice = 0
    for rec in papers:
        regs = slice_with_midpaper_anchors(rec["text"], blocks=rec.get("blocks"))
        for r in regs:
            for s in r.sentences:
                if not _looks_like_sentence(s):
                    continue
                b, a = find_context(s, rec["text"])
                items.append((s, b, a, rec["id"]))
        total_slice += sum(len(r.sentences) for r in regs)
    print(f"  Slice: {total_slice} sentences across {len(papers)} papers ({total_slice/len(papers):.0f}/paper)")

    # dedup
    seen = set(); dedup_items = []
    for s, b, a, pid in items:
        key = (pid, " ".join(s.lower().split())[:60])
        if key in seen:
            continue
        seen.add(key); dedup_items.append((s, b, a, pid))
    print(f"  After dedup: {len(dedup_items)} sentences")
    items = dedup_items

    client = get_llm_client()

    # Pass 1
    print(f"\n  Pass 1 (strict GAP/JUNK) ...", flush=True)
    t0 = time.time()
    p1 = call_batched(client, "openai/gpt-4o", SYS_STRICT, items, label_yes="GAP", label_no="JUNK")
    p1_items = [it for it, k in zip(items, p1) if k]
    print(f"    kept {len(p1_items)}/{len(items)} ({time.time()-t0:.1f}s)")

    # Pass 2
    print(f"\n  Pass 2 (confirm) ...", flush=True)
    t0 = time.time()
    p2 = call_batched(client, "openai/gpt-4o", SYS_CONFIRM, p1_items, label_yes="YES", label_no="NO")
    final = [it for it, k in zip(p1_items, p2) if k]
    print(f"    confirmed {len(final)}/{len(p1_items)} ({time.time()-t0:.1f}s)")

    # Save dataset v2
    rows = [{"paper_id": pid, "gap_sentence": s, "context_before": b, "context_after": a}
            for s, b, a, pid in final]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "paper_gaps.tsv", sep="\t", index=False)
    print(f"\nDataset v2 saved: {out_dir}/paper_gaps.tsv ({len(df)} gaps)")

    # === Phase 4: print all gaps for user review ===
    print("\n" + "=" * 70)
    print("PHASE 4: ALL GAPS for validation")
    print("=" * 70)
    if df.empty:
        print("  (no gaps)")
    else:
        for pid, sub in df.groupby("paper_id"):
            print(f"\n--- {pid} ({len(sub)} gaps) ---")
            for i, (_, row) in enumerate(sub.iterrows(), 1):
                ctx = f"...{row['context_before'][-50:]} **{row['gap_sentence'][:130]}** {row['context_after'][:50]}..."
                print(f"  {i}. {ctx}")


if __name__ == "__main__":
    main()
