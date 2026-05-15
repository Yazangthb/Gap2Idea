"""One-paper smoke test: compare OLD (plain-text regex) vs NEW (style-aware
heading detection + expanded vocab) on a single arXiv PDF.

Run:  uv run python scripts/test_extraction_v2.py data/pdf_test/0807.0023.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

from gap2idea.pipeline.pdf_text import (
    blocks_to_text, extract_pdf_blocks, extract_pdf_text,
)
from gap2idea.pipeline.sections import (
    DISCUSSION_HEAD_RE, FUTURE_HEAD_RE, LIMITATION_HEAD_RE,
    extract_sections_for_paper,
)


def _summary(sections: list[dict]) -> None:
    if not sections:
        print("  (no sections found)")
        return
    for s in sections:
        head = s.get("heading", "")
        st = s.get("section_type", "")
        body = s.get("section_text", "")
        print(f"  [{st:11s}] heading={head!r}  body={len(body)}ch  preview={body[:120]!r}")


def main(pdf_path: str) -> None:
    pdf = Path(pdf_path)
    print(f"== {pdf.name} ==\n")

    # OLD path: plain text only
    print("--- OLD: plain-text regex ---")
    text = extract_pdf_text(pdf)
    old_sections = extract_sections_for_paper(pdf.stem, text, blocks=None)
    _summary(old_sections)

    # NEW path: style-aware blocks
    print("\n--- NEW: style-aware blocks ---")
    blocks = extract_pdf_blocks(pdf)
    n_headings = sum(1 for b in blocks if b["role"] == "heading")
    print(f"  detected {n_headings} headings out of {len(blocks)} blocks")
    print("  heading candidates that matched a keyword regex:")
    for b in blocks:
        if b["role"] != "heading":
            continue
        title = b["text"]
        match = None
        if LIMITATION_HEAD_RE.search(title):
            match = "limitations"
        elif FUTURE_HEAD_RE.search(title):
            match = "future_work"
        elif DISCUSSION_HEAD_RE.search(title):
            match = "discussion"
        if match:
            print(f"    [{match:11s}]  size={b['size']:.1f}  bold={b['bold']}  text={title!r}")

    text_from_blocks = blocks_to_text(blocks)
    new_sections = extract_sections_for_paper(pdf.stem, text_from_blocks, blocks=blocks)
    print("\n  sections kept:")
    _summary(new_sections)


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "data/pdf_test/0807.0023.pdf"
    main(p)
