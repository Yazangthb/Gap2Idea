"""Extract 10 fresh papers (not in dataset_v3) with v2 column-aware extractor."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2  # noqa: E402

PAPER_IDS = ["2407.16431", "2407.04841", "2508.1594", "2408.06227", "2411.1977",
             "2601.06185", "2409.19037", "2404.10102", "2407.0638", "2508.0681"]


def main():
    pdf_dir = ROOT / "runs/ai/data/pdfs"
    out_dir = ROOT / "data/dataset_v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    for pid in PAPER_IDS:
        pdf = pdf_dir / f"{pid}.pdf"
        if not pdf.exists():
            print(f"  {pid}: PDF missing")
            continue
        t0 = time.time()
        blocks = extract_pdf_blocks_v2(pdf)
        text = blocks_to_text_v2(blocks)
        rec = {"id": pid, "text": text, "n_chars": len(text),
               "n_blocks": len(blocks),
               "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
               "blocks": blocks}
        papers.append(rec)
        print(f"  {pid}: {len(text)} chars, {rec['n_headings']} headings ({time.time()-t0:.1f}s)")

    out = out_dir / "papers_v4.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {out} ({len(papers)} papers)")


if __name__ == "__main__":
    main()
