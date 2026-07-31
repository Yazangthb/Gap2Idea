"""Extract 10 fresh papers (v7) for prompt validation."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2  # noqa: E402

PAPER_LIST = [('ai', '2302.00509'), ('ai', '2502.20573'), ('ai', '2403.01384'),
              ('ai', '2405.08644'), ('math', '2201.05342'), ('math', '1909.10327'),
              ('math', '2511.04622'), ('ml', '2502.01282'), ('ml', '2008.08289'),
              ('ml', '2002.04406')]


def sanitize(s):
    if not isinstance(s, str):
        return s
    return ''.join(c for c in s if c.isprintable() or c in '\n\t ')


def main():
    out_dir = ROOT / "data/dataset_v7"
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    for domain, pid in PAPER_LIST:
        pdf = ROOT / f"runs/{domain}/data/pdfs/{pid}.pdf"
        if not pdf.exists():
            print(f"  {domain} {pid}: missing")
            continue
        t0 = time.time()
        blocks = extract_pdf_blocks_v2(pdf)
        for b in blocks:
            b["text"] = sanitize(b.get("text", ""))
        text = sanitize(blocks_to_text_v2(blocks))
        rec = {"id": pid, "domain": domain, "text": text, "n_chars": len(text),
               "n_blocks": len(blocks),
               "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
               "blocks": blocks}
        papers.append(rec)
        print(f"  {domain:5s} {pid}: {len(text)} chars, {rec['n_headings']} headings ({time.time()-t0:.1f}s)")

    out = out_dir / "papers_v7.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"\nWrote {out} ({len(papers)} papers)")


if __name__ == "__main__":
    main()
