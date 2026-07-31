"""Extract 5 math + 5 ml papers (different domains) with v2 column-aware."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2  # noqa: E402

PAPER_LIST = [('math', '2002.06309'), ('math', '2205.00834'), ('math', '2012.04551'),
              ('math', '2010.03116'), ('math', '2212.08837'),
              ('ml', '2212.13902'), ('ml', '2010.06408'), ('ml', '2412.09369'),
              ('ml', '2111.1214'), ('ml', '2005.14425')]


def main():
    out_dir = ROOT / "data/dataset_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    for domain, pid in PAPER_LIST:
        pdf = ROOT / f"runs/{domain}/data/pdfs/{pid}.pdf"
        if not pdf.exists():
            print(f"  {domain} {pid}: PDF missing")
            continue
        t0 = time.time()
        blocks = extract_pdf_blocks_v2(pdf)
        text = blocks_to_text_v2(blocks)
        rec = {"id": pid, "domain": domain, "text": text, "n_chars": len(text),
               "n_blocks": len(blocks),
               "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
               "blocks": blocks}
        papers.append(rec)
        print(f"  {domain:5s} {pid}: {len(text)} chars, {rec['n_headings']} headings ({time.time()-t0:.1f}s)")

    out = out_dir / "papers_v5.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {out} ({len(papers)} papers)")


if __name__ == "__main__":
    main()
