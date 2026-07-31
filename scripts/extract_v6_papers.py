"""Extract 4 ai + 3 math + 3 ml fresh papers for v6 generalization test."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2  # noqa: E402

PAPER_LIST = [('ai', '2403.09017'), ('ai', '2511.17323'), ('ai', '2502.1898'),
              ('ai', '2503.21902'), ('math', '2310.09808'), ('math', '2008.09911'),
              ('math', '2203.15386'), ('ml', '2004.13148'), ('ml', '1805.08728'),
              ('ml', '2002.09677')]


def main():
    out_dir = ROOT / "data/dataset_v6"
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    for domain, pid in PAPER_LIST:
        pdf = ROOT / f"runs/{domain}/data/pdfs/{pid}.pdf"
        if not pdf.exists():
            print(f"  {domain} {pid}: missing")
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

    out = out_dir / "papers_v6.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {out} ({len(papers)} papers)")


if __name__ == "__main__":
    main()
