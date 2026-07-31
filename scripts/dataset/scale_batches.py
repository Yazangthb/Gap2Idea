"""Scale-out: pick 40 fresh papers (4 batches of 10), extract them, send all to GPU
for Stage A/B in one shot, then run Stage C V5 + an LLM auto-validator that
classifies each gap as VALID / GARBLED / FP. Append clean ones to cumulative.
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


def sanitize(s):
    if not isinstance(s, str):
        return s
    return ''.join(c for c in s if c.isprintable() or c in '\n\t ')


def main():
    import random
    random.seed(11)

    # Exclude all previously-used papers
    prev = set([
        '2309.09902','2507.03922','2511.13548','2510.04514','2002.09564','2003.01908','2102.04998','2208.03805','2511.03443','2211.01962',
        '2505.18658','2309.06553','2310.0565','2402.12261','2502.20968','2402.15309','2512.11573','2406.13331','2601.09515','2509.24328',
        '2407.16431','2407.04841','2508.1594','2408.06227','2411.1977','2601.06185','2409.19037','2404.10102','2407.0638','2508.0681',
        '2002.06309','2205.00834','2012.04551','2010.03116','2212.08837','2212.13902','2010.06408','2412.09369','2111.1214','2005.14425',
        '2403.09017','2511.17323','2502.1898','2503.21902','2310.09808','2008.09911','2203.15386','2004.13148','1805.08728','2002.09677',
        '2302.00509','2502.20573','2403.01384','2405.08644','2201.05342','1909.10327','2511.04622','2502.01282','2008.08289','2002.04406',
        '2310.01329','2409.16686','2510.22317','2506.14064','1807.01343','2407.20805','2009.11481','2007.05824','2507.11367','1904.10631',
    ])

    def pool(domain):
        return [p.stem for p in (ROOT / f"runs/{domain}/data/pdfs").glob('*.pdf')
                if p.stem not in prev and p.stat().st_size > 100_000]

    ai = pool('ai'); math = pool('math'); ml = pool('ml')
    print(f"pools: ai={len(ai)} math={len(math)} ml={len(ml)}")
    random.shuffle(ai); random.shuffle(math); random.shuffle(ml)

    # Take everything we can — but cap at 40 total
    # mix: equal weights, 13/13/14 across ai/math/ml or by availability
    selection = []
    for d, lst in [('ai', ai), ('math', math), ('ml', ml)]:
        n = min(14, len(lst))
        for pid in lst[:n]:
            selection.append((d, pid))
        print(f"  picked {n} {d} papers")
    print(f"Total picked: {len(selection)}")

    # Extract all
    from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2
    out_dir = ROOT / "data/dataset_v9plus"
    out_dir.mkdir(parents=True, exist_ok=True)

    papers = []
    for d, pid in selection:
        pdf = ROOT / f"runs/{d}/data/pdfs/{pid}.pdf"
        if not pdf.exists():
            continue
        t0 = time.time()
        try:
            blocks = extract_pdf_blocks_v2(pdf)
            for b in blocks:
                b["text"] = sanitize(b.get("text", ""))
            text = sanitize(blocks_to_text_v2(blocks))
        except Exception as e:
            print(f"  {pid}: extract failed ({e})")
            continue
        if len(text) < 1000:
            continue
        rec = {"id": pid, "domain": d, "text": text, "n_chars": len(text),
               "n_blocks": len(blocks),
               "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
               "blocks": blocks}
        papers.append(rec)
        print(f"  {d:5s} {pid}: {len(text)} chars ({time.time()-t0:.1f}s)")

    out = out_dir / "papers_v9plus.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"\nWrote {out} ({len(papers)} papers)")


if __name__ == "__main__":
    main()
