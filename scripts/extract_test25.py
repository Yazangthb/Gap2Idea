"""Pick 25 random fresh papers (excluding ALL prior runs), extract for held-out test."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2  # noqa: E402


def sanitize(s):
    if not isinstance(s, str):
        return s
    return ''.join(c for c in s if c.isprintable() or c in '\n\t ')


def main():
    import random
    random.seed(2025)

    prev = set()
    # All jsonls of prior batches
    for vf in ["data/dataset_v3/paper_gaps.tsv",
                "data/dataset_v4/papers_v4.jsonl",
                "data/dataset_v5/papers_v5.jsonl",
                "data/dataset_v6/papers_v6.jsonl",
                "data/dataset_v7/papers_v7.jsonl",
                "data/dataset_v8/papers_v8.jsonl",
                "data/dataset_v9plus/papers_v9plus.jsonl",
                "data/dataset_r2/papers_r2.jsonl"]:
        p = ROOT / vf
        if not p.exists():
            print(f"  skipping {vf} (missing)")
            continue
        if vf.endswith(".jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    prev.add(json.loads(line)["id"])
        else:
            df = pd.read_csv(p, sep="\t", dtype={"paper_id": str})
            for pid in df["paper_id"]:
                prev.add(str(pid))

    # Explicit v3 papers (in case the read missed them)
    for pid in ["2505.18658","2309.06553","2310.0565","2402.12261","2502.20968",
                "2402.15309","2512.11573","2406.13331","2601.09515","2509.24328",
                "2309.09902","2507.03922","2511.13548","2510.04514","2002.09564",
                "2003.01908","2102.04998","2208.03805","2511.03443","2211.01962"]:
        prev.add(pid)

    print(f"Total excluded: {len(prev)}")

    def pool(domain):
        out = []
        for v in ["", "_v1"]:
            d = ROOT / f"runs/{domain}{v}/data/pdfs"
            if d.exists():
                for p in d.glob("*.pdf"):
                    if p.stem not in prev and p.stat().st_size > 100_000:
                        out.append((f"{domain}{v}" if v else domain, p.stem))
        return out

    ai = pool("ai"); math = pool("math"); ml = pool("ml")
    print(f"Available: ai={len(ai)} math={len(math)} ml={len(ml)}")
    random.shuffle(ai); random.shuffle(math); random.shuffle(ml)

    # Pick 8 ai + 8 math + 9 ml = 25
    chosen = ai[:8] + math[:8] + ml[:9]
    random.shuffle(chosen)

    out_dir = ROOT / "data/dataset_test25"
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    for d, pid in chosen:
        pdf = ROOT / f"runs/{d}/data/pdfs/{pid}.pdf"
        if not pdf.exists():
            print(f"  {d} {pid}: missing")
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
        domain_clean = d.split("_")[0]
        rec = {"id": pid, "domain": domain_clean, "source": d, "text": text,
               "n_chars": len(text), "n_blocks": len(blocks),
               "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
               "blocks": blocks}
        papers.append(rec)
        print(f"  {d:8s} {pid}: {len(text)} chars ({time.time()-t0:.1f}s)")

    out = out_dir / "papers_test25.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"\nWrote {len(papers)} papers -> {out}")
    print(f"By domain: ai={sum(1 for p in papers if p['domain']=='ai')}, "
          f"math={sum(1 for p in papers if p['domain']=='math')}, "
          f"ml={sum(1 for p in papers if p['domain']=='ml')}")


if __name__ == "__main__":
    main()
