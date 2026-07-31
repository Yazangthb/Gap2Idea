"""Round 2: pick 50 fresh papers from extended pool (incl _v1 dirs), extract."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.pdf_text_v2 import extract_pdf_blocks_v2, blocks_to_text_v2  # noqa: E402


def sanitize(s):
    if not isinstance(s, str):
        return s
    return ''.join(c for c in s if c.isprintable() or c in '\n\t ')


def main():
    import random
    random.seed(53)

    # Aggregate all previously-used IDs
    prev = set()
    for vf in ["data/dataset_v3/paper_gaps.tsv",
                "data/dataset_v4/papers_v4.jsonl",
                "data/dataset_v5/papers_v5.jsonl",
                "data/dataset_v6/papers_v6.jsonl",
                "data/dataset_v7/papers_v7.jsonl",
                "data/dataset_v8/papers_v8.jsonl",
                "data/dataset_v9plus/papers_v9plus.jsonl"]:
        p = ROOT / vf
        if not p.exists():
            continue
        if vf.endswith(".jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    prev.add(json.loads(line)["id"])
        else:
            df = pd.read_csv(p, sep="\t", dtype={"paper_id": str})
            for pid in df["paper_id"]:
                prev.add(str(pid))
    # Hardcoded gold + v3
    for pid in ["2309.09902","2507.03922","2511.13548","2510.04514","2002.09564",
                "2003.01908","2102.04998","2208.03805","2511.03443","2211.01962",
                "2505.18658","2309.06553","2310.0565","2402.12261","2502.20968",
                "2402.15309","2512.11573","2406.13331","2601.09515","2509.24328"]:
        prev.add(pid)
    print(f"prev: {len(prev)} papers excluded")

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
    print(f"pools: ai={len(ai)} math={len(math)} ml={len(ml)}")
    random.shuffle(ai); random.shuffle(math); random.shuffle(ml)
    selection = ai[:17] + math[:17] + ml[:17]
    print(f"Picked {len(selection)} papers")

    out_dir = ROOT / "data/dataset_r2"
    out_dir.mkdir(parents=True, exist_ok=True)
    papers = []
    for d, pid in selection:
        # try both _v1 and main if needed
        pdf = ROOT / f"runs/{d}/data/pdfs/{pid}.pdf"
        if not pdf.exists():
            continue
        try:
            blocks = extract_pdf_blocks_v2(pdf)
            for b in blocks:
                b["text"] = sanitize(b.get("text", ""))
            text = sanitize(blocks_to_text_v2(blocks))
        except Exception:
            continue
        if len(text) < 1000:
            continue
        domain_clean = d.split("_")[0]
        rec = {"id": pid, "domain": domain_clean, "source": d, "text": text,
               "n_chars": len(text), "n_blocks": len(blocks),
               "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
               "blocks": blocks}
        papers.append(rec)

    out = out_dir / "papers_r2.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in papers:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    print(f"Wrote {len(papers)} papers -> {out}")


if __name__ == "__main__":
    main()
