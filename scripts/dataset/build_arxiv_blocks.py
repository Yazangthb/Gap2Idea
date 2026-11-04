# -*- coding: utf-8 -*-
"""Recover arXiv section structure WITHOUT GROBID, from the PDFs we already have.

Heading signal = union of (1) the PDF outline/bookmarks (exact LaTeX \\section titles,
present in ~half the papers), (2) numbered-section regex ("4 Discussion", "5.1 ..."),
(3) strict font/bold gating. Much tighter than the generic pdf_to_blocks heuristic.

Produces heading/text blocks per paper -> slice_section_aware(blocks). Measures the Stage-A
localization ceiling (recall of arXiv gold gaps in the slice) vs the current flat-text path.
Gold alignment preserved: we match gold SENTENCES to slice sentences by token containment
(no re-segmentation of the gold).

  python scripts/dataset/build_arxiv_blocks.py

Writes data/gold/arxiv_blocks.jsonl ({id, blocks}) and docs/experiments/stage_a_nogrobid.md.
"""
from __future__ import annotations
import sys, json, re, statistics
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
import fitz
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import (
    _pdf_lines_pymupdf, slice_section_aware, token_containment, classify_grobid_heading,
)
from gap2idea.pipeline.gap_prefilter import normalize_text

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
PDFS = ROOT / "data" / "pdfs"
OUTB = ROOT / "data" / "gold" / "arxiv_blocks.jsonl"
DOC = ROOT / "docs" / "experiments" / "stage_a_nogrobid.md"
TAU = 0.7
_NUMHEAD = re.compile(r"^\s*(\d+(\.\d+)*)\.?\s+[A-Z][A-Za-z0-9].{1,58}$")
_CAPTION = re.compile(r"^\s*(figure|fig\.|table|tab\.|algorithm|appendix)\s*\d", re.IGNORECASE)
_ENDPUNC = (".", ",", ";", ":")


def outline_titles(doc):
    out = []
    for lvl, title, pg in doc.get_toc(simple=True):
        t = re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", title).strip()  # strip leading number
        if t:
            out.append(normalize_text(t)[:60])
    return set(out)


def build_blocks(pdf):
    doc = fitz.open(pdf)
    titles = outline_titles(doc)
    lines = _pdf_lines_pymupdf(pdf)  # [{text,size,bold}]
    doc.close()
    if not lines:
        return []
    body = statistics.median([l["size"] for l in lines])
    blocks, buf = [], []

    def flush():
        if buf:
            blocks.append({"role": "text", "text": " ".join(buf)}); buf.clear()

    for l in lines:
        t = l["text"].strip(); w = t.split()
        is_head = False
        if t and not _CAPTION.match(t) and len(w) <= 12:
            norm = normalize_text(re.sub(r"^\s*\d+(\.\d+)*\.?\s*", "", t))[:60]
            is_outline = bool(titles) and any(norm and (norm in ti or ti in norm) for ti in titles)
            is_numbered = bool(_NUMHEAD.match(t))
            big = l["size"] >= body * 1.12
            bold = l["bold"] and l["size"] >= body * 0.98
            is_font = (big or bold) and not t.endswith(_ENDPUNC)
            is_head = is_outline or is_numbered or (is_font and len(w) <= 8)
        if is_head:
            flush(); blocks.append({"role": "heading", "text": t})
        else:
            buf.append(t)
    flush()
    return blocks


def main():
    gold = defaultdict(set); dom = {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            g = json.loads(ln); gold[g["paper_id"]].add(g["idx"]); dom[g["paper_id"]] = g["domain"]
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in dom}
    arxiv = sorted(pid for pid in gold if dom[pid] == "arxiv_cs")
    arxiv_text = {}
    for ln in (ROOT / "data" / "paper_texts.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); arxiv_text[str(r["id"])] = str(r["text"])

    g_hit = g_tot = g_keep = g_total = 0   # blocks path
    f_hit = f_tot = f_keep = f_total = 0   # flat text
    nhead = Counter(); target_papers = 0; nopdf = 0
    fout = OUTB.open("w", encoding="utf-8")
    for n, pid in enumerate(arxiv, 1):
        p = corp[pid]; sents = p["sents"]; nsent = len(sents)
        pdf = PDFS / f"{pid}.pdf"
        blocks = build_blocks(str(pdf)) if pdf.exists() else []
        if not pdf.exists(): nopdf += 1
        heads = [b["text"] for b in blocks if b["role"] == "heading"]
        cls = [classify_grobid_heading(h) for h in heads]
        if any(c in ("discussion", "limitations", "future_work") for c in cls):
            target_papers += 1
        for c in cls: nhead[c] += 1
        fout.write(json.dumps({"id": pid, "blocks": blocks}) + "\n")

        # flat-text baseline
        fsl = [s for r in slice_section_aware(arxiv_text.get(pid, " ".join(sents))) for s in r.sentences]
        f_total += nsent; f_keep += len(fsl)
        for idx in gold[pid]:
            f_tot += 1; f_hit += any(token_containment(sents[idx], ss) >= TAU for ss in fsl)
        # blocks path
        bsl = [s for r in slice_section_aware("", blocks=blocks) for s in r.sentences] if blocks else []
        g_total += nsent; g_keep += len(bsl)
        for idx in gold[pid]:
            g_tot += 1; g_hit += any(token_containment(sents[idx], ss) >= TAU for ss in bsl)
        if n % 40 == 0:
            print(f"  {n}/{len(arxiv)}", flush=True)
    fout.close()

    def rate(h, t): return f"{100*h/t:.1f}%" if t else "n/a"
    L = ["# Stage A on arXiv WITHOUT GROBID (PDF outline + numbered-heading + font)\n",
         f"arXiv gold papers: {len(arxiv)} (no PDF: {nopdf}). Papers where we recovered a "
         f"Discussion/Conclusion/Limitations/Future heading: **{target_papers}/{len(arxiv)}**.\n",
         f"Ceiling = recall of arXiv gold gaps in the Stage-A slice (token-containment >= {TAU}); "
         "retention = sentences kept. Same existing gold sentences (gold idx preserved).\n",
         "| Stage A input | ceiling recall | retention |\n|---|---|---|",
         f"| flat text (current) | {rate(f_hit, f_tot)} | {rate(f_keep, f_total)} |",
         f"| **PDF blocks (no GROBID)** | **{rate(g_hit, g_tot)}** | {rate(g_keep, g_total)} |",
         "\nRecovered heading classes: " + ", ".join(f"{k}:{v}" for k, v in nhead.most_common()),
         "\n### Notes\n- No Docker / no network: headings from the PDF outline (bookmarks) + numbered-"
         "section regex + strict font gating.\n- `data/gold/arxiv_blocks.jsonl` is the structured "
         "full-paper input (heading/text blocks) for the scalable pipeline.\n"
         "- Blocks feed `slice_section_aware(blocks=...)`; gold matched by sentence containment."]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L)); print("\nwrote", OUTB, DOC)


if __name__ == "__main__":
    main()
