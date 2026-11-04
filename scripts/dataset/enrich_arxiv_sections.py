# -*- coding: utf-8 -*-
"""Recover authoritative section structure for the arXiv gold papers via GROBID, and
measure the Stage-A localization ceiling with real sections vs the flat-text fallback.

Why: arXiv papers were ingested as flat text (no sections), which caps Stage A at ~56%
recall. We recover sections WITHOUT re-segmenting (so the gold `idx` labels stay valid):
GROBID -> section tree; each EXISTING gold sentence is mapped to the section whose text
contains it (token containment).

Outputs:
  data/gold/arxiv_sections.json       {paper_id: {"sections":[{n,heading,cls,text}],
                                                   "sent_sec":[cls per existing sentence idx],
                                                   "grobid_ok":bool}}
  docs/experiments/stage_a_grobid.md  ceiling comparison (GROBID vs flat text)

Needs a running GROBID (docker run --rm -d -p 8070:8070 lfoppiano/grobid:0.8.1).
  python scripts/dataset/enrich_arxiv_sections.py
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import (
    slice_section_aware, slice_grobid_regions, token_containment, classify_grobid_heading,
)
from gap2idea.pipeline.grobid_sections import extract_sections, grobid_available

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
PDFS = ROOT / "data" / "pdfs"
OUT = ROOT / "data" / "gold" / "arxiv_sections.json"
DOC = ROOT / "docs" / "experiments" / "stage_a_grobid.md"
TAU = 0.7


def main():
    if not grobid_available():
        sys.exit("GROBID not reachable at localhost:8070 — start it first "
                 "(docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1).")

    gold = defaultdict(set); dom = {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            g = json.loads(ln); gold[g["paper_id"]].add(g["idx"]); dom[g["paper_id"]] = g["domain"]
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in dom}
    arxiv = [pid for pid in gold if dom[pid] == "arxiv_cs"]
    arxiv_text = {}
    for ln in (ROOT / "data" / "paper_texts.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); arxiv_text[str(r["id"])] = str(r["text"])

    store = {}
    # ceiling accumulators
    g_hit = g_tot = g_keep = g_total = 0      # GROBID
    f_hit = f_tot = f_keep = f_total = 0      # flat-text fallback
    ok = 0; nopdf = 0; fail = 0
    cls_counter = Counter()

    for n, pid in enumerate(sorted(arxiv), 1):
        p = corp[pid]; sents = p["sents"]; nsent = len(sents)
        pdf = PDFS / f"{pid}.pdf"
        secs = extract_sections(str(pdf)) if pdf.exists() else None
        if not pdf.exists(): nopdf += 1
        elif secs is None: fail += 1

        # ---- flat-text Stage A (baseline) ----
        fsl = [s for r in slice_section_aware(arxiv_text.get(pid, " ".join(sents))) for s in r.sentences]
        f_total += nsent; f_keep += len(fsl)
        for idx in gold[pid]:
            f_tot += 1; f_hit += any(token_containment(sents[idx], ss) >= TAU for ss in fsl)

        rec = {"grobid_ok": bool(secs)}
        if secs:
            ok += 1
            for s in secs:
                s["cls"] = classify_grobid_heading(s.get("heading", ""))
                cls_counter[s["cls"]] += 1
            # ---- GROBID Stage A ----
            gsl = [s for r in slice_grobid_regions(secs) for s in r.sentences]
            g_total += nsent; g_keep += len(gsl)
            for idx in gold[pid]:
                g_tot += 1; g_hit += any(token_containment(sents[idx], ss) >= TAU for ss in gsl)
            # ---- per-sentence section alignment (preserve gold idx) ----
            sent_sec = []
            for s in sents:
                best = max(secs, key=lambda sc: token_containment(s, sc.get("text", "")), default=None)
                sc = token_containment(s, best.get("text", "")) if best else 0.0
                sent_sec.append(best["cls"] if (best and sc >= 0.5) else None)
            rec["sections"] = [{"n": s.get("n"), "heading": s.get("heading"), "cls": s["cls"]} for s in secs]
            rec["sent_sec"] = sent_sec
        store[pid] = rec
        if n % 40 == 0:
            print(f"  {n}/{len(arxiv)} (grobid_ok={ok}, fail={fail}, nopdf={nopdf})", flush=True)

    OUT.write_text(json.dumps(store), encoding="utf-8")

    def rate(h, t): return f"{100*h/t:.1f}%" if t else "n/a"
    L = ["# Stage A on arXiv — GROBID sections vs flat text\n",
         f"arXiv gold papers: {len(arxiv)} | GROBID parsed: {ok} | failed: {fail} | no PDF: {nopdf}.\n",
         "Ceiling = recall of arXiv gold gaps inside the Stage-A slice (token-containment >= "
         f"{TAU}); retention = sentences kept. Measured on the SAME existing gold sentences.\n",
         "| Stage A input | ceiling recall | retention (kept) |\n|---|---|---|",
         f"| flat text (current) | {rate(f_hit, f_tot)} | {rate(f_keep, f_total)} |",
         f"| **GROBID sections** | **{rate(g_hit, g_tot)}** | {rate(g_keep, g_total)} |",
         f"\n(GROBID row over the {ok} papers it parsed; flat-text over all {len(arxiv)}.)\n",
         "GROBID heading classes seen: " + ", ".join(f"{k}:{v}" for k, v in cls_counter.most_common()),
         "\n### Notes\n- Sections aligned onto the EXISTING gold sentences (no re-segmentation) -> gold idx preserved.\n"
         "- `data/gold/arxiv_sections.json` holds per-paper section tree + per-sentence section label.\n"
         "- Enables a realistic full-structured-paper input for the scalable pipeline."]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nwrote", OUT, "and", DOC)


if __name__ == "__main__":
    main()
