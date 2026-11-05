# -*- coding: utf-8 -*-
"""For arXiv papers with LaTeX sections: where do MISSED target gaps live?
Classify each gold limitation/future_work gap by (a) in Stage-A slice? and
(b) which LaTeX section it belongs to (kept-type vs not)."""
import sys, json
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path(r"C:\Users\yazan\OneDrive\Desktop\thesis\Gap2Idea")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from gap2idea.pipeline.gap_funnel import (slice_section_aware, token_containment,
    SECTION_HEAD_RE, classify_grobid_heading)

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
LATEX = ROOT / "data" / "gold" / "arxiv_latex_blocks.jsonl"
TARGET = {"limitation", "future_work"}
TAU = 0.7

gold = defaultdict(list); domain = {}
for ln in GOLD.read_text(encoding="utf-8").splitlines():
    if ln.strip():
        g = json.loads(ln); gold[g["paper_id"]].append((g["sentence"], g["label"])); domain[g["paper_id"]] = g["domain"]

latex = {}
for ln in LATEX.read_text(encoding="utf-8").splitlines():
    if ln.strip():
        r = json.loads(ln)
        if r.get("blocks"): latex[str(r["id"])] = r["blocks"]

def section_of(sent, blocks):
    """Return (heading, kept?) of the section whose body best contains `sent`."""
    best, best_sc, cur_head = None, 0.0, "(preamble)"
    for b in blocks:
        if b["role"] == "heading":
            cur_head = b["text"]
        else:
            sc = token_containment(sent, b["text"])
            if sc > best_sc:
                best_sc, best = sc, cur_head
    if best_sc < TAU:
        return None, False
    kept = bool(SECTION_HEAD_RE.search(best)) or classify_grobid_heading(best) in {
        "limitations", "future_work", "discussion", "introduction"}
    return best, kept

verified_section_regions = 0
miss_in_kept = miss_in_other = miss_unfound = 0
hit_total = miss_total = 0
other_heads = Counter()
for pid, blocks in latex.items():
    if domain.get(pid) != "arxiv_cs":
        continue
    regions = slice_section_aware("", blocks=blocks)
    if any(r.section_type == "section" for r in regions):
        verified_section_regions += 1
    slice_sents = {s for r in regions for s in r.sentences}
    for sent, lab in gold[pid]:
        if lab not in TARGET:
            continue
        in_slice = any(token_containment(sent, ss) >= TAU for ss in slice_sents)
        if in_slice:
            hit_total += 1
            continue
        miss_total += 1
        head, kept = section_of(sent, blocks)
        if head is None:
            miss_unfound += 1
        elif kept:
            miss_in_kept += 1; other_heads[f"KEPT:{head[:30]}"] += 1
        else:
            miss_in_other += 1; other_heads[head[:30]] += 1

print(f"latex arXiv papers: {len(latex)} ; with >=1 whole-section region: {verified_section_regions}")
print(f"target gaps  hit={hit_total}  miss={miss_total}  (ceiling {100*hit_total/(hit_total+miss_total):.1f}%)")
print(f"\nMISSED gaps by location:")
print(f"  in a KEPT section (Disc/Concl/Lim/Fut/Intro) but not sliced: {miss_in_kept}")
print(f"  in a NON-kept section (Methods/Exp/Related/etc.)           : {miss_in_other}")
print(f"  not matched to any section body (text/scramble)            : {miss_unfound}")
print(f"\nTop sections where missed gaps live:")
for h, c in other_heads.most_common(15):
    print(f"  {c:3d}  {h}")
