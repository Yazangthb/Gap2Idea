# -*- coding: utf-8 -*-
"""Diagnostic: does LaTeX-recovered section structure lift arXiv's Stage-A ceiling?

arXiv gold papers are ingested as flat text, so Stage A's whole-section keep never
fires and the ceiling caps ~56% vs biomed's 84%. build_arxiv_latex_blocks.py recovers
clean sections from the LaTeX source (keeping the paper's own flat-text sentences). This
script re-runs Stage A on arXiv under two conditions and compares to biomed:

    flat   : blocks=None  (current ingestion)
    latex  : blocks from data/gold/arxiv_latex_blocks.jsonl (when available)

Metrics per condition:
  * ceiling recall  — fraction of gold limitation+future_work sentences that land in the
    Stage-A slice (the upper bound Stage B/C can reach);
  * retention       — mean kept sentences / paper and as a fraction of the paper body
    (how much load Stage A hands Stage B — lower at equal recall is the win).

    python scripts/bench/benchmark_arxiv_latex.py

Writes docs/experiments/stage_a_latex_sections.md.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import defaultdict

try:  # Windows consoles default to cp1252 and choke on ≥/→ in the report echo
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import slice_section_aware, token_containment
from gap2idea.pipeline.sections import _cut_before_references
from gap2idea.pipeline.gap_prefilter import split_sentences
from benchmark_gold import sal_blocks, TARGET, TAU

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
LATEX = ROOT / "data" / "gold" / "arxiv_latex_blocks.jsonl"
ARXIV_TEXT = ROOT / "data" / "paper_texts.jsonl"
OUTDOC = ROOT / "docs" / "experiments" / "stage_a_latex_sections.md"


def slice_sents(text, blocks):
    out, seen = [], set()
    for r in slice_section_aware(text, blocks=blocks):
        for s in r.sentences:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def ceiling_hits(slice_sentences, gold_targets):
    return sum(1 for s in gold_targets
               if any(token_containment(s, ss) >= TAU for ss in slice_sentences))


def main() -> None:
    # gold per paper
    gold = defaultdict(list)
    domain = {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        g = json.loads(ln)
        gold[g["paper_id"]].append((g["sentence"], g["label"]))
        domain[g["paper_id"]] = g["domain"]

    corp = {p["paper_id"]: p for p in load_corpus()}
    arxiv_text = {}
    for ln in ARXIV_TEXT.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln)
            arxiv_text[str(r["id"])] = str(r["text"])

    latex_blocks, latex_fmt = {}, {}
    if LATEX.exists():
        for ln in LATEX.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            r = json.loads(ln)
            latex_fmt[str(r["id"])] = r["src_format"]
            if r.get("blocks"):
                latex_blocks[str(r["id"])] = r["blocks"]

    # accumulators: cond -> domain -> dict
    acc = {c: defaultdict(lambda: {"hit": 0, "tgt": 0, "kept": 0, "body": 0, "papers": 0,
                                   "latex_used": 0})
           for c in ("flat", "latex", "biomed")}

    for pid in gold:
        p = corp.get(pid)
        if not p:
            continue
        dom = domain[pid]
        g_tgt = [s for s, lab in gold[pid] if lab in TARGET]
        if not g_tgt:
            pass  # still count retention

        if dom == "biomed_sal":
            blocks = sal_blocks(p["sents"], p["secs"])
            sl = slice_sents("", blocks)
            body = len([u for u in p["sents"]])
            a = acc["biomed"][dom]
            a["hit"] += ceiling_hits(sl, g_tgt); a["tgt"] += len(g_tgt)
            a["kept"] += len(sl); a["body"] += body; a["papers"] += 1
            continue

        text = arxiv_text.get(pid, " ".join(p["sents"]))
        body = len(split_sentences(_cut_before_references(text)))

        # flat
        sl_flat = slice_sents(text, None)
        a = acc["flat"][dom]
        a["hit"] += ceiling_hits(sl_flat, g_tgt); a["tgt"] += len(g_tgt)
        a["kept"] += len(sl_flat); a["body"] += body; a["papers"] += 1

        # latex (fall back to flat when no usable blocks)
        blocks = latex_blocks.get(pid)
        used_latex = blocks is not None
        sl_latex = slice_sents("" if used_latex else text, blocks) if used_latex else sl_flat
        a = acc["latex"][dom]
        a["hit"] += ceiling_hits(sl_latex, g_tgt); a["tgt"] += len(g_tgt)
        a["kept"] += len(sl_latex); a["body"] += body; a["papers"] += 1
        a["latex_used"] += int(used_latex)

    def row(a):
        rec = 100 * a["hit"] / a["tgt"] if a["tgt"] else 0.0
        kept = a["kept"] / a["papers"] if a["papers"] else 0.0
        ret = 100 * a["kept"] / a["body"] if a["body"] else 0.0
        return rec, kept, ret

    n_latex = sum(acc["latex"][d]["latex_used"] for d in acc["latex"])
    L = ["# Stage A — does LaTeX section recovery lift the arXiv ceiling?\n",
         f"arXiv gold papers with usable LaTeX sections: **{n_latex}** "
         f"(rest fall back to flat text). Matching: token-containment ≥ {TAU}; "
         "target gaps = limitation + future_work.\n",
         "## arXiv (CS) — flat vs LaTeX sections\n",
         "| condition | ceiling recall | mean kept sents/paper | retention (kept/body) |",
         "|---|---|---|---|"]
    for cond in ("flat", "latex"):
        rec, kept, ret = row(acc[cond]["arxiv_cs"])
        tag = "flat text (current)" if cond == "flat" else "LaTeX sections"
        L.append(f"| {tag} | {rec:.1f}% | {kept:.1f} | {ret:.1f}% |")
    L.append("\n## biomed (SAL) — reference upper bound (authored XML sections)\n")
    L.append("| condition | ceiling recall | mean kept sents/paper | retention (kept/body) |")
    L.append("|---|---|---|---|")
    rec, kept, ret = row(acc["biomed"]["biomed_sal"])
    L.append(f"| XML sections | {rec:.1f}% | {kept:.1f} | {ret:.1f}% |")

    # deltas
    rf, kf, tf = row(acc["flat"]["arxiv_cs"])
    rl, kl, tl = row(acc["latex"]["arxiv_cs"])
    L.append("\n## Verdict\n")
    L.append(f"- **ceiling recall:** {rf:.1f}% → {rl:.1f}%  (Δ {rl - rf:+.1f} pts)")
    L.append(f"- **mean kept sents/paper:** {kf:.1f} → {kl:.1f}  (Δ {kl - kf:+.1f})")
    L.append(f"- **retention:** {tf:.1f}% → {tl:.1f}%  (Δ {tl - tf:+.1f} pts)")
    L.append(f"- biomed (sectioned) ceiling is {rec:.1f}% — the structure-rich upper bound "
             "arXiv is being compared against.")
    L.append("\n_Note: LaTeX recovery keeps each paper's own flat-text sentences and only "
             "adds section boundaries, so the ceiling change isolates the effect of sectioning "
             "(no text-source mismatch vs the gold)._\n")

    doc = "\n".join(L) + "\n"
    OUTDOC.parent.mkdir(parents=True, exist_ok=True)
    OUTDOC.write_text(doc, encoding="utf-8")
    print(doc)
    print("wrote", OUTDOC)


if __name__ == "__main__":
    main()
