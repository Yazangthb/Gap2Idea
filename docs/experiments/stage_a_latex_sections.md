# Stage A — does LaTeX section recovery lift the arXiv ceiling? (No.)

**TL;DR — negative result, decisive.** arXiv's low Stage-A ceiling (~56% vs biomed's 84%)
is **not** caused by missing section structure. Recovering clean, authored sections from the
LaTeX source (175/190 papers, 92%) leaves the ceiling flat (55.6% → 55.2%) and only raises
retention. The gap is caused by *where CS/ML gaps live*, not by parsing: 82% of the missed
gaps sit in Methods/Experiments/Results — sections a wrap-up gate would never keep. This
closes the "give arXiv sections" line (incl. MinerU) for the purpose of recall.

Matching throughout: token-containment ≥ 0.7; target gaps = limitation + future_work (the
funnel's scope; research_gap is the LLM/full-paper path). The LaTeX recovery keeps each
paper's **own flat-text sentences** and only adds section boundaries, so the ceiling change
isolates the effect of sectioning — no text-source mismatch against the gold.

## 1. arXiv (CS) — flat vs LaTeX sections

| condition | ceiling recall | mean kept sents/paper | retention (kept/body) |
|---|---|---|---|
| flat text (current) | 55.6% | 47.7 | 18.2% |
| LaTeX sections | 55.2% | 56.2 | 21.4% |

Reference upper bound — biomed (SAL), authored XML sections:

| condition | ceiling recall | mean kept sents/paper | retention |
|---|---|---|---|
| XML sections | **84.1%** | 57.5 | 28.8% |

**Sectioning bought nothing for recall (Δ −0.5 pts) and cost retention (+8.6 kept sents/paper).**

## 2. Why — where the missed arXiv gaps actually live

On the 175 LaTeX-sectioned arXiv papers (867 hit / 684 missed target gaps, ceiling 55.9%),
classifying each **missed** gap by the LaTeX section it belongs to:

| location of missed gap | count | share of misses |
|---|---|---|
| **NON-kept section** (Methods / Experiments / Results / Evaluation / …) | **560** | **82%** |
| kept section (Disc/Concl/Lim/Fut/**Intro**) but not sliced | 72 | 11% |
| not matched to any section (text / PDF scramble) | 52 | 8% |

Top sections where missed gaps live: Experiments 90, Introduction 60, Methods 35, Results 34,
Preliminaries 21, Evaluation 21, Methodology 20, Background 13, Numerical experiments 12.

**Oracle bound.** Even a *perfect* section parser that keeps every
Discussion/Conclusion/Limitations/Future/Introduction section wholesale would reach only
(867 + 72) / 1551 = **60.5%** — the 560 gaps in Methods/Experiments/Results stay out. To
approach biomed's 84% you would have to keep Methods+Experiments+Results too, i.e. keep most
of the paper, which defeats the point of a Stage-A gate.

## 3. Interpretation

CS/ML papers **do not concentrate their limitations and future work in a terminal section.**
They state them inline, next to the relevant result — "our method only evaluates on X",
"this assumes Y", "we leave Z to future work" — distributed across Methods, Experiments, and
Results. Biomedical papers (the SAL set) follow the clinical template with a canonical
"Limitations" paragraph inside Discussion, which is why a section gate works there (84%) and
not on arXiv (56%). This is a **content-distribution difference between fields**, not a
document-parsing problem.

## 4. Consequences for the design

- **Do not pursue section recovery for arXiv recall** — not LaTeX, not MinerU, not GROBID.
  All are sectioning tools; the ceiling is bounded by gap *location*, which sectioning cannot
  change. The 92%-available LaTeX source confirmed the hypothesis is testable and the answer
  is no.
- **The recall-first, section-agnostic Stage A is vindicated.** The learned recall-first
  localizer (keep ~62%, R ≈ 96.5% on the flat text) works precisely *because* it does not
  assume gaps are localized in a wrap-up section — it scores every sentence. This result
  explains why that design beats a section gate on arXiv.
- **Recall must come from the detector (Stage B), not the structural gate** — which points
  straight at the Stage-B fine-tuning lever (frozen encoder plateaus ~F1 0.45; a fine-tuned
  transformer is the next step). Sectioning is orthogonal and does not help.

## 5. Where sections still may help (not recall)

- biomed stays sectioned (XML) — no change.
- For arXiv, LaTeX sections could still improve **precision/attribution** (knowing a sentence
  is in Related Work lets us *drop* prior-work "gaps"), and the `section_type` provenance
  column is nicer for downstream theme mining. That is a precision/metadata use, evaluated
  separately — it is not a recall lever.

---
_Build: `scripts/dataset/build_arxiv_latex_blocks.py` → `data/gold/arxiv_latex_blocks.jsonl`
(175 latex / 13 pdf-only / 2 unlocatable). Diagnostic:
`scripts/bench/benchmark_arxiv_latex.py`. Location analysis logged 2026-10-27._
