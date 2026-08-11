# Gap2Idea — Thesis Draft

> Sections: Research Questions → Literature Review → Method → Results → Current
> Progress. Grounded in the repository code (`src/gap2idea/`), the experiment
> records (`docs/experiments/`), the verified related-work sweep
> (`docs/related_work_analysis.md`), the reviewed literature sheet
> (82 papers, `artifacts/lit_review_papers.md`), and the benchmarks run to date.
> **Claim discipline:** no invented numbers; "competitive with / matches the
> range of," never "beats SOTA"; every pending item marked.

---

## 1. Research Questions

**Overall goal.** Scientific output grows faster than any researcher can read.
Turning that literature into *what has not yet been done* — concrete, novel,
feasible research directions — is a core scholarly task that does not scale
manually, and that naïve per-paper LLM use cannot scale *economically* (a
per-paper extractor costs ≈$4,000 per million papers and is rate-limited). This
thesis builds and evaluates **Gap2Idea**, an end-to-end system that mines
research gaps from papers, organises them into a graph, and generates evaluated
research ideas. Three research questions structure both the system and the
literature.

**RQ2 — Text-Based Gap Mining.** *Can research gaps (limitations, future-work
directions) be extracted from scientific papers at corpus scale without a
per-paper LLM, at a quality competitive with fine-tuned transformers?*
- RQ2a: What architecture localises and classifies gap sentences cheaply and at
  high recall? (structural slice → cheap classifier → LLM precision filter)
- RQ2b: Is extraction quality bounded by **training data or model capacity**?
- RQ2c: How does the funnel compare, on public benchmarks (LimGen, BAGELS, the
  future-work corpus), to fine-tuned and generative baselines?

**RQ1 — Graph-Based Forecasting.** *Can research gaps be organised into a
multi-relational graph whose structure identifies promising, novel directions
(by recombination / bridging) better than ungrounded pairing?*
- RQ1a: How to embed and cluster gaps into interpretable communities?
- RQ1b: Do graph-derived **bridge** and **frontier** seeds yield better idea
  candidates than random gap pairs? (the extrinsic clustering signal)

**RQ3 — LLM-Driven Ideation.** *Can an LLM, grounded in retrieved gap evidence,
generate novel, feasible, non-hallucinated research ideas, and can idea quality
be evaluated reliably?*
- RQ3a: What grounding and anti-hallucination constraints keep generated ideas
  faithful to evidence?
- RQ3b: Does a multi-agent critic/revise loop improve idea quality over a single
  pass?
- RQ3c: Can idea quality be assessed by combining an LLM-judge panel with a human
  expert study, and do LLM judgments correlate with human ratings?

The pipeline instantiates the three in sequence: **RQ2 (extract) → RQ1 (graph) →
RQ3 (ideate)**, with a cross-cutting evaluation methodology.

---

## 2. Literature Review

*Organised by the three RQs, drawing on the 82 reviewed papers plus the
adversarially-verified extraction sweep. Confidence tags from the sweep: `✓✓`
independently confirmed, `~` likely/unverified.*

### 2.1 RQ2 — Text-based gap mining

**The task and its two shapes.** Work on "gaps" splits into *extraction* (locate
the sentences that state a limitation/future-work) and *generation* (synthesise
a limitations/future-work passage). Extraction systems converge on a single
architecture — **high-recall structural/lexical retrieval → high-precision
classifier filter** — which is exactly the funnel this thesis uses:
- **Hu & Wan (2015, arXiv 1507.02140):** regex future-work extractor → 4-way
  classifier `✓✓`.
- **Zhang et al. (2022, J. Informetrics / arXiv 2212.13860):** two stages —
  Naïve Bayes binary recognition (**Macro-F1 90.7%** on their corpus) → SciBERT
  6-way typing (weighted-F1 72.6%) `✓✓`. Releases a labelled NLP/ACL future-work
  corpus (9,009 FWS / 55,887 non-FWS).
- **RCT self-acknowledged limitations (PMC11807350, 2024):** keyword slice →
  fine-tuned **PubMedBERT** filter, **detection F1 0.821**, scaled to ~12k
  articles with no per-paper LLM `~`. The closest analog to our funnel; supplies
  a realistic target (F1 ≈ 0.82) and confirms rules alone are a strong baseline
  (rule-only 0.800 vs 0.821).
- **FutureGen (2025, arXiv 2503.16561):** states the pattern outright — regex
  high-recall → **LLM filter** → RAG generation — and trims input "to reduce API
  cost," i.e. it pays exactly the per-paper cost our cheap Stage B avoids `✓✓`.

**Limitation-focused datasets and generation.**
- **LimGen (2024, arXiv 2403.15529):** harvests the **mandated "Limitations"
  sections** of 4,068 ACL papers as gold and benchmarks BART/PEGASUS/T5/Llama-2
  + DPR for *suggestive limitation generation*. This mandated-section harvest is
  the distant-supervision recipe this thesis adopts.
- **BAGELS (2025, arXiv 2505.18207):** builds a limitations dataset over
  ACL/NeurIPS/PeerJ (verbatim author-stated + peer-review-supplemented) and
  benchmarks generation (BERTScore/ROUGE + coverage). Its ACL split, scored by
  Coverage-of-Ground-Truth, is the benchmark used in §4.
- **Al Azher et al. (LimTopic JCDL'24; graph-limitation JCDL'24; visual
  limitations BigData'24):** the same group iterating on limitation
  *generation* — topic-modelled aggregation, per-paper graph + RAG, and a
  multimodal chart-limitation variant.
- **Sci-Sentence (2025, arXiv 2508.04337)** and **Problem/Method sentences
  (2026, arXiv 2606.26481):** sentence-level classification of literature-review
  / problem / method sentences; the latter reports fine-tuned models beat LLM
  in-context learning on this task, and diagnoses formulaic-expression
  shortcut-learning — a failure mode any cue-based classifier inherits.
- **GAPMAP (2025, arXiv 2510.25055)** and **LimitGen / "Can LLMs Identify
  Critical Limitations" (2025, arXiv 2507.02694):** push toward *implicit* gaps
  (not author-stated), the harder, more useful frontier — and the boundary of
  what a faithful *extractor* (this thesis) deliberately does not attempt.

**The hard problems, corroborated.** The literature independently confirms our
findings: gaps concentrate in **terminal sections** (Discussion/Future-Work/
Conclusion) but some hide mid-paper `✓`; authors conflate future-work with
limitations, and in a 60-paper study >50% of stated limitations cannot be
discerned from text alone `~`; **binary detection is easy, fine-grained typing is
hard for everyone** (SciBERT wF1 72.6% on 6 future-work types; PubMedBERT F1 0.49
on 24 limitation sub-types) `✓✓`. Extraction datasets are **domain-locked** (ACL,
biomedical); none targets cross-domain arXiv AI/ML/math, motivating our own small
gold set.

### 2.2 RQ1 — Graph-based forecasting

The reviewed RQ1 cluster (34 papers) frames "finding promising directions" as a
**graph/link-prediction-over-time** problem, which motivates our gap-graph:
- **Knowledge-graph link prediction for trend forecasting:** *Science4Cast*
  ("Predicting the Future of AI with AI," arXiv 2210.00881) forecasts future
  concept links in a 100k-paper network; *Impact4Cast* and "Forecasting
  high-impact research topics via ML on evolving knowledge graphs"
  (arXiv 2402.08640) predict *high-impact* future links; "Forecasting the future
  of AI…" (Nat. Mach. Intell. 2023) and *Technology opportunity analysis…*
  (dual link prediction, 70c) do the same for patents/tech.
- **Emerging-topic detection over dynamic graphs:** "Tracking the dynamics of
  co-word networks" (48c) and "Study on the predictability of new topics of
  scholars" use temporal co-word/knowledge networks; *HorizonAI* ("Reconstructing
  Scientific History to Forecast Future Trends") couples a knowledge graph with an
  LLM.
- **Graph topic modelling / clustering** (the machinery for our communities):
  GINopic (graph-isomorphism topic model, arXiv 2404.02115), NGTM, "GCN-
  strengthened topic modeling," graph-based topic-diversity clustering, and
  multiview online research-topic clustering — the graph-native analogues of the
  Leiden-community step.
- **Graph representation learning** (embedding + negative sampling): S2GAE
  (156c), "Understanding Negative Sampling in Graph Representation Learning"
  (MCNS, 263c) — foundational for how gap nodes are embedded and linked.
- **Recombination / hypothesis generation as link prediction:** "Link prediction
  for hypothesis generation" (temporal graph, active curriculum) and
  "Literature-based Hypothesis Generation" tie graph structure directly to *new
  idea* production — the RQ1→RQ3 bridge.

Gap2Idea's contribution relative to this cluster is to run the same
graph/recombination machinery over **extracted research-gap nodes** (not
concept/citation nodes), and to expose **bridge** (edge-betweenness) and
**frontier** structure as *idea seeds*.

### 2.3 RQ3 — LLM-driven ideation

The reviewed RQ3 cluster (21 + hybrid) covers generation, grounding, and
evaluation:
- **Knowledge-graph-grounded ideation:** "Generation and human-expert evaluation
  of interesting research ideas using knowledge graphs and LLMs" (MPG) — closest
  to our graph-seeded generation with a human study; **CHIMERA** (arXiv
  2505.20779), a knowledge base of *idea recombinations*; **CAM** (creative
  analogy mining); the **Ramon Llull machine** (arXiv 2508.19200) — combinatorial
  theme×domain×method ideation, the explicit recombination hypothesis.
- **Idea-development tools:** IdeaSynth, SCI-IDEA (token/sentence-embedding "aha"
  detection), *Scaffolding Flexible Ideation Workflows*, CRISP-IM — human-in-the-
  loop idea facet evolution and workflow scaffolding.
- **Evidence/quality studies:** "Can ChatGPT generate scientific hypotheses?"
  (73c); "Comparing the Ideation Quality of Humans With Generative AI";
  "Improving Research Idea Generation Through Data" (reports +20% feasibility /
  +7% quality from metadata grounding) — motivating our evidence-grounding and
  human evaluation.
- **Surveys / role framings:** "A Review of LLM-Assisted Ideation" (61 studies),
  "The Evolving Role of LLMs in Scientific Innovation" (Evaluator/Collaborator/
  Scientist), AI4Research, "From AI for Science to Agentic Science" — position
  Gap2Idea as an *end-to-end* system spanning all three roles.
- **LLM-as-judge:** used across the cluster (FutureGen, BAGELS, AbGen's ablation
  meta-evaluation); the recurring caveat — **self-evaluation bias** when judge and
  generator share a model family — directly motivates our cross-provider panel +
  human study.

**Where Gap2Idea sits.** Individually, each component has precedent. The
contribution is (i) the **cost-efficient extraction funnel** feeding (ii) a
**gap-graph** whose bridges/frontiers seed (iii) **grounded, hallucination-gated,
multi-agent** ideation, validated by (iv) a **human-anchored** evaluation
methodology — an integrated, deployed, reproducible pipeline rather than any
single new model.

---

## 3. Method

### 3.0 System overview

Six stages, passing plain files (`data/` in, `artifacts/` out), exposed as a CLI,
a Streamlit app, and an MCP server:

```
select-papers → download-pdfs → extract-text (GROBID; PyMuPDF fallback)
   → gap extraction (funnel: Stage A/B/C)            [RQ2]
   → theme-mine (embed → gap-graph: Leiden + bridge/frontier)   [RQ1]
   → generate-ideas (grounded, multi-agent, gated)   [RQ3]
   → evaluate-ideas (LLM panel + S2 novelty + human study)
   → export-ideas (LaTeX/PDF/full-paper) · serve-mcp
```

An LLM-provider abstraction (`pipeline/llm.py`) routes every LLM call through one
OpenAI-compatible client; `LLM_PROVIDER` selects OpenRouter (multi-model) or
YandexGPT, with model-slug rewriting so no call-site changes when switching.

### 3.1 Ingestion (RQ2 input)

`extract-text` parses PDFs with **GROBID** (TEI section tree; ML-based, reliable
on scrambled two-column PDFs), falling back to **PyMuPDF** when GROBID is
unavailable. GROBID ingestion is the corpus-scale "PDF-parsing line item" and
aligns with the AllenAI S2ORC pipeline (which also wraps GROBID); it feeds clean
text + a real section tree to the whole pipeline (`pipeline/grobid_sections.py`,
`pipeline/gap_funnel.slice_grobid_regions`).

### 3.2 Gap extraction — the funnel (RQ2)

Three stages of increasing cost, each shrinking the input for the next
(`pipeline/gap_funnel.py`, `pipeline/gap_llm_filter.py`):

- **Stage A — structural slice** (`slice_terminal_regions` / `slice_grobid_regions`):
  on the reading-order sentence stream, union three recall sources — a span after
  a Limitations/Future-Work/Conclusion **heading**, a ±window around an **inline
  keyword** (catches column-scrambled headings), and the **terminal tail**. With
  GROBID, keep Limitations/Future-Work/Discussion/Introduction sections and
  blacklist Related-Work/Background. Position-gated, priority-capped. Free
  (regex/CPU); **drops ~82% of sentences.**
- **Stage B — classify** (`cue_label` + `EmbeddingGapHead`): high-precision cue
  rules give a free fast-accept + type; a logistic head over frozen
  **bge-small-en-v1.5** embeddings catches cue-less gaps, but only inside
  *explicit* Limitations/Future-Work regions (tail/discussion → rules only, else
  false positives flood). Self-distilled from teacher labels with negatives drawn
  from the body *outside* the slice; the shipped head adds **~1,500 harvested ACL
  Limitations sentences** as clean positives.
- **Stage C — LLM precision filter** (`gap_llm_filter.LLMGapFilter`): an LLM
  judges only the ~6 survivors/paper and drops false positives (acknowledgments,
  formulas, citations, contribution claims, **prior-work critiques**, PDF
  scramble). Two design refinements from this work: (i) a **batched judge** — one
  structured `json_schema` call per ~40 candidates instead of one per sentence,
  and (ii) **section-aware protection** — cue-rule hits inside explicit
  Limitations/Future-Work sections are trusted; hits in Discussion/Intro/tail
  (where prior-work critiques concentrate) are judged. The judge prompt
  explicitly rejects *prior-work* limitations and vague self-promotion.

A **context ablation** found that feeding the judge the surrounding paragraph
(±1 sentence) or the paper title *lowered* precision — a gap sentence and its
adjacent motivating/prior-work sentence have opposite ownership, so any local
window blurs the own-vs-prior boundary the bare sentence draws cleanly.

### 3.3 Gap graph — theme mining (RQ1)

`pipeline/theme_mining.py` / `gap_graph.py`: gap sentences are embedded, a
multi-relational graph is built (semantic-similarity, same-paper, same-section,
shared-method edges), **Leiden** community detection partitions it, and
**edge-betweenness** identifies *bridge* gaps (spanning communities) and
*frontier* nodes; these seed novelty-by-recombination. *[Method verified against
code; extrinsic evaluation pending — see §5.]*

### 3.4 Idea generation (RQ3)

`pipeline/openai_ideas.py`, `agents.py`, `orchestrator.py`: five modes (**bridge,
within-community, method-gap, frontier, orchestrated**) turn seeds into ideas,
grounded in retrieved evidence with **anti-hallucination gates** — verbatim-
evidence constraint, evidence-overlap check, and required **named baseline +
falsifiable prediction**. The orchestrated mode runs a **multi-agent critic /
revise / sanity** loop (`agents.py`, `sanity.py`).

### 3.5 Evaluation (cross-cutting)

`pipeline/evaluation.py`: a **cross-provider LLM-judge panel** scores ideas
(novelty / specificity / feasibility / grounding) with inter-judge agreement; an
automated **Semantic-Scholar novelty** check (`semantic_scholar.py`); and a
**human expert study** form (Krippendorff's α), with LLM-judge-vs-human
correlation as the justification for using the LLM panel at all.

### 3.6 Output & deployment

`export.py`, `paper_drafter.py`: per-idea LaTeX (`minimal`/`standard`/`ieee`), a
consolidated PDF, or a `--full-paper` plan; `mcp_server.py` exposes the corpus to
Claude Desktop / Cursor. Containerised (Docker), deployable on Cloud Run, with CI
and a test suite.

---

## 4. Results

*Extraction (RQ2) is the benchmarked component. RQ1/RQ3 results are pending real
runs (§5) — reported honestly, not fabricated.*

### 4.1 Stage A — localisation (the recall ceiling)

On the clean 19-gap / 9-paper gold (`data/bench_gap/`, token-containment match):

| containment τ | recall all | future_work | limitation |
|---|---|---|---|
| 0.80 | **0.84** | **0.91** | 0.75 |
| 0.70 | 0.89 | 1.00 | 0.75 |

Stage A drops **82%** of sentences for free. Future-work localisation is
effectively solved; the residual misses are *mid-paper* own-work limitations
(structural, not slicer, limits).

### 4.2 Stage B / end-to-end, and the data-not-model finding

| Stage B head | preds/paper | gap recall | limitation recall | type acc |
|---|---|---|---|---|
| rules only | 2.2 | 0.32 | 0.11 | 1.00 |
| hybrid (bge+logreg) | 4.2 | 0.42 | 0.11 | 1.00 |
| **+ ACL limitations** *(shipped)* | 6.1 | **0.53** | **0.44** | 0.90 |

**Data, not model (clean negative result):** classifier method (logreg ≈
DistilBERT ≈ SetFit) *and* frozen encoder (bge-small ≈ bge-base ≈ mpnet ≈
SPECTER) all tie at limitation recall ≈ 0.11; only **clean data** — harvesting
mandated ACL Limitations sections — moved limitation recall **0.11 → 0.44 (4×)**
and end-to-end **0.42 → 0.53**, saturating at ~1,500 sentences. Upgrading the
model is a dead end; the literature agrees (domain-BERT also caps ~0.5 on
limitation typing).

### 4.3 Cost (RQ2's headline)

| Approach | Cost / 1M papers |
|---|---|
| Per-paper LLM (`openai_gaps`, gpt-4.1-mini, ~9k tok) | **~$4,000** |
| Funnel (Stage A slice + Stage B head) | **~$3–31** |
| Stage C batched judge (this work) | **~$25–30** (≈2 calls / 25 papers; −88% tokens vs per-sentence) |

The funnel is **~130–160× cheaper**; adding the batched Stage C keeps it within
the same order (~$30/1M vs ~$180/1M for a per-sentence LLM filter).

### 4.4 Same-data comparison vs prior art (LimGen)

Leakage-clean, on LimGen ACL data: our cheap **frozen stacking ensemble F1 0.627**
beats reproduced Zhang BernoulliNB (0.553), TF-IDF (0.610), and base bge (0.610),
and is ~0.05 behind fine-tuned DistilBERT (0.674). **Stage C lift (196-sentence
sample):** Stage B 0.643 → **+Stage C 0.725** (precision 0.56 → 0.79) — indicative
that the funnel *matches the range of a fine-tuned transformer at a fraction of
the cost* (stated as indicative given sample size).

### 4.5 New benchmarks (this work)

- **BAGELS extraction (ACL, 127 papers, 1,212 verbatim gold limitation
  sentences).** Funnel Stage A+B coverage of gold: **verbatim 0.738; semantic
  (bge cos ≥ .75) 0.790** (per table: ACL_23 0.822/0.859, ACL_24 0.699/0.758).
  BAGELS is a *generation* benchmark; its best generation Coverage-of-Ground-Truth
  is **76.62% (GPT-3.5)**. On the comparable coverage metric our *extraction*
  matches/exceeds their best *generation* — but this is an easier task (section
  present) with a stricter matcher, so it is framed as **establishing the first
  cheap extraction baseline on BAGELS**, not beating their leaderboard. Stage C is
  a precision filter; on a limitations-only gold (no negatives) it can only cost
  coverage, so A+B is the coverage figure to quote.
- **Future-work sentence corpus (Zhang et al., 64,896 sentences).** Sentence
  classification is a *lexical-cue* task on pre-segmented sentences, which favours
  bag-of-words. Our head: **zero-shot macro-F1 0.765; +Stage C 0.772** (Stage C
  removed 314 false positives vs 118 true FWS → precision 0.80→0.88). We do **not**
  beat their reported 0.907 — but that number is **not reproducible**: a faithful
  reproduction of their own Bernoulli-NB on the full corpus reaches **0.820**, and
  all text-only methods (their NB, TF-IDF, our embeddings) cluster at 0.76–0.82.
  These pre-segmented sentence benchmarks do not exercise the funnel's actual
  contributions (structural localisation, corpus-scale extraction from raw PDFs).
- **Provider robustness note.** YandexGPT occasionally issues a content-filter
  refusal on a batch; the per-sentence fallback in Stage C absorbs it with no lost
  paper — a small reliability data point for single-provider deployments.

---

## 5. Current Progress

Honest component scorecard (built ≠ evaluated):

| Component | Built | Evaluated | State |
|---|---|---|---|
| **RQ2 — Gap extraction (funnel A/B/C)** | ✅ | ✅ small gold + LimGen + **BAGELS/FWS (this work)** | **paper-ready** with honest, indicative claims |
| **RQ1 — Gap graph (Leiden + bridge/frontier)** | ✅ novel | ⚠️ prior bench was N=11 and excluded the shipped graph method; no extrinsic eval | **needs real eval** on hundreds of gaps from `runs/*` |
| **RQ3 — Idea generation (5 modes + multi-agent)** | ✅ substantive | ⚠️ committed runs are the simple path; no orchestrated/critic runs saved | **needs runs** |
| **Idea evaluation / novelty** | ✅ code (panel, S2, human form, α) | ❌ panel unused (single judge, self-bias); human study has **no real responses**; S2 novelty on ~7/27 | **make-or-break; not yet credible** |
| **Output + deployment (MCP, Docker, CI, drafter)** | ✅ strong | ✅ 19 tests | **paper-ready** |

**Done this session (extraction hardening + benchmarking):** YandexGPT provider
integration; **batched Stage C** (2 calls / 25 papers, −88% tokens) with prior-work
+ self-promotion rejection and section-aware protection; the **context ablation**
(bare > context); the **BAGELS extraction benchmark** (0.74 verbatim / 0.79
semantic) with a fair comparison to BAGELS's own generation numbers; the **FWS
benchmark** + the **reproducibility finding** (reported SOTA 0.907 does not
reproduce; ~0.82 is the reproducible bar); a **dataset-suitability survey**
concluding no ideal cross-domain full-paper extraction benchmark exists.

**Highest-value next steps (from `docs/paper/00_full_system_review.md`), in
order:**
1. **P0 — evaluation integrity:** delete all placeholder/fake human data; run a
   **real human study** (5–10 domain readers, ~15 ideas, Krippendorff's α); run
   the **cross-provider judge panel** and report **LLM-vs-human correlation**; fix
   S2 novelty coverage.
2. **P1 — justify the pipeline logic:** the **graph-vs-random ablation** — do
   bridge/frontier seeds produce ideas rated higher (panel + humans, blind)? This
   is the experiment that earns every upstream component its place. Plus a
   **with/without-critic** ablation for the multi-agent path.
3. **P2 — component evals at real size:** re-run clustering on hundreds of `runs/*`
   gaps **including** the shipped Leiden-graph method; enlarge the extraction gold
   (19 → a few hundred).

**Framing for write-up:** a **system + evaluation-methodology** contribution —
cost-efficient extraction, graph-seeded recombination, anti-hallucination
engineering, deployability, and a **human-validated** idea-quality study — not a
"beats SOTA" claim.
