# Gap Extraction — Flow & Architecture

> Living dev doc. Update the **Status** section as components land.
> Scope: how the pipeline turns raw papers into typed research-gap sentences,
> and the migration from per-paper LLM calls to a cheap, scalable funnel.
>
> **⚠️ Read order:** the **shipped** design is §7 (status) + §8 (the built funnel:
> Stage A/B/C). §3–§6 are the original *Tier 0/1/2 hypothesis*, kept for the
> reasoning trail but **superseded** — §5.1 shows why the lexical Tier 0 was
> replaced by the structural Stage-A slice. §5.1/§5.2 are cited from
> `gap_funnel.py`; do not renumber them.
>
> **Writing the paper?** All experiments, numbers, and reproduce steps are in
> [`experiments/`](experiments/README.md) — the lab notebook
> ([experiment_log.md](experiments/experiment_log.md)), the consolidated
> [results_registry.md](experiments/results_registry.md), and the prior-art
> [related_work_analysis.md](related_work_analysis.md). Stage A deep-dive:
> [stage_a_explained.md](stage_a_explained.md).

---

## 1. Why we're changing

The current Stage-1 extractor calls an LLM **once per paper**. That does not
scale to millions of papers:

| Approach | Cost / 1M papers | Wall clock |
|---|---|---|
| Per-paper LLM (`openai_gaps.py`, gpt-4.1-mini, ~9K tok) | **~$4,000** | days–weeks (rate-limited) |
| Funnel (Stage A slice + Stage B head, §8) | **~$3–31** | hours, offline, shardable |

Target metric at scale = **sentences classified per dollar**, offline and
embarrassingly parallel.

---

## 2. Current flow (as-is)

```
PDF / unarXive text
   │
   ▼
pdf_text.py  ──►  paper_texts.jsonl        (id, text, [blocks])
   │
   ▼
sections.py  ──►  sections_extracted.jsonl (id, section_type, heading, section_text)
   │              heading-regex; precision-tuned; falls back to tail/window
   ▼
openai_gaps.py ─► gaps.tsv                 (id, gap_type, gap_sentence, paragraph_text, confidence)
   │              LLM, ≤2 gaps/paper, VERBATIM sentences
   ▼
theme_mining.py / gap_graph.py / ...       (downstream, unchanged)
```

Known weaknesses (measured on the N=10 bench, see §5):
- `sections.py` is **precision-tuned for headings** → misses gaps under
  non-standard titles; 6/10 bench papers fell back to tail/window.
- `openai_gaps.py` **≤2 gaps/paper cap** → recall ceiling ~23%; macro-F1 0.19.
- Cost is linear in papers and gated by API rate limits.

---

## 3. Initial hypothesis — Tier 0/1/2 funnel (SUPERSEDED → see §8)

> ⚠️ This was the first design sketch. The **shipped** funnel (§8) replaced the
> lexical Tier 0 with a structural **Stage A** slice, made **Stage B** a cue-rule
> + embedding-head classifier, and made **Stage C** an LLM filter over survivors
> (`gap_llm_filter.py`). Kept for the reasoning trail (§5.1–§5.2).

Cheap → expensive, each tier shrinks the input for the next.

```
all sentences
   │
   ▼  Tier 0 — lexical prefilter   (gap_prefilter.py)        FREE, recall-tuned
   │   mined cue-phrase dictionary; drops ~90% of sentences
   │   PRIORITY = recall (it is the ceiling for everything below)
   ▼  ~10% survive
   │
   ▼  Tier 1 — small distilled classifier (local_gaps.py)    CHEAP, the workhorse
   │   MiniLM/DistilBERT fine-tuned 4-way {none,limitation,future_work,open_problem}
   │   int8 / ONNX; CPU fleet or 1 GPU
   ▼  gaps.tsv  (same schema as today → downstream unchanged)
   │
   ▼  Tier 2 — LLM audit on ~0.1% sample (optional)          drift detection only
       never in the bulk path
```

Training data for Tier 1 = **self-distillation**: the LLM teacher's existing
output (`runs/*/data/gaps.tsv`) as positives, same-paragraph unpicked
sentences as negatives. Zero new annotation.

---

## 4. Component map (shipped)

| File | Role |
|---|---|
| `pipeline/gap_funnel.py` | **the funnel** — Stage A (`slice_terminal_regions`), Stage B (`cue_label` rules + `EmbeddingGapHead`/`BertGapHead`), corpus runner `extract_all_gaps` |
| `pipeline/gap_llm_filter.py` | **Stage C** — LLM precision filter over survivors (local Qwen or API), validate/junk prompts |
| `pipeline/gap_prefilter.py` | shared text utils (`normalize_text`, `split_sentences`, n-grams) used by the funnel + the superseded miner |
| `data/gap_head.joblib` (+ `.meta.json`) | shipped Stage-B head: bge-small + 1500 ACL limitations |
| `scripts/training/train_gap_head.py` | trains the embedding head (self-distillation, negatives from body outside the slice) |
| `scripts/dataset/harvest_acl_limitations.py` | ACL mandated-Limitations harvest — the data fix (§8.4) |
| `scripts/dataset/build_gap_gold.py` | extract→verify gold → `data/bench_gap/gold_sentences.tsv` |
| `scripts/bench/bench_gap_recall.py` | Stage-A localization + end-to-end bench vs gold |
| CLI | `gap2idea extract-gaps-funnel --mode {rules,model,hybrid} --head data/gap_head.joblib` |

*Superseded (kept for the §5.1 reasoning trail):* the lexical Tier-0 dictionary —
`scripts/archive/mine_tier0_dictionary.py`, `scripts/archive/eval_tier0.py`,
`data/tier0_dictionary.json`, and the bench label sheet
(`scripts/archive/build_label_sheet.py`, `scripts/archive/eval_gap_extraction.py`). Replaced by
the structural Stage-A slice. `local_gaps.py` / `train_gap_classifier.py` were
never built — the real modules are `gap_funnel.py` / `train_gap_head.py`.

**Shared text normalization** (lowercase; strip `{{formula:…}}`/`{{cite:…}}` and
LaTeX; n-gram lookup) lives in `gap_prefilter.py`; the funnel imports it.

---

## 5. Benchmark & evaluation

- **Eval set (held out):** `data/bench/` — 10 unarXive papers (2008-era),
  per-sentence labels in `data/bench/label_sheet.tsv`
  (`silver_label` from gpt-4o; `gold_label` filled on human adjudication;
  effective label = gold else silver). 88 gap sentences total.
- **Mining set (disjoint):** `runs/{ai,ml,math}[_v1]/data/` — 2018–2024 papers.
  **No paper-ID overlap with bench** → no leakage (asserted in the miner).

Metrics:
- **Tier 0:** recall (target ≥ 0.95), specificity / filter-rate,
  survivor-rate (= load handed to Tier 1), precision (expected low), F1.
- **Current LLM (baseline):** per-type P/R/F1, macro-F1, selection &
  type-classification accuracy. Baseline run: macro-F1 **0.187**,
  selection-acc 0.91, recall starved by the ≤2/paper cap.

Re-run after adjudication:
```bash
python scripts/archive/eval_gap_extraction.py --bench-dir data/bench   # gold only
```

### 5.1 Tier 0 results (measured)

Dictionary mined from `runs/*` (189 positives / 3461 negatives, 0 bench leakage),
711-phrase pool. Validated on the bench:

| Config | recall | survivor-rate (load→Tier1) | drop |
|---|---|---|---|
| Full pool (recall-first) | **0.989** | 0.832 | ~17% |
| Hybrid (drop low-lift ∧ high-df_neg) | 0.92 | 0.79 | ~21% |
| Cue-phrases only (lift ≥ 3) | 0.50 | 0.30 | ~70% |

**Finding:** a purely lexical Tier 0 cannot hit high recall *and* strong
filtering on this data. Raising the lift floor *lowers* recall — the pool's 0.99
recall depends on generic words because **~50% of gap sentences carry no lexical
cue** (e.g. open problems phrased as questions). Lexical Tier 0 realistically
**drops ~17–21% of sentences at ≥0.92 recall** — real savings, but not 90%.

**Why precision is so low (verified, `scripts/archive/verify_tier0.py`):** inspecting
*which* phrase matched each true positive shows gaps are flagged on incidental
**stopwords** ("does/one/such"), not real cues — the genuine cue word (e.g.
"flaw") is often absent from the dictionary because only 189 mined positives
never put it above min-support. The dictionary is **cue-vocabulary-starved**, so
it leans on ubiquitous words → high recall, useless specificity. Lever: more
positives → richer cues.

**Context window (verified):** flagging sentence *i* if it OR a ±w neighbor
matches: w=1 → recall 1.000 but survivor 0.983; w=2 → survivor 0.996. A window
catches cue-less gaps (recall→1.0) but, with a noisy dictionary, floods the
filter. Distinguish **context-for-flagging** (bad trade here) from
**context-for-output** (attach ±N sentences to the payload handed to Tier 1, like
the LLM's `paragraph_text` — good practice, not yet built; does not change these
metrics).

**Measurement caveats:** labels are silver (gpt-4o, not human-adjudicated); unit
is the isolated sentence; this per-sentence recall is a different axis than the
LLM baseline's per-extracted-gap macro-F1 (0.187) — not directly comparable.

### 5.2 CRITICAL: "open gap" is a discourse-level property (`scripts/archive/check_resolved.py`)

Inspecting the text *after* suspect GT gaps shows several are **resolved in-paper**:
- "How does one prove such a statement?" → next sentence: "As it turns out, the
  task can be reduced to proving a direct product theorem…" — **resolved**.
- "We have since discovered a flaw in that argument." → "…superseded by the later
  results of Klauck et al." — **resolved/superseded**.
- Control "Can we improve the complexity of this reduction?" → followed by
  acknowledgments — **genuinely open** ✓.

**Root causes:** (1) `sections.py` matched "Limitations" inside "**Oracle
Limitations**" — a technical *results* section, not self-critique — sweeping proof
exposition into the GT (this one paper = 22/88 gaps). (2) Per-sentence labeling
cannot see resolution; a rhetorical question looks identical to an open problem in
isolation.

**Consequence for the whole approach:** a true gap = **(posed as a gap) AND (left
unresolved by the paper)**. No sentence-level classifier (silver LLM, BERT Tier 1,
lexical Tier 0) can decide the second half from the sentence alone. Gap detection
is a **document/discourse-level** task, not sentence classification. The current GT
is *systematically* contaminated (biased toward rhetorical setups), so all metrics
above inherit that bias. Fixes:
1. **Resolution-aware re-labeling:** label sentence + following context; drop ones
   the paper resolves.
2. **Position prior:** trust gaps in *terminal* Future-Work/Open-Problems sections
   (just before refs/acks) far more than "difficulties" in mid-paper discussion.
3. **Fix section extraction:** distinguish "Limitations *of this work*" (self-
   critique, near the end) from "[topic] Limitations" (a results section).

**Implications / revised plan:**
1. Keep the lexical layer, but re-scope it as (a) a free ~17% pre-reject, and
   (b) a **high-precision fast-accept**: very-high-lift cues ("future work" 7.4,
   "limitation" 6.7) ≈ certain gaps → label without a model, even pre-type them.
2. The real filter must be **semantic**: an **embedding-based Tier 0** (tiny
   encoder → cosine-to-gap-centroid or a logreg head) should catch the cue-less
   half. Still cheap/batchable/scalable. (`data/lim_embeddings.npy` suggests an
   embedding pass already exists to reuse.)
3. Caveat: bench labels are silver (gpt-4o). If silver over-labels marginal
   context sentences as gaps, true cue-based recall is understated — **human
   adjudication of `label_sheet.tsv` would tighten these numbers.**

---

### 5.3 Full-paper gold dataset (`scripts/dataset/build_gold_dataset.py`)

Per the §5.2 finding, gaps are re-extracted by passing the **whole paper** to
gpt-4o (not regex sections), so the model can judge resolution. 10 fresh papers
from `runs/*` (4 AI / 3 ML / 3 math), refs stripped, minimal schema
(`gap_type, gap_sentence, paragraph_context, resolution_status`).

Result — **27 gaps, 10 papers**:
- `resolution_status`: **21 open / 4 resolved_in_paper / 2 partially_addressed** —
  the full-paper view flags 22% as not-fully-open, the discourse signal the
  section-based silver labeler could not produce.
- `gap_type`: 15 future_work / 12 limitation / 0 open_problem.
- `location_fraction` median **0.67** — gaps cluster toward the end (terminal-
  section prior holds).

**PDF reading-order corruption (important):** strict verbatim matched only 8/27.
Investigation (`scripts/verify`…, token-recall) shows source `paper_texts.jsonl`
text is **scrambled** (two-column lines interleaved); gpt-4o silently
de-scrambles. token_recall mean **0.989** (min 0.917) → sentences are *faithful*,
just not contiguous. We therefore grade provenance **exact (8) / fuzzy (19,
≥90% words present) / weak (0)** instead of a misleading binary. **Implication:**
the current `openai_gaps.py` "verbatim" guarantee is silently compromised by the
same corruption — it just never validated. Fix upstream extraction (use `blocks`
reading order) or accept fuzzy provenance.

Artifacts: `data/bench_gold/{papers_manifest.tsv, gaps_full.jsonl, gaps_full_flat.tsv}`.
Caveat: these 10 papers were the Tier-0 mining corpus → use this set for
dataset-building / approach-validation, **not** as a clean Tier-0 eval.

## 6. How to run

```bash
# cheap funnel (no per-paper LLM): paper_texts.jsonl -> gaps.tsv (drop-in for extract-gaps)
gap2idea extract-gaps-funnel --mode hybrid --head data/gap_head.joblib

# benchmark Stage A localization + end-to-end vs gold
python scripts/bench/bench_gap_recall.py --head data/gap_head.joblib
```

*Superseded lexical-Tier-0 experiment (kept for reference only):*
```bash
python scripts/archive/mine_tier0_dictionary.py --out data/tier0_dictionary.json
python scripts/archive/eval_tier0.py --dict data/tier0_dictionary.json --bench-dir data/bench
```

---

## 7. Status

- [x] N=10 bench + per-sentence silver labels (`label_sheet.tsv`)
- [x] Baseline eval of current LLM extractor (macro-F1 0.187)
- [x] Tier 0: shared prefilter module (`gap_prefilter.py`)
- [x] Tier 0: dictionary miner + artifact (`data/tier0_dictionary.json`, 711 phrases)
- [x] Tier 0: validated on bench — **lexical drops only ~17–21% at high recall** (see §5.1)
- [x] **Funnel built + benchmarked (§8): Stage A slice + Stage B rules/embedding head**
- [x] Clean gap-sentence gold via extract→verify (`data/bench_gap/gold_sentences.tsv`)
- [x] Runnable pipeline: `gap2idea extract-gaps-funnel` (drop-in `gaps.tsv`)
- [x] Model-vs-data settled: bge+logreg ≈ DistilBERT ≈ SetFit (all limitation R≈0.11) → data-bound
- [x] **Stage B limitations fixed** via ACL mandated-Limitations harvest: limitation
      recall 0.11 → 0.44, end-to-end 0.42 → 0.53 (§8.4); shipped as default head
- [x] No off-the-shelf model exists for the task (direct HF Hub crawl, 0 hits)
- [x] **Same-data comparison vs prior art** (LimGen ACL, leakage-clean): our cheap frozen
      stacking ensemble F1 **0.627** beats reproduced Zhang BernoulliNB (0.553), TF-IDF
      (0.610), base bge (0.610); ~0.05 behind fine-tuned DistilBERT (0.674). See
      `experiments/research_comparison.md`. **Stage B finished (frozen path).**
- [ ] Close the ~0.05 gap: LLM filter (Stage C) / fused stack / GPU fine-tune (need GPU or credits)
- [ ] Optional: lift Stage-A recall on mid-paper limitations; bigger gold than 19 gaps

---

## 8. The cheap funnel for future_work + limitation (built & benchmarked)

Scope narrowed to the two **structurally localized** gap types (future_work,
limitation); open_problem is left to the LLM (it is the discourse-level case,
§5.2). Code: `src/gap2idea/pipeline/gap_funnel.py`. Run: `gap2idea
extract-gaps-funnel [--mode rules|model|hybrid] [--head data/gap_head.joblib]`.

### 8.1 Stages
- **Stage A — structural slice** (`slice_terminal_regions`): on the reading-order
  sentence stream, union three sources — generous span after a Limitations/
  Future-Work/Conclusion *heading*, ±window around an *inline* keyword (catches
  two-column-scrambled headings), and a *terminal tail* (unheaded conclusions).
  Load cap by anchor priority; mid-paper Discussion gated out. FREE (regex/CPU).
- **Stage B — classify** (`cue_label` + `EmbeddingGapHead`): high-precision cue
  rules give a free fast-accept + type; a logreg head on bge-small-en-v1.5
  embeddings catches cue-less gaps, but only inside *explicit* Limitations/
  Future-Work regions (tail/discussion → rules only, or it floods FPs). Self-
  distilled from teacher labels; negatives drawn from body OUTSIDE the slice.

### 8.2 Benchmark (clean, minimal, leakage-guarded)
- **Gold** (`scripts/dataset/build_gap_gold.py`): full paper → gpt-4o extract → gpt-4o
  **verify** filter (drops prior-work limitations, contributions/cross-refs
  mislabelled as gaps, vague gestures). **19 gap sentences / 9 papers**
  (`data/bench_gap/gold_sentences.tsv`). The verify pass was essential: a raw
  single-pass gold was ~25% contaminated, concentrated in math papers.
- **Leakage**: the head's training papers are asserted disjoint from the 10 gold
  papers (`data/gap_head.meta.json`; checked in `bench_gap_recall.py`).
- **Metric**: token-CONTAINMENT (not substring) — 19/27 gold gaps are PDF-
  scrambled, so localization is reported at τ ∈ {0.90, 0.80, 0.70}.
- Run: `python scripts/bench/bench_gap_recall.py --head data/gap_head.joblib`.

### 8.3 Results
**Stage A — localization recall (the ceiling) + load**

| containment τ | recall all | future_work | limitation |
|---|---|---|---|
| 0.90 | 0.74 | 0.82 | 0.63 |
| **0.80** | **0.84** | **0.91** | 0.75 |
| 0.70 | 0.89 | 1.00 | 0.75 |

Drops **82%** of sentences for free. future_work localization is effectively
solved; the 2 limitation misses are *mid-paper* own-work limitations stated in
setup (not a Limitations section) — a structural, not a slicer, limit.

**End-to-end vs gold** (10 papers, ~$/1M papers ≈ a few dollars vs ~$4000 LLM):

| Stage B head | preds/paper | gap recall | limitation recall | type acc |
|---|---|---|---|---|
| rules only | 2.2 | 0.32 | 0.11 | 1.00 |
| hybrid (bge+logreg) | 4.2 | 0.42 | 0.11 | 1.00 |
| **hybrid + ACL limitations** *(shipped)* | **6.1** | **0.53** | **0.44** | **0.90** |

### 8.4 Stage B: the limitation problem, and the fix that worked
future_work works well (cue-rich + terminal). **Limitations were the weak spot**
and took several iterations to diagnose honestly:

1. **Bug (fixed):** self-distillation used the teacher's ≤2-gaps/paper output as
   the *only* positives, so real gaps it skipped became negatives — the head
   learned "gaps are none" (predicted `none` at p=0.91 on real limitations).
   Fix: draw negatives from the body *outside* the slice.
2. **It's data, not the model (proven on THREE axes):** classifier method
   (logreg ≈ DistilBERT ≈ SetFit, all limitation R ≈ 0.11;
   `scripts/training/test_bert_stageb.py`, `test_setfit_stageb.py`), and — after the ACL
   fix — the frozen ENCODER too: bge-small ≈ bge-base ≈ mpnet ≈ SPECTER all land
   at recall 0.526 / limitation 0.444 (`scripts/training/sweep_encoders.py`). Upgrading
   the model or encoder is a dead end; the literature agrees (even domain-BERT
   caps ~0.5 on limitation typing). We keep **bge-small** (smallest → most
   scalable, best future-work recall, ties on the rest).
3. **Distant supervision (rejected):** weak labels from *every* explicit-section
   sentence raised recall but flooded predictions (143/10 papers — formulas,
   acknowledgments). Reverted.
4. **The fix (shipped):** harvest the **mandated "Limitations" sections of ACL
   papers** (LimGen, CC-BY-4.0) as clean limitation positives —
   `scripts/dataset/harvest_acl_limitations.py` → 6,433 sentences, leakage-filtered vs
   gold. A cap sweep (`scripts/training/sweep_acl_cap.py`) shows limitation recall
   **0.11 → 0.44 (4×)** and end-to-end **0.42 → 0.53**, saturating at ~1,500
   sentences, *without* the distant-supervision flood (61 preds, not 143). This
   is the literature-standard recipe (mandated-section harvesting); credits were
   never needed. Cue-harvested *future-work* positives were net-negative (noisy)
   and excluded by the sweep.

**Residual:** end-to-end recall is now bounded by Stage A's localization ceiling
(~16/19 in slice), not Stage B. Further gains need better Stage-A recall on
*mid-paper* limitations or a larger/cleaner gold than 19 gaps.
```
