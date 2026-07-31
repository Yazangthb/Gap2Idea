# Paper notes — Component 1: Gap Extraction (DONE; revisit for write-up)

> Paper-ready notes for the extraction module of Gap2Idea (IEEE Access, full
> system). This component is **built, benchmarked, and documented** — see
> `docs/experiments/` for everything. Below is what goes in the paper + the
> honest claim discipline. **Come back here when writing §Extraction.**

## Role in the system
Turns raw papers → typed research-gap sentences (`future_work`, `limitation`)
that feed clustering → idea generation. The design goal is **scale**: classify
sentences-per-dollar offline, not per-paper LLM.

## Method (3 stages) — for §Extraction
- **Stage A — structural slice** (`gap_funnel.slice_terminal_regions`): on the
  reading-order sentence stream, union three recall sources — heading-anchored
  span, inline-keyword window (handles two-column-scrambled PDFs), terminal tail.
  Position-gated, priority-capped. FREE (regex/CPU). Drops ~82% of sentences.
- **Stage B — classify** (`cue_label` + `EmbeddingGapHead`): high-precision cue
  rules + a frozen bge-small + logreg head (self-distilled; negatives from body
  OUTSIDE the slice). Best frozen result via a **stacking ensemble [bge+tfidf+cue]**.
- **Stage C — LLM precision filter** (`gap_llm_filter`): a small local instruct
  model (Qwen2.5-1.5B, API-swappable) judges only the ~6 survivors/paper and
  drops false positives (acknowledgments, formulas, citations, math exposition).

## Key results (HONEST — use these numbers)
- **Stage A localization recall** 0.84 (future_work 0.91), drops 82% free.
- **Cost:** ~$3–31 / 1M papers vs ~$4,000 per-paper LLM (~128–166× cheaper).
- **Data, not model (clean negative result):** logreg ≈ DistilBERT ≈ SetFit ≈
  bge/bge-base/mpnet/SPECTER all tie; only **clean data** moved it. The
  limitation-data fix = harvesting mandated ACL "Limitations" sections (LimGen,
  CC-BY-4.0) → limitation recall 0.11 → 0.44.
- **Vs prior art on LimGen (same data, leakage-clean):** our frozen stacking
  ensemble **F1 0.627** beats reproduced Zhang BernoulliNB (0.553), TF-IDF
  (0.610), bge (0.610); fine-tuned DistilBERT 0.674.
- **Stage C lift (LimGen sample):** Stage B 0.643 → **+Stage C 0.725** (precision
  0.56→0.79), i.e. competitive with / above the fine-tuned transformer at a
  fraction of the cost.

## Claim discipline (what to write / NOT write)
- ✅ "A cost-efficient funnel that **matches a fine-tuned transformer's range at
  a fraction of the cost**, with an LLM applied only to ~6 sentences/paper."
- ✅ "Extraction quality is **data-bound, not model-bound** (ablation)."
- ❌ Do NOT claim "beats fine-tuned BERT / SOTA" — the Stage-C win is on a
  **196-sentence sample** (±noise, different test size). State it as *indicative*.
- ❌ Do NOT present the 19-gap gold as a definitive benchmark — it is small,
  silver (LLM-built). Frame as a *clean-but-small* internal eval.

## Figures ready
`docs/figures/stage_a_flow.mmd`, `stage_a_union.svg`, `stage_a_localization.png`,
`stage_a_funnel.png`. (Architecture/figure for the funnel + the cost funnel.)

## Gaps to close BEFORE final submission (revisit)
1. **Bigger / human-checked extraction gold** (19 → a few hundred, or human-spot-
   checked) — strengthens the extraction eval.
2. **Full-scale LimGen comparison** (thousands of sentences, significance) instead
   of the 196-sentence Stage-C sample — needs a GPU/stable box (CPU fine-tune +
   long LLM runs were unreliable here).
3. Optional: SSC / context-aware classifier; Stage-A mid-paper limitation recall.

## Pointers
Experiments: `docs/experiments/{experiment_log,results_registry,research_comparison,
stage_c_output,stage_c_limgen}.md`. Code: `src/gap2idea/pipeline/{gap_funnel,
gap_llm_filter}.py`. Reproduce: `docs/experiments/README.md`.
