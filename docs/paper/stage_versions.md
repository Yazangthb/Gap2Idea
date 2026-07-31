# Stage versions registry (paper-archival)

> Each version is frozen with measured numbers. New versions are added with
> suffixes (`_v2`, `_v3`); old versions are NEVER modified. This lets us roll
> back, A/B test, and cite specific versions in the paper.

---

## Stage A — structural slicer

### v1 (FROZEN) — `gap_funnel.slice_terminal_regions`
- **Status:** Production-shipped, paper-baseline.
- **Method:** Terminal-section + inline-keyword + tail. Union of three sources, position-gated for generic Discussion anchors.
- **Measured (LimGen):** Stage A drop 82%, gold v1 localization 0.842 @ τ=0.80.
- **Measured (our gold v1, 19 gaps):** localization 0.842; recall ceiling 0.842 (only 3/19 missed at slice).
- **Measured (our gold v2, 49 gaps):** localization 0.510 @ τ=0.70 — recall ceiling for mid-paper gaps.
- **Trade-off:** efficient (82% drop) but misses mid-paper scope/assumption sentences.
- **Code:** `src/gap2idea/pipeline/gap_funnel.py::slice_terminal_regions`
- **Do not modify.** Any improvements live in a separate function.

### v2 (DEV) — `gap_funnel.slice_with_midpaper_anchors`
- **Status:** implemented, A/B-tested on gold v2.
- **Method:** v1 output + additional mid-paper anchors that catch scope/assumption sentences anywhere in the paper. ±2-sentence windows around matches of `we focus|paper focuses on|we restrict|we assume|throughout this paper|focus is to|simplify our treatment|in the interest of simplicity`.
- **Anchor span:** ±2 sentences (narrow, to avoid bloating the slice).
- **Measured (gold v2, 49 gaps, A/B vs v1):**

  | metric | v1 | v2 | Δ |
  |---|---|---|---|
  | drop rate | 81.7% | 79.5% | −2.2 pts (slice +12%) |
  | regions/paper | 4.3 | 9.0 | +4.7 |
  | loc @ τ=0.70 | 0.510 | **0.612** | +0.102 |
  | loc @ τ=0.80 | 0.449 | **0.531** | +0.082 |
  | loc @ τ=0.90 | 0.347 | 0.408 | +0.061 |

- **Verdict:** +0.10 localization recall on comprehensive gold for +12% slice size. Cost-justified.
- **Code:** `src/gap2idea/pipeline/gap_funnel.py::slice_with_midpaper_anchors` ✅
- **A/B test:** `scripts/ab_test_stage_a.py`

---

## Stage B — classifier head

### v1 (FROZEN) — `gap_head.joblib` (bge-small + logreg, ACL-augmented)
- **Status:** earlier shipped head.
- **Measured (LimGen full test, 13,319 sents):** F1 0.61.
- **Code:** training in `scripts/train_gap_head.py`.

### v2 (FROZEN, current ship) — SciBERT-FT (2-epoch LimGen + ACL)
- **Status:** **paper-baseline**, the F1=0.743 number.
- **Method:** `allenai/scibert_scivocab_uncased` fine-tuned, 2 epochs, lr 3e-5, bs 24.
- **Measured (LimGen full test, 13,319 sents):** **F1 0.743, P 0.809, R 0.687**.
- **Measured (gold v1, 19 gaps):** recall 0.526 (10/19), precision_floor 0.245.
- **Measured (gold v2, 49 gaps):** recall 0.20 (10/49) — gold v2 has many mid-paper gaps Stage A v1 misses.
- **Code:** `scripts/finetune_and_chain.py::train_bert` and `scripts/test_scibert_gold.py`.
- **Do not retrain unless documented as a new version.**

---

## Stage C — LLM precision filter

### v1 (FROZEN) — `gap_llm_filter.LLMGapFilter(mode='validate')` original
- **Status:** earlier shipped, validation mode.
- **Measured (LimGen):** −0.034 F1 with 14B.

### v2 (FROZEN) — `mode='validate_v5'` (default-accept categorical)
- **Status:** improved permissive validator.
- **Measured (LimGen):** −0.010 F1 with 3B, neutral.

### v3 (FROZEN) — `mode='validate_v7'` (LimGen-aware surgical)
- **Status:** explicit FP-category targeting.
- **Measured (LimGen):** −0.010 F1 with 3B, −0.012 with 7B.

### v4 (FROZEN, current Stage C) — `GAP/JUNK + ±30-word context`
- **Status:** **paper-baseline**, the gold-positive result.
- **Method:** binary GAP/JUNK classification with surrounding context, gpt-4o batched.
- **Measured (LimGen):** −0.20 F1 (correctly rejects section-membership false positives).
- **Measured (gold v1, 10 papers):** **+0.076 F1** (0.334 → 0.410), 49→25 preds.
- **Code:** `scripts/test_gap_junk_context.py` and `scripts/eval_all_stages_gold.py`.

---

## Evaluation gold sets

### gold v1 (FROZEN) — `data/bench_gap/gold_sentences.tsv`
- **Status:** initial verified gold.
- **Method:** gpt-4o extract → gpt-4o verify (single-pass), token_recall ≥ 0.80.
- **Size:** 19 gaps / 9 papers (1 paper has 0 gold gaps).
- **Audit:** 17/19 confirmed real by gpt-4o adjudication (2 borderline scope).
- **Coverage:** ~50% (audit found 17 more real gaps Stage B emitted that gold missed).

### gold v2 (current) — `data/bench_gap/gold_sentences_v2.tsv`
- **Status:** comprehensive re-extraction.
- **Method:** gpt-4o exhaustive extract (Pass 1) → gpt-4o categorical adjudicate with ±30-word context (Pass 2) → dedup at τ=0.8.
- **Size:** 49 gaps / 10 papers.
- **Types:** scope=14, future_work=14, limitation=11, assumption=9, open_problem=1.
- **Overlap with v1:** 14/19 sentences carry over.

---

## How to use this registry

- **Paper claims** must cite the specific version: *"Stage A v1 achieves 82% drop"*, *"Stage B v2 (SciBERT-FT) achieves F1 0.743 on LimGen"*, *"Stage C v4 (GAP/JUNK+context) improves gold v1 F1 by +0.076"*.
- **Benchmarks** should run all relevant versions side-by-side.
- **Optimizations** should add a new version (`_v3`, etc.) and update this registry — never modify a frozen version.
