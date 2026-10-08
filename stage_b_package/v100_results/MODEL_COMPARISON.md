# Stage B — full comparison (our results vs. research papers)

Two kinds of result are kept **separate** because they are not on the same test data:
- **Table 1** — public-benchmark head-to-heads vs. the papers (repo's logged runs; *not* re-verified this session).
- **Table 2** — this session's Stage-B development, on our **internal 10-paper gold** (49 gaps). No external paper uses this set, so it is not a benchmark-vs-paper comparison.

Metrics differ per benchmark (macro-F1 / positive-F1 / recall / coverage) and test conditions differ — do not compare numbers across rows.

---

## Table 1 — Public benchmarks vs. published SOTA

| Benchmark | Task / metric | Published SOTA (paper) | Our / repo result | Comparable? |
|---|---|---|---|---|
| **FWS** (Zhang 2022, [2212.13860]) | future-work sentence recognition, macro-F1 | **0.907** reported (balanced, full-data feature selection) — repo showed this is **leakage-inflated**; honest in-fold **0.825** | repo balanced split: bge **0.879**, PubMedBERT 0.877, SciBERT 0.864; paper-grouped realistic: bge 0.850 | same corpus/split family; **logged, not re-run** |
| **SAL / RCT** (Lan 2024, [JBI 10.1016/j.jbi.2024.104628]) | limitation-sentence detection, F1 on section pool (~20.7% prev) | **~0.81** (PubMedBERT, authors' unreleased model) | repo detector **0.762 → 0.812** with Stage-C LLM filter (gpt-oss / Yandex) | reconstructed pool, not authors' checkpoint; **logged** |
| **LimGen** (Faizullah 2024, [2403.15529]) | originally *generation*; repo repurposed to binary limitation detection, F1 | no official detection SOTA; reproduced prior art DistilBERT **0.629** | repo SciBERT-FT **0.743** (binary) | repurposed task; **logged, checkpoint missing** |
| **BAGELS** (Al Azhar 2025, [2505.18207]) | limitation coverage (token-containment) | generation benchmark; no detection-precision SOTA | repo coverage **0.738** (A+B) → 0.434 after C; positive-only (no precision) | coverage metric only; **logged** |
| **Sci-Sentence** (Bolaños-Burgos 2025, [2508.04337]) | 7-class rhetorical sentence classification, F1 on their test | **>0.96** F1 (fine-tuned, e.g. GPT-4o / strong encoders) | our PubMedBERT FT: sci val macro-F1 **0.922** (our split) | **NOT comparable** — our split ≠ their test |

---

## Table 2 — Our Stage-B development (this session), internal gold (10 arXiv papers, 49 gaps)

Head = sentence encoder + class-weighted logistic regression (frozen) or full fine-tune (FT).
Train = Sci-Sentence + FWS + SAL. Metric = end-to-end recall (rules+head): **recall_all / limitation / future_work / kept**.

| Rank | model | recall_all | limitation | future_work | kept |
|---|---|---|---|---|---|
| 1 | **MiniLM-L6 + logreg** (frozen) | **0.571** | **0.636** | **0.857** | 170 |
| 1 | mpnet + logreg (frozen) | 0.571 | 0.455 | 0.786 | 186 |
| 3 | bge-small + logreg + SAL (frozen) | 0.429 | 0.545 | 0.786 | 97 |
| 4 | bge-base / e5-base + logreg (frozen) | 0.408 | 0.455 | 0.786 | 71–76 |
| 5 | SciBERT (full FT) | 0.327 | 0.273 | 0.714 | 61 |
| 5 | PubMedBERT (full FT) | 0.327 | 0.364 | 0.714 | 65 |
| 7 | DistilBERT (full FT) | 0.306 | 0.182 | 0.714 | 63 |

Best in-distribution validation (macro-F1, our split): PubMedBERT FT **0.890** — but it is near-bottom on gold (0.327). Validation is anti-correlated with gold here.

### Balancing check (MiniLM frozen)
A class-weighted (baseline) **0.571** · B no class-weight 0.265 · C explicit resample 0.551 · D + aug paraphrases 0.571. → class-weighting already balances; extra balancing/augmentation gives no gold gain.

---

## Honest caveats
1. **Table 1 numbers are logged, not reproduced here.** The audit flagged the repo's "SOTA" logs as needing fresh verification; several checkpoints/predictions are missing.
2. **Table 2 is on a 10-paper internal gold** — small sample, our own annotation (GPT-4o silver). Good for ranking methods, not a published-benchmark claim, and not comparable to Table 1.
3. The FWS 0.907 "SOTA" is **leakage-inflated** (feature selection on full data); the honest figure is ~0.825 — important when citing it as a baseline to beat.
4. Full fine-tunes top validation but overfit to gold; frozen embeddings generalize best on our task.

## Not yet run
Regularized fine-tune (LoRA / freeze-encoder+head), SPECTER2 encoder, and any *fresh* re-run of the Table-1 benchmarks under controlled splits.
