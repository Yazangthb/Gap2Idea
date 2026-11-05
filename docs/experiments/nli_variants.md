# NLI recall/precision variants on gold validation (2026-11-05)

`tasksource/ModernBERT-large-nli` (revision `ca476cb9`) on all 34 gold validation papers, 11,201 sentences, 699 gaps.
No test paper was loaded. The server received sentence text only; scoring ran locally against LLM gold, any gap type.
Papers were split by domain into half A (17 papers, 351 gaps; variants, thresholds) and half B (17 papers, 348 gaps; confirmation).
The junk filter applies to every variant. 14 hypotheses × 2 premises, 420 s on one GPU.

Scripts: `scripts/bench/nli_score_remote.py`, `scripts/bench/nli_variants.py`. Variant table: `nli_variants.json` (gitignored).

## Half A: what moves precision and recall

| Change | Effect |
|---|---|
| NLI vs bge-small head | Precision at recall ≥70%: 0.49–0.51 vs 0.30; at ≥80%: 0.34–0.39 vs 0.22; at ≥90%: 0.20–0.26 vs 0.18 |
| Adding ±1 context to the premise | Worse in every case (best F1 about 0.45–0.47 vs 0.60) |
| Subtracting competing "not a gap" hypotheses | Worse, especially at high recall (P 0.08–0.10 at recall ≥90%) |
| Rank-averaging with the bge-small head | No gain |
| Entailment vs contradiction score | Usually better than plain entailment, mainly for `max` |

Differences among the top single-sentence variants are small relative to sample noise.

## Half B: shortlist, thresholds fixed on half A

| Variant, target | Kept | Precision | Recall | F1 | Limitation | Future work | Research gap |
|---|---|---|---|---|---|---|---|
| bge-small head, R≥80 | 978 | 28.2% | 79.3% | 0.416 | 68.5% | 81.5% | 85.5% |
| bge-small head, R≥90 | 1,422 | 21.4% | 87.6% | 0.345 | 83.8% | 87.7% | 90.1% |
| Original hypothesis, best F1 | 377 | 54.6% | 59.2% | 0.568 | 36.0% | 49.2% | 77.9% |
| Original hypothesis, R≥80 | 735 | 37.4% | 79.0% | 0.508 | 62.2% | 75.4% | 91.3% |
| Original hypothesis, R≥90 | 1,085 | 27.5% | 85.6% | 0.416 | 74.8% | 83.1% | 93.6% |
| All typed, mean, R≥80 | 659 | 39.5% | 74.7% | 0.516 | 45.9% | 90.8% | 87.2% |
| **Short typed, R≥80** | **704** | **39.2%** | **79.3%** | **0.525** | **69.4%** | **69.2%** | **89.5%** |
| **Short typed, R≥90** | **1,096** | **27.7%** | **87.4%** | **0.421** | **83.8%** | **78.5%** | **93.0%** |

Short typed = `typed_c_max_evc`: "This is a limitation." / "This is future work." / "This is a research gap.", maximum over the three,
entailment vs contradiction. Thresholds 0.669 (R≥80) and 0.354 (R≥90).

## Reading
- At matched recall, short typed NLI flags 28–23% fewer sentences than the bge-small head with higher precision.
- It matches the head on limitation recall, gains on research gaps, and loses on future work.
- All recall targets undershot on half B: 90% on half A became 84–87%.
- Half B has now been used to compare the shortlist, so it is no longer clean. The test papers remain untouched.
