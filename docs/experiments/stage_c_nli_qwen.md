# NLI as Stage B, then Qwen Stage C, on the gold test split (2026-11-05)

Setup frozen on validation before any test scoring (`nli_variants.md`): ModernBERT-large-nli, revision `ca476cb9`, hypotheses
"This is a limitation." / "This is future work." / "This is a research gap.", maximum of entailment-vs-contradiction,
junk filter, τ 0.669 and τ 0.354. Stage C: Qwen3.8-27B, same keep-biased rubric, context, batch 20, no thinking, seed 0.
34 test papers, 11,741 sentences, 664 gaps, LLM gold. No failed batches.

Script: `scripts/bench/stage_c_nli_qwen.py`. Outputs (gitignored): `stage_c_nli_qwen.json`, `stage_c_nli_r80_verdicts.jsonl`,
`stage_c_nli_r90_verdicts.jsonl`, `data/gold/nli_test_scores.npz`.

Reproduction check: the original ModernBERT hypothesis at τ 0.75 gives P .626, R .602, F1 **.614** against the recovered .617.

## Overall

| Setup | Candidates | Requests | Final kept | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| ModernBERT original, alone | n/a | 0 | 639 | 62.6% | 60.2% | 0.614 |
| bge-small head + Qwen | 1,438 | 72 | 711 | 64.3% | 68.8% | 0.665 |
| Short NLI τ 0.669 + Qwen | 1,633 | 82 | 922 | 57.3% | 79.5% | 0.666 |
| **Short NLI τ 0.354 + Qwen** | **2,522** | **127** | **1,029** | **55.8%** | **86.4%** | **0.678** |

Before Stage C: τ 0.669 had P 33.3%, R 81.9%; τ 0.354 had P 24.0%, R 91.3%. Stage C kept 97.1% and 94.7% of gaps, and removed
63.8% and 76.3% of negatives.

## Paired F1 differences, papers resampled (2,000 draws)

| Comparison | Difference | 95% CI |
|---|---|---|
| τ 0.354 + Qwen vs bge-small + Qwen | +0.013 | −0.034 to +0.069 |
| τ 0.669 + Qwen vs bge-small + Qwen | +0.001 | −0.035 to +0.041 |
| τ 0.354 + Qwen vs ModernBERT alone | **+0.064** | **+0.011 to +0.128** |
| τ 0.669 + Qwen vs ModernBERT alone | **+0.052** | **+0.018 to +0.094** |
| τ 0.669 vs τ 0.354, both + Qwen | −0.012 | −0.045 to +0.018 |

## Recall by gap type

| Type (n) | ModernBERT alone | bge-small + Qwen | τ 0.669 + Qwen | τ 0.354 + Qwen |
|---|---|---|---|---|
| Limitation (220) | 46.4% | 61.4% | 70.9% | **80.5%** |
| Future work (97) | 43.3% | 64.9% | 70.1% | **86.6%** |
| Research gap (347) | 73.8% | 74.6% | 87.6% | **90.2%** |

## By domain, P / R / F1

| Domain | bge-small + Qwen | τ 0.669 + Qwen | τ 0.354 + Qwen |
|---|---|---|---|
| arXiv (28 papers) | .601 / .687 / .641 | .532 / .793 / .637 | .513 / .854 / .641 |
| Biomedical (6 papers, subset of SAL-40) | .872 / .693 / .772 | .803 / .803 / .803 | .821 / .905 / .861 |

## Per stage, τ 0.354

| Stage | Sentences kept | Gaps kept | Precision | Recall | F1 |
|---|---|---|---|---|---|
| All test sentences | 11,741 | 664 | 5.7% | 100% | 0.107 |
| A: junk filter | 9,604 | 644 | 6.7% | 97.0% | 0.125 |
| B: NLI | 2,522 | 606 | 24.0% | 91.3% | 0.380 |
| C: Qwen | 1,029 | 574 | 55.8% | 86.4% | 0.678 |

## False positives (`scripts/bench/stage_c_nli_fp_analysis.py`)

- 428 of 455 are arXiv; 411 of 455 were predicted as limitation.
- 5 of 34 papers hold 59% of them (269). These papers are long (32% of test sentences) and sparsely labelled: 4.2 gaps per 100 sentences against a median of 7.6. Without them P is 70.1% and R 86.7%. Median per-paper precision is 76.5%. This is a diagnostic, not a result.
- Only 7 of 455 were flagged by either labelling judge.
- Gold gaps stay near 18–24 per paper across lengths of 220 to 824 sentences among the worst papers, which may mean under-labelling in long papers. Labels do reach the end of the text, so truncation looks unlikely.
- One non-human reading of 40 random arXiv false positives: about 15 look like real gaps, 13 borderline, 12 clearly not. No precision claim should rest on this.

## Limits
- F1 against bge-small + Qwen is not distinguishable; the gain is recall, at a precision cost.
- Excluding the 196 candidates used to choose the Stage C setup: τ 0.354 P 54.2%, R 85.5%, F1 0.663; τ 0.669 P 55.9%, R 78.7%, F1 0.653.
- Without `research_gap`, τ 0.354 scores P 36.5%, R 82.3%, F1 about 0.505, against 0.515 for bge-small + Qwen.
- LLM gold only. Thresholds were fixed on validation, but the rubric was chosen using test-split samples.
