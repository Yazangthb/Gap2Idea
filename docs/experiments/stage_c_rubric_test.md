# Stage C rubric strictness on the gold test split (2026-11-05)

Short NLI at τ 0.354, 2,522 candidates from 34 test papers (664 gaps), Qwen3.8-27B, context, batch 20, no thinking, seed 0. LLM gold. No failed batches.
Rubric wording and τ were fixed before this run; see `stage_c_rubric_val.md`. All three rubrics were run and are reported as fixed settings.
Script: `scripts/bench/stage_c_rubric_test.py`. Outputs (gitignored): `stage_c_rubric_test.json`, `stage_c_rubric_test_*.jsonl`.

| Rubric | Keeps gaps | Removes non-gaps | Kept | Precision | Recall | F1 | F1 excl. 196 selection sentences |
|---|---|---|---|---|---|---|---|
| Keep-biased | 93.9% | 76.9% | 1,011 | 56.3% | 85.7% | 0.679 | 0.664 |
| Neutral | 87.3% | 87.1% | 776 | 68.2% | 79.7% | **0.735** | 0.723 |
| Strict | 62.9% | 96.9% | 440 | 86.6% | 57.4% | 0.690 | 0.675 |

| Recall by type | Limitation (220) | Future work (97) | Research gap (347) |
|---|---|---|---|
| Keep-biased | 79.5% | 86.6% | 89.3% |
| Neutral | 71.4% | 84.5% | 83.6% |
| Strict | 55.9% | 82.5% | 51.3% |

| P / R / F1 by domain | arXiv (28 papers) | Biomedical (6 papers, subset of SAL-40) |
|---|---|---|
| Keep-biased | .517 / .846 / .642 | .826 / .898 / .860 |
| Neutral | .642 / .784 / .706 | .872 / .847 / .859 |
| Strict | .834 / .533 / .650 | .971 / .730 / .833 |

## Paired F1 differences, papers resampled (2,000 draws)

| Comparison | Difference | 95% CI |
|---|---|---|
| Neutral vs keep-biased | **+0.055** | **+0.025 to +0.080** |
| Strict vs keep-biased | +0.011 | −0.054 to +0.060 |
| Strict vs neutral | **−0.045** | **−0.095 to −0.006** |
| Neutral vs bge-small + Qwen (0.665) | **+0.070** | **+0.033 to +0.111** |
| Neutral vs ModernBERT alone (0.614) | **+0.121** | **+0.083 to +0.162** |
| Keep-biased vs bge-small + Qwen | +0.015 | −0.027 to +0.066 |
| Keep-biased vs ModernBERT alone | **+0.065** | **+0.017 to +0.126** |
| Strict vs bge-small + Qwen | +0.025 | −0.037 to +0.087 |
| Strict vs ModernBERT alone | **+0.076** | **+0.020 to +0.132** |

## Without `research_gap`, P / R / F1

| Keep-biased | Neutral | Strict |
|---|---|---|
| .369 / .817 / .509 | .492 / .754 / .595 | .775 / .640 / .701 |

## Run-to-run stability

Keep-biased was judged again from scratch with the same inputs and seed: 32 of 2,522 verdicts changed (1.3%), 5 of them on gaps. F1 was 0.678 in the first run and 0.679 in this one. This is the first stability measure for Qwen on the shared GPU.

## Limits

- LLM gold only. arXiv precision may be understated if long papers are under-labelled.
- Neutral and strict wording was informed by reading false positives from the earlier test run, so test is not untouched for these rubrics. Excluding the 196 selection sentences does not remove this influence.
- Neutral also ranked first on validation, in both halves (`stage_c_rubric_val.md`), but only by 0.012 overall.
- SAL-40 results are in `docs/publication/STAGE_C_MODES_RESULTS.md`. These rubrics have not been run on FWS.
