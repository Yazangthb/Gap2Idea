# NLI post-Qwen thresholds on the complete test split

Compare Qwen3.8-27B FP8 on the same 34 test papers and same up-to-17 groups (17, 10, 5, 2 papers). Full-paper Qwen and the fine-tuned NLI funnel use their saved original responses; the latter uses surrounding previous/next sentences. We apply the two user-selected thresholds to the saved fine-tuned NLI scores **after** Qwen: a Qwen-positive sentence remains only if its score meets the threshold. Neither Qwen's inputs nor its decisions change. No model inference or new GPU calls were run.

| Method | Sentences kept | TP | FP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Full-paper Qwen | 560 | 478 | 82 | 85.36% | 71.99% | 78.10% |
| Fine-tuned funnel, no additional post-filter | 643 | 510 | 133 | 79.32% | 76.81% | 78.04% |
| Funnel, post-Qwen threshold 0.02 | 586 | 484 | 102 | 82.59% | 72.89% | 77.44% |
| Funnel, post-Qwen threshold 0.1 | 542 | 461 | 81 | 85.06% | 69.43% | 76.45% |

The recall denominator is all 664 LLM-labeled gaps in the 34 papers, including gaps removed by Stage A or NLI before Qwen. Relative to the unfiltered funnel, threshold 0.02 removes 57 Qwen outputs: 26 labeled positives and 31 labeled negatives. Threshold 0.1 removes 101: 49 labeled positives and 52 labeled negatives. Compared with full-paper Qwen, 0.02 finds 6 more labeled positives but returns 20 more labeled negatives; 0.1 finds 17 fewer labeled positives and returns one fewer labeled negative.

## Domain breakdown

| Method | 28 arXiv papers P / R / F1 | 6 biomedical papers P / R / F1 |
|---|---|---|
| Full-paper Qwen | 82.28 / 71.35 / 76.42 | 99.03 / 74.45 / 85.00 |
| Fine-tuned funnel, no post-filter | 75.56 / 77.42 / 76.48 | 99.03 / 74.45 / 85.00 |
| Funnel, 0.02 | 79.22 / 73.06 / 76.01 | 99.00 / 72.26 / 83.54 |
| Funnel, 0.1 | 82.02 / 69.26 / 75.10 | 98.97 / 70.07 / 82.05 |

The biomedical subset has only six papers, and these are LLM-generated three-type labels rather than the separate SAL human limitation labels. The test split has been examined in earlier experiments; the thresholds were chosen after development exploration. This is exploratory and does not establish independent or statistically significant superiority. Labels have not been fully validated by humans.

## Qwen token usage

| Method | Input tokens | Output tokens | Total tokens |
|---|---:|---:|---:|
| Full-paper Qwen | 612,965 | 134,727 | 747,692 |
| Fine-tuned NLI funnel → Qwen | 259,776 | 27,734 | 287,510 |
| Funnel reduction | 57.6% | 79.4% | 61.5% |

Both post-Qwen thresholds use exactly the same saved funnel calls, so their Qwen input and output token counts are identical to the unfiltered funnel. Post-filtering does not reduce tokens already used. These are Qwen token counts, not end-to-end cost: they exclude NLI, preprocessing and any hardware expenses. The NLI score is the maximum typed entailment/(entailment+contradiction) score, not a calibrated probability. Exact outputs, thresholds, hashes, domain counts, and code appear in `results.json`, `plan.json`, the original run records, and `scripts/bench/nli_post_qwen_test.py`.
