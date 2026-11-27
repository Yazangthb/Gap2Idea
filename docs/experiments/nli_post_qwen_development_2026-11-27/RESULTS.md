# Fine-tuned NLI score thresholds applied after Qwen: development set

This analysis uses the saved Qwen-positive sentences and precomputed fine-tuned NLI scores for all 34 development papers; no models were run and Qwen inputs, predictions and token use are unchanged. A post-filter retains a Qwen-positive sentence only when its saved NLI score is at least the threshold. The score is the maximum across three type-specific entailment / (entailment + contradiction) scores; it should not be interpreted as a calibrated gap probability. The original Stage A + NLI threshold remains in place before Qwen. The comparison starts at its original threshold, approximately 0.00210337, which retains all Qwen positives.

Primary setup: the existing fine-tuned funnel with surrounding neighbors and up-to-17 paper groups (17, 10, 5, 2). Precision, recall and F1 are sentence-level; recall uses all 699 LLM-labeled development gaps, including sentences lost before Qwen.

| Post-Qwen NLI threshold | Sentences kept | TP | FP | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00210337 (no additional filtering) | 706 | 534 | 172 | 75.64% | 76.39% | 76.01% |
| 0.003 | 695 | 528 | 167 | 75.97% | 75.54% | 75.75% |
| 0.005 | 677 | 521 | 156 | 76.96% | 74.54% | 75.73% |
| 0.01 | 652 | 509 | 143 | 78.07% | 72.82% | 75.35% |
| 0.02 | 620 | 494 | 126 | 79.68% | 70.67% | 74.91% |
| 0.05 | 573 | 474 | 99 | 82.72% | 67.81% | 74.53% |
| 0.10 | 555 | 467 | 88 | 84.14% | 66.81% | 74.48% |
| 0.20 | 532 | 450 | 82 | 84.59% | 64.38% | 73.11% |
| 0.50 | 490 | 423 | 67 | 86.33% | 60.52% | 71.15% |
| 0.90 | 433 | 387 | 46 | 89.38% | 55.36% | 68.37% |

At 0.02, 86 Qwen-positive sentences are removed: 40 labeled positives and 46 labeled negatives. Precision rises 4.04 percentage points while recall falls 5.72 points. At 0.05, 133 sentences are removed: 60 labeled positives and 73 labeled negatives. These are descriptive differences, not independent evidence that one threshold generalizes.

## Batch-grouping checks

The same saved Qwen decisions were post-filtered for the complete development split under one-paper and up-to-five groupings. These are separate original Qwen runs, not re-batching of the same predictions.

| Grouping | Threshold | Kept | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| 1 paper | baseline | 637 | 79.91% | 72.82% | 76.20% |
| 1 paper | 0.02 | 558 | 84.77% | 67.67% | 75.26% |
| 1 paper | 0.05 | 527 | 86.53% | 65.24% | 74.39% |
| Up to 5 | baseline | 596 | 84.40% | 71.96% | 77.68% |
| Up to 5 | 0.02 | 541 | 87.06% | 67.38% | 75.97% |
| Up to 5 | 0.05 | 509 | 88.61% | 64.52% | 74.67% |
| Up to 17 | baseline | 706 | 75.64% | 76.39% | 76.01% |
| Up to 17 | 0.02 | 620 | 79.68% | 70.67% | 74.91% |
| Up to 17 | 0.05 | 573 | 82.72% | 67.81% | 74.53% |

For all three saved output sets, these thresholds improve measured precision and reduce recall; F1 decreases relative to no additional post-filtering. This operation does not save input tokens from the already executed Qwen request. Applying a threshold before Qwen instead would change the Qwen input and potentially its other decisions, so these results do not predict that experiment.

All development papers and LLM-derived labels have prior development exposure; the checkpoint and original threshold were selected using the validation pool. No human adjudication or independent confirmation was added. Per-domain results, exact thresholds, input hashes, baseline responses and code are in `results.json`, `plan.json`, and `scripts/bench/nli_post_qwen_development.py`. Test outputs were not scored by this analysis.
