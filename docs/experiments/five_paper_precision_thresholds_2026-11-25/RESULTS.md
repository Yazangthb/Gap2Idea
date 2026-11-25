# Precision-oriented NLI threshold development

Three fine-tuned NLI thresholds were compared using the same neutral Qwen3.8-27B-FP8 prompt and ±1 original sentence as context, with one Qwen request per five-paper group. Ten validation papers were selected from those not used in the three recent five-paper pilot groups; five were used for tuning and five for a separate check. Both groups belong to the validation pool used in fine-tuned checkpoint and original threshold selection, and their labels are LLM-generated.

| Fine-tuned NLI threshold, tuning group | Candidates | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 0.002103 | 224 | 83 | 74 | 52.9% | 83.8% | 0.648 |
| Medium 0.01 | 163 | 77 | 58 | 57.0% | 77.8% | 0.658 |
| High 0.05 | 134 | 69 | 30 | 69.7% | 69.7% | 0.697 |

The high threshold was selected by the prespecified maximum precision subject to at least 60% overall tuning recall. On the separate five-paper development check:

| Threshold | Candidates | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original 0.002103 | 215 | 56 | 14 | 80.0% | 65.9% | 0.723 |
| High 0.05 | 106 | 57 | 25 | 69.5% | 67.1% | 0.683 |

The high threshold did not consistently improve precision in this single-request setup. A post-hoc intersection of the two Qwen output sets yielded 70.4% precision / 69.7% recall on tuning, and 84.7% precision / 58.8% recall on the check group, but requires both Qwen calls and was invented after examining both outputs. This observation motivated a separate exploratory run on the previously analyzed 25-paper pool; it is not independent confirmation.
