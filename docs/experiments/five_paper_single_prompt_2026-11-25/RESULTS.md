# Single-request precision prompt experiment

One Qwen3.8-27B FP8 request per five-paper group, using the same fine-tuned NLI candidates and one original neighboring sentence on each side. All results are sentence-level, against the existing LLM-generated labels; recall uses all labeled gaps in the papers as its denominator.

The plan fixed a random 15-paper tuning set and a separate 10-paper check set, grouped in fives. The selected prompt maximized pooled tuning precision subject to at least 60% recall. All groups and prompts were frozen before inference. These development papers and the NLI model have prior development exposure, so the check is not an untouched test set or a human evaluation.

| Tuning prompt, 15 papers | TP | FP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Neutral | 274 | 102 | 72.87% | 78.06% | 75.38% |
| Strict (selected) | 221 | 28 | 88.76% | 62.96% | 73.67% |
| Contrastive examples | 293 | 186 | 61.17% | 83.48% | 70.60% |
| Focus | 293 | 142 | 67.36% | 83.48% | 74.55% |

| Check prompt, 10 papers | TP | FP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Neutral | 143 | 113 | 55.86% | 78.14% | 65.15% |
| Strict | 126 | 151 | 45.49% | 68.85% | 54.78% |

The strict prompt did not generalize: it reduced both precision and recall on the check groups. Its behavior varied substantially: on group 4 it achieved 100% precision and 29.51% recall (18 TP, 0 FP); on group 5 it achieved 41.70% precision and 88.52% recall (108 TP, 151 FP). The neutral prompt remains the stronger choice on this check. Neither single-group differences nor these historically studied, automatically labeled data establish reliable paper-level performance.

The frozen prompt texts, input hashes, group identities, raw outputs, token counts, and selection rule are in the JSON files beside this report; the runner is `scripts/bench/five_paper_single_prompt_tune.py`. The 27B service was already running before this experiment and was left running; no GPU service was started by the experiment.
