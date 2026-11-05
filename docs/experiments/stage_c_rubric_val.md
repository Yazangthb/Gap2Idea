# Stage C rubric strictness on gold validation (2026-11-05)

Short NLI at τ 0.354, 2,278 candidates from the 34 validation papers (699 gaps), Qwen3.8-27B, context, batch 20, no thinking, seed 0.
Before Stage C: half A P 25.8%, R 86.9%; half B P 27.7%, R 87.4%. No test paper was loaded. LLM gold.
Script: `scripts/bench/stage_c_rubric_val.py`. Outputs (gitignored): `stage_c_rubric_val.json`, `stage_c_rubric_val_*.jsonl`.

| Rubric | Keeps gaps | Removes non-gaps | Precision | Recall | F1 | F1 half A / B |
|---|---|---|---|---|---|---|
| Keep-biased (current) | 94.1% | 82.9% | 66.8% | 82.0% | 0.736 | 0.734 / 0.738 |
| Neutral | 84.9% | 90.1% | 75.7% | 74.0% | 0.748 | 0.739 / 0.758 |
| Strict | 60.3% | 97.7% | 90.4% | 52.5% | 0.664 | 0.661 / 0.668 |

| Recall by type, half A / B | Limitation | Future work | Research gap |
|---|---|---|---|
| Keep-biased | 73.9 / 77.5% | 78.9 / 78.5% | 89.1 / 89.0% |
| Neutral | 65.0 / 70.3% | 71.9 / 78.5% | 80.3 / 78.5% |
| Strict | 54.8 / 50.5% | 59.6 / 76.9% | 43.8 / 47.1% |

The only difference between rubrics is the final instruction: favour YES, favour neither, or favour NO with extra NO rules.
Strict and neutral wording was written after reading false positives from the earlier test run, before any validation verdict.
These validation papers are easier for the current setup than test (F1 0.736 vs 0.678).
