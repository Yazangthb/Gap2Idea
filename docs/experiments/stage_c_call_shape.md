# Stage C call shape: ids-only answers and one call per paper (2026-11-07)

Follow-up to task 5a (`docs/experiments/baseline_qwen.md`, branch `task-5a-qwen-baselines`), which found that
Qwen given whole papers beats the funnel on arXiv gold. Same model and server settings as the funnel
(Qwen3.8-27B, temperature 0, seed 0, no thinking), same Stage A+B candidates (short NLI, tau 0.354; 2,522 on gold
test). Token counts are Qwen's own, logged per call. Gold labels are LLM labels.

## 1. Ids-only answer, 20 candidates per call (`scripts/bench/stage_c_ids_only.py`)

The funnel answered `{"id", "keep"}` for every candidate: 11.3 output tokens per candidate (measured on 200), so
about 24k output tokens on arXiv gold against 1.6-2.6k for the whole-paper baseline. Changing only the answer to
`{"positives": [...]}`:

| Mode (gold test, 34 papers) | Per-item answer P / R / F1 | Ids-only P / R / F1 | Paired F1 diff (95% CI) | Ids-only output tokens |
|---|---|---|---|---|
| keep_biased | 56.3 / 85.7 / 0.679 | 53.9 / 85.5 / 0.662 | -0.018 (-0.040 to +0.006) | 6,547 |
| neutral | 68.2 / 79.7 / 0.735 | 65.0 / 78.5 / 0.711 | -0.023 (-0.051 to +0.003) | 5,129 |
| strict | 86.6 / 57.4 / 0.690 | 79.2 / 62.0 / 0.696 | +0.006 (-0.031 to +0.042) | 3,507 |

The format changed 7.6-9.3% of verdicts; repeating the same batches in one format changed 0.4% (70 batches run
twice by accident). The model keeps more with the ids-only answer, so precision drops.

Files: `stage_c_ids_only.json`, `stage_c_ids_only_{mode}.jsonl` (one line per batch: start, positives, tokens).

## 2. One call per paper (`scripts/bench/stage_c_per_paper.py`)

All of a paper's candidates in reading order in one call, ids-only answer. Two variants; the choice rule (higher
mean F1 over the three modes on gold VALIDATION, then one test run of the winner) was written in the script
before any call.

Validation (34 papers):

| Mode | 20 per call (old) | Per paper, no context | Per paper, +-1 context |
|---|---|---|---|
| keep_biased | 66.8 / 82.0 / 0.736 | 71.5 / 75.4 / 0.734 | 81.2 / 66.2 / 0.730 |
| neutral | 75.7 / 74.0 / 0.748 | 76.7 / 68.7 / 0.725 | 86.8 / 59.2 / 0.704 |
| strict | 90.4 / 52.5 / 0.664 | 85.8 / 54.6 / 0.668 | 95.8 / 45.9 / 0.621 |
| Mean F1 | 0.716 | **0.709** (chosen) | 0.685 |
| Input tokens per mode | ~424k (estimated) | ~109k | ~223k |

Test, no-context variant (all 34 papers):

| Mode | 20 per call (old) | Per paper | Paired F1 diff (95% CI) |
|---|---|---|---|
| keep_biased | 56.3 / 85.7 / 0.679 | 66.3 / 80.7 / 0.728 | +0.049 (-0.012 to +0.114) |
| neutral | 68.2 / 79.7 / 0.735 | 69.0 / 77.9 / 0.732 | -0.003 (-0.067 to +0.056) |
| strict | 86.6 / 57.4 / 0.690 | 72.9 / 61.6 / 0.668 | -0.022 (-0.088 to +0.040) |

Test, 28 arXiv papers, against the whole-paper baseline (task 5a setup A, its saved verdicts):

| Mode | Per paper P / R / F1 | Whole paper P / R / F1 | Per paper minus whole (95% CI) | Input tokens | Qwen seconds |
|---|---|---|---|---|---|
| keep_biased | 62.1 / 81.0 / 0.703 | 74.6 / 73.6 / 0.741 | -0.038 (-0.071 to +0.001) | 94,879 vs 552,686 | 63 vs 112 |
| neutral | 65.0 / 78.7 / 0.712 | 83.1 / 68.1 / 0.749 | -0.036 (-0.116 to +0.046) | 94,235 vs 552,042 | 68 vs 108 |
| strict | 68.2 / 59.8 / 0.637 | 89.2 / 54.8 / 0.679 | -0.042 (-0.132 to +0.047) | 96,027 vs 553,834 | 61 vs 104 |

Output tokens are similar (2.0-3.0k vs 1.6-2.6k). Times are summed call times on different GPU servers (ours on
GPU 0, task 5a on GPU 1) and exclude Stage B.

Reading: one call per paper costs about 5.7x fewer tokens than whole-paper prompting and about a quarter of the
old funnel, at about the same F1 as the old funnel, but the strict mode loses its precision on test (72.9% vs
86.6% overall; 68.2% on arXiv). The +-1 context variant was not run on test.

Files in `docs/experiments/stage_c_per_paper/`: `{val,test}_{variant}_{mode}.jsonl` (one line per paper: keys,
positives, tokens, seconds), `{val,test}_{variant}.json` (scores), `chosen_variant.txt`.

## Earlier funnel verdicts added with this note

For task 5a's same-mode comparisons (its note 2): `stage_c_rubric_test_{keep_biased,neutral,strict}.jsonl`,
`stage_c_sal40_modes_*.jsonl` and the NLI score files `data/gold/nli_*_scores.npz` that map their `i` back to
(paper, idx) through `stage_c_nli_qwen.candidates` and `stage_c_sal40_modes.py`.
