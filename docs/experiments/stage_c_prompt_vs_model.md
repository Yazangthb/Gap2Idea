# Stage C: model or prompt? (2026-11-05)

Same 80 candidates Sonnet judged on 2026-10-26: balanced set, `random.seed(11)`, 27 papers, 19 gold gaps
(15 research_gap, 4 limitation, 0 future_work), 61 gold negatives. Labels are LLM gold, not human.
Script: `scripts/bench/stage_c_prompt_vs_model.py`. Verdicts: `docs/experiments/stage_c_prompt_vs_model.json`
(gitignored because it holds gold labels).

| Setup | Requests | Kept | Gaps kept | False positives | Precision | F1 in sample |
|---|---|---|---|---|---|---|
| No filter | 0 | 80 | 19/19 | 61 | 23.8% | 0.384 |
| Yandex, `validate`, no context, batch 40 | 2 | 14 | 8/19 | 6 | 57.1% | 0.485 |
| Yandex, Sonnet rubric, context, batch 20 | 4 | 47 | 19/19 | 28 | 40.4% | 0.576 |
| Yandex, Sonnet rubric, context, all 80 in one call | 1 | 61 | 19/19 | 42 | 31.1% | 0.475 |
| Sonnet, same rubric, all 80 at once (Oct 26) | n/a | 30 | 18/19 | 12 | 60.0% | 0.735 |

Sonnet's verdicts were rebuilt from its reported 12 false positives and 1 false negative; the rebuild
reproduces its 18 TP and 12 FP.

### Qwen3.8-27B-FP8 (self-hosted vLLM, revision `017b9c7a`, temperature 0, seed 0)

Verdicts: `docs/experiments/stage_c_prompt_vs_model_qwen.json`. Run with `stage_c_prompt_vs_model.py qwen`.

| Setup | Requests | Time | Kept | Gaps kept | False positives | Precision | F1 in sample |
|---|---|---|---|---|---|---|---|
| `validate`, no context, batch 40, no thinking | 2 | 7 s | 13 | 5/19 | 8 | 38.5% | 0.313 |
| **Sonnet rubric, context, batch 20, no thinking** | 4 | 12 s | 35 | **19/19** | **16** | 54.3% | **0.704** |
| Sonnet rubric, context, all 80 in one call, no thinking | 1 | 10 s | 37 | 18/19 | 19 | 48.6% | 0.643 |
| Sonnet rubric, context, batch 20, thinking on | 4 | 177 s | 36 | 18/19 | 18 | 50.0% | 0.655 |

## Reading
- **Recall comes from the prompt setup.** Same model, gaps kept 8/19 → 19/19. This step also changes context
  and batch size, so they are not separated from the prompt.
- **Precision comes from the model.** Same rubric and same all-at-once view: Yandex keeps 42 negatives, Sonnet 12.
- **Batch size affects Yandex precision.** Same rubric: 28 false positives at batch 20, 42 in one call.
- `LLMGapFilter._batch_messages` appends a fixed instruction to drop "limitations of prior or other work".
  The `validate` setup lost 8 of 15 research_gap items, many of them about prior work. The rubric runs bypass it.

## Limits
19 gaps only, no future_work, one run per setup. Sonnet helped create the gold. Rubric prompt was written
before this test but has not been tuned or checked in reading order. Some false positives may be gaps the gold missed.
