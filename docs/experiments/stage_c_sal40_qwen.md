# SAL-40 Stage C: Yandex vs Qwen, only the model changed (2026-11-05)

Same 391 PubMedBERT A+B positives (180 gold limitations) in document order, `rct_loose` prompt, ±1 inline context, batch 20,
same `LLMGapFilter` code path as the Yandex run. Qwen3.8-27B-FP8, revision `017b9c7a`, no thinking, temperature 0, seed 0.
Labels are SAL human gold, 205 limitations across 40 papers. 20 requests, 53 s, no fallback calls.

Script: `scripts/bench/stage_c_sal40_qwen.py`. Output (gitignored): `stage_c_sal40_qwen.json`. The script stops unless the
saved Yandex verdicts reproduce 269 kept and 156 TP; they did.

| Setup | Kept | Limitations kept | Non-limitations removed | Precision | Recall | F1 (95% CI, papers resampled) |
|---|---|---|---|---|---|---|
| A+B, no filter | 391 | 180/180 | 0/211 | 46.0% | 87.8% | 0.604 |
| + Yandex `yandexgpt-5-lite` | 269 | 156/180 (86.7%) | 98/211 (46.4%) | 58.0% | 76.1% | 0.658 (0.602–0.709) |
| **+ Qwen3.8-27B** | 327 | **179/180 (99.4%)** | 63/211 (29.9%) | 54.7% | **87.3%** | **0.673 (0.614–0.729)** |

Paired difference, Qwen minus Yandex: **F1 +0.015 (95% CI −0.018 to +0.047)**, so the two are not distinguishable on F1.
They agree on 283 of 391 verdicts. Limitations only one model kept: Qwen 24, Yandex 1. Non-limitations only one model kept:
Qwen 59, Yandex 24.

## Reading
- Qwen trades precision for recall relative to Yandex: recall is almost untouched, and it removes fewer false positives.
- Compared with no filter, Qwen raises precision by 8.7 points for a 0.5-point recall cost.
- Qwen was not selected on SAL. This is its first run here, but `rct_loose` was developed earlier using SAL test feedback.

## Limits
- SAL precision is capped by labels: an earlier single-reader review of 22 Yandex-run false positives put about 45% as real limitations SAL did not tag.
- This full-stream figure is not comparable with Lan et al.'s 0.821, which is on a section pool at 20.7% prevalence.
