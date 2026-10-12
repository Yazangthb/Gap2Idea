# Stage C — SAL-criteria experiments (Oct 2026)

Self-contained scripts + result artifacts for the SAL-annotation-criteria Stage C prompt
(strict and keep-biased variants) on the SAL (Lan et al. 2024) limitation benchmark.
Full write-up and tables: [`docs/experiments/sal_stage_c_summary.md`](../../docs/experiments/sal_stage_c_summary.md).

## Fixture
- `sal_test_allmodels.json` — SAL test set, 7,968 sentences in per-paper reading order; each row:
  `pmcid`, `sent`, `sec` (ABSTRACT/INTRO/METHOD/RESULT/DISCUSS/CONCL/…), `label` (1=gold limitation),
  `preds` (per-sentence 0/1 from all 8 Stage-B models). This is the evaluation pool for every script here.

## Scripts (run from repo root; need `.env` with YANDEX_API_KEY + YANDEX_FOLDER_ID — kept local)
| script | what it does | API calls |
|---|---|---|
| `stagec_newcriteria_sweep.py` | pick the judge model: 3 YandexGPT models × keep-biased SAL prompt on the 120-sentence hard sample | ~18 |
| `stagec_faithful_pubmed.py` | **the canonical run** — judge PubMedBERT's own 391 A+B positives in doc order; both prompts; emits (a) end-to-end funnel + (b) prevalence-20 | ~40 |
| `stagec_newcriteria_ab_prev.py` | 8-model union variant (superseded by the faithful run for absolute F1; batch-composition biased — see summary §⚠️) | ~104 |

## Result JSONs
- `newcriteria_sweep_results.json` — model sweep P/R/F1 (§1 of summary)
- `faithful_pubmed_verdicts.json` — per-sentence keep verdicts, both prompts (basis for §3/§4/§6)
- `ab_prev_funnel_results.json` — 8-model union funnel table (§5)
- `funnel_salcriteria_results.json` — earlier strict-SAL funnel (8 models, yandexgpt-5-pro)

## Headline (PubMedBERT, faithful run)
- End-to-end funnel: A+B F1 0.604 → **+C 0.652** (keep-biased SAL) / 0.658 (rct_loose).
- Prevalence-20: Stage C ~neutral (0.741–0.757 vs detector 0.749); committed airtight SOTA run = 0.810–0.812 vs their 0.821.
- Measured precision ~0.59 understates true precision (~0.87): ~45% of "false positives" are limitations SAL didn't label (gold under-labeling, unfixable by prompt).

## Reproduce
```bash
python stage_c_package/sal_criteria/stagec_faithful_pubmed.py
```
Note: `judge_batch` is batch-composition sensitive — judge on the real detector output in reading
order (as `stagec_faithful_pubmed.py` does), not on a mixed sample.
