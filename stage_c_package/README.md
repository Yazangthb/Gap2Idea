# Stage C package

Self-contained notebooks for **Stage C** — the LLM precision filter that runs on the funnel's
survivors (Stage A localize + Stage B classify). Uses **Yandex paid models** via the repo's LLM layer.

## Files
| File | What it is |
|---|---|
| `stage_c_all_experiments.ipynb` | (1) model x batch-size probe over Yandex models on a small labeled SAL sample; (2) the real metric: Stage B detector -> Stage C on SAL, full-stream and prevalence-matched vs Lan 2024 PubMedBERT 0.821. |
| `stage_c_final.ipynb` | The shipped config: `validate_rct` judge on `yandexgpt-5-lite` @ batch 20. A clean `stage_c_filter(sentences)` step, a demo, and the SAL headline. |

## What Stage C is
Stage A+B is recall-first and leaks false positives (prior-work critiques, contribution claims,
acknowledgments, formulas). Stage C asks an LLM per candidate *"is this the authors' own gap?"* and
drops the No's. Because A+B already cut ~98% of sentences, the LLM sees ~6 sentences/paper.

- Imports `LLMGapFilter` + the tuned prompts from `gap2idea` (single source of truth).
- Mode `validate_rct` (biomedical, keep-biased) for SAL; `validate` for general/arXiv text.
- **Precision filter:** only improves F1 where there are true negatives -> evaluated on **SAL**
  (not positives-only gold, where a filter can only cost recall).

## Model choice (from the probe)
`yandexgpt-5-lite` @ batch 20 is the default (best F1 / cost on the probe); flip `WINNER` to
`yandexgpt-5-pro` for robustness. All Yandex models honored strict JSON; the OpenRouter free tier
did not, which is why we use Yandex paid here.

## How to run
Open on the `gap2idea` (`.venv`) kernel and **Run All**. Needs `YANDEX_API_KEY` + `YANDEX_FOLDER_ID`
in `.env` (provider is forced to Yandex). `TEST_CAP` keeps a smoke run fast; set it to `None` for the
full SAL metric. Stage C spends real Yandex credits (small — only the Stage-B positives are judged).
