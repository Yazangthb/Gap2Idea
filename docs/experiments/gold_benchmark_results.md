# Funnel benchmark on the gap gold

Gold: **4462 gaps / 230 papers** (arXiv CS + biomedical SAL). Funnel targets **limitation + future_work**; **research_gap is out of funnel scope** (LLM/full-paper path) and excluded from funnel P/R. Matching: token-containment >= 0.7.

## Stage A — localization recall ceiling (target gaps inside the section slice)

| domain | ceiling recall |
|---|---|
| arxiv_cs | 55.6% (957/1720) |
| biomed_sal | 84.1% (427/508) |
| ALL | 62.1% (1384/2228) |

## rules — Stage A + cue rules

| domain | recall (target) | precision (strict / lenient) | F1 (strict) | preds |
|---|---|---|---|---|
| arxiv_cs | 30.6% | 47.1% / 61.9% | 26.7% | 679 |
| biomed_sal | 33.7% | 66.0% / 73.8% | 28.7% | 141 |
| ALL | 31.3% | 50.4% / 63.9% | 27.1% | 820 |

Per-type recall (all): future_work 54.9% (n=685), limitation 20.9% (n=1543)

## hybrid — Stage A + cue rules + bge-small head (hybrid)

| domain | recall (target) | precision (strict / lenient) | F1 (strict) | preds |
|---|---|---|---|---|
| arxiv_cs | 47.6% | 27.3% / 42.1% | 31.4% | 2323 |
| biomed_sal | 66.1% | 40.5% / 47.9% | 42.2% | 553 |
| ALL | 51.8% | 29.9% / 43.3% | 33.7% | 2876 |

Per-type recall (all): future_work 74.2% (n=685), limitation 41.9% (n=1543)

### Notes
- **strict precision** = prediction matches a gold *limitation/future_work*; **lenient** = matches any gold gap (incl. research_gap the funnel found but doesn't target).
- research_gap (the majority gold type) is deliberately not a funnel target; Stage C / full-paper LLM is the path for it.
- hybrid uses the bge-small(frozen)+logreg head (`data/pipeline_head.joblib`); the encoder pairing is assumed from the 384-dim head.
- gold is LLM-ensemble labelled (pending human validation), so these are provisional.
