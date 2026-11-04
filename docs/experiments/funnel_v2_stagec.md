# Funnel v2 — Stage C (yandexgpt-5-lite, mode=validate)

Judged subsample of 120 candidates from `funnel_v2_candidates_balanced.json` in 3 calls; kept 33/120. Gold gaps in pool: 46.

| | detector (before) | + Stage C |
|---|---|---|
| precision | 38.3% | **75.8%** |
| gaps retained (within pool) | 100% | 54.3% |

Stage C lifts precision 38.3% -> 75.8% while retaining 54.3% of the true gaps in the judged pool. Labels are LLM-gold.
