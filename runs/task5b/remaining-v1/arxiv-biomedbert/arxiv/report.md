# Task 5b - arxiv (full)



Positive metrics use pooled sentences. CI resamples whole papers (conditional on the fitted model).

| Method | Target | P | R | F1 [95% CI] | Macro F1 [95% CI] |
|---|---|---:|---:|---:|---:|
| biomedbert_ft | any_gap | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.487 [0.483, 0.490] |

Funnel paired delta: **N/A until complete aligned sentence predictions are supplied**.
Run `compare` with those predictions; aggregate headline F1 values cannot support pairing.

Lan EDA/PBA: **not implemented as a claimed reproduction**; original augmentation recipe/code is missing.
Generic BiomedBERT fine-tuning is available as a separate baseline.

arXiv limitation rules have a different positive target from any-gap detectors; do not rank those rows together.
Section-proxy rules and raw-text Zhang adaptation must retain their qualification in the paper.
