# Task 5b - arxiv (full)



Positive metrics use pooled sentences. CI resamples whole papers (conditional on the fitted model).

| Method | Target | P | R | F1 [95% CI] | Macro F1 [95% CI] |
|---|---|---:|---:|---:|---:|
| tfidf_lr | any_gap | 0.442 | 0.560 | 0.494 [0.442, 0.550] | 0.732 [0.706, 0.760] |
| binary_nb | any_gap | 0.485 | 0.030 | 0.057 [0.027, 0.093] | 0.516 [0.501, 0.533] |
| modernbert_nli | any_gap | 0.499 | 0.710 | 0.586 [0.508, 0.653] | 0.779 [0.738, 0.815] |
| modernbert_nli_typed | any_gap | 0.413 | 0.738 | 0.529 [0.451, 0.601] | 0.747 [0.705, 0.785] |

Funnel paired delta: **N/A until complete aligned sentence predictions are supplied**.
Run `compare` with those predictions; aggregate headline F1 values cannot support pairing.

Lan EDA/PBA: **not implemented as a claimed reproduction**; original augmentation recipe/code is missing.
Generic BiomedBERT fine-tuning is available as a separate baseline.

arXiv limitation rules have a different positive target from any-gap detectors; do not rank those rows together.
Section-proxy rules and raw-text Zhang adaptation must retain their qualification in the paper.
