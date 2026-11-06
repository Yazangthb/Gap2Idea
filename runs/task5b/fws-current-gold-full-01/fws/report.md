# Task 5b - fws (full)

Protocol: **fws_current_gold_v1**; lowercase every sentence before exclusion, training and inference. Historical funnel pairing requires exact sentence identity, not matching counts.

Positive metrics use pooled sentences. CI resamples whole papers (conditional on the fitted model).

| Method | Target | P | R | F1 [95% CI] | Macro F1 [95% CI] |
|---|---|---:|---:|---:|---:|
| tfidf_lr | future_work | 0.802 | 0.700 | 0.748 [0.689, 0.800] | 0.854 [0.820, 0.883] |
| binary_nb | future_work | 0.808 | 0.632 | 0.709 [0.651, 0.762] | 0.833 [0.801, 0.863] |
| modernbert_nli | future_work | 0.387 | 0.495 | 0.434 [0.370, 0.492] | 0.661 [0.623, 0.695] |
| modernbert_nli_typed | future_work | 0.711 | 0.705 | 0.708 [0.649, 0.761] | 0.829 [0.795, 0.860] |
| zhang_nb | future_work | 0.769 | 0.682 | 0.723 [0.663, 0.778] | 0.839 [0.805, 0.870] |

Funnel paired delta: **N/A until complete aligned sentence predictions are supplied**.
Run `compare` with those predictions; aggregate headline F1 values cannot support pairing.

Lan EDA/PBA: **not implemented as a claimed reproduction**; original augmentation recipe/code is missing.
Generic BiomedBERT fine-tuning is available as a separate baseline.

arXiv limitation rules have a different positive target from any-gap detectors; do not rank those rows together.
Section-proxy rules and raw-text Zhang adaptation must retain their qualification in the paper.
