# Task 5b - sal (full)

Positive metrics use pooled sentences. CI resamples whole papers (conditional on the fitted model).

| Method | Target | P | R | F1 [95% CI] | Macro F1 [95% CI] |
|---|---|---:|---:|---:|---:|
| tfidf_lr | limitation | 0.473 | 0.380 | 0.422 [0.355, 0.488] | 0.704 [0.670, 0.738] |
| binary_nb | limitation | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.493 [0.492, 0.494] |
| kilicoglu_section_proxy | limitation | 0.467 | 0.478 | 0.472 [0.363, 0.559] | 0.729 [0.675, 0.774] |
| modernbert_nli | limitation | 0.331 | 0.585 | 0.423 [0.377, 0.470] | 0.701 [0.677, 0.725] |

Funnel paired delta: **N/A until complete aligned sentence predictions are supplied**.
Run `compare` with those predictions; aggregate headline F1 values cannot support pairing.

Lan EDA/PBA: **not implemented as a claimed reproduction**; original augmentation recipe/code is missing.
Generic BiomedBERT fine-tuning is available as a separate baseline.

arXiv limitation rules have a different positive target from any-gap detectors; do not rank those rows together.
Section-proxy rules and raw-text Zhang adaptation must retain their qualification in the paper.
