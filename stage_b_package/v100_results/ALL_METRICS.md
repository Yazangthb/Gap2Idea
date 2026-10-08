# Stage B — all metrics, all models, all datasets

Consolidated from the saved result JSONs in this folder. Datasets: **Sci-Sentence (sci)**, **FWS (fws)**,
**SAL/RCT (sal)** for training/validation; **10 arXiv gold papers (49 gaps)** for end-to-end eval.
Head types: frozen = sentence-encoder + class-weighted logistic regression; FT = full fine-tune (all weights).

## 1. Validation macro-F1 (paper-disjoint split)

Per-dataset macro-F1 is over the classes present in that dataset (sci = 3 classes; fws = none+future_work;
sal = none+limitation). `overall` is the 3-class score on the pooled validation. Not comparable across columns.

| model | train data | sci | fws | sal | overall |
|---|---|---|---|---|---|
| bge-small + logreg | sci+fws+sal | 0.638 | 0.798 | 0.731 | 0.718 |
| bge-base + logreg | sci+fws+sal | 0.677 | 0.808 | 0.770 | 0.742 |
| e5-base + logreg | sci+fws+sal | 0.676 | 0.833 | 0.749 | 0.753 |
| mpnet + logreg | sci+fws+sal | 0.583 | 0.711 | 0.728 | 0.668 |
| MiniLM-L6 + logreg | sci+fws+sal | 0.635 | 0.746 | 0.635 | 0.656 |
| PubMedBERT (full FT) | sci+fws+sal | 0.922 | 0.891 | 0.921 | 0.890 |

Earlier models (trained on sci+fws only, single combined in-dataset report — no per-dataset split, no SAL):

| model | combined in-dataset macro-F1 |
|---|---|
| bge-small + logreg | 0.633 |
| DistilBERT (full FT) | 0.745 |
| SciBERT (full FT) | 0.846 |

## 2. Gold end-to-end — 10 arXiv papers, 49 gaps

Format: **recall_all / limitation / future_work / kept-sentences**. `kept` = load / precision proxy
(true precision not measurable on positive-only gold).

| model | train data | head-only | rules + head |
|---|---|---|---|
| bge-small | sci+fws | 0.347 / 0.273 / 0.714 / 90 | 0.429 / 0.455 / 0.786 / 104 |
| DistilBERT FT | sci+fws | 0.204 / 0.000 / 0.643 / 47 | 0.306 / 0.182 / 0.714 / 63 |
| SciBERT FT | sci+fws | 0.204 / 0.000 / 0.643 / 46 | 0.327 / 0.273 / 0.714 / 61 |
| bge-small | sci+fws+sal | 0.388 / 0.364 / 0.786 / 84 | 0.429 / 0.545 / 0.786 / 97 |
| PubMedBERT FT | sci+fws+sal | 0.245 / 0.182 / 0.714 / 50 | 0.327 / 0.364 / 0.714 / 65 |
| bge-base | sci+fws+sal | 0.327 / 0.273 / 0.714 / 60 | 0.408 / 0.455 / 0.786 / 76 |
| e5-base | sci+fws+sal | 0.347 / 0.273 / 0.786 / 57 | 0.408 / 0.455 / 0.786 / 71 |
| mpnet | sci+fws+sal | 0.531 / 0.364 / 0.714 / 175 | 0.571 / 0.455 / 0.786 / 186 |
| **MiniLM-L6** | sci+fws+sal | 0.429 / 0.364 / 0.714 / 156 | **0.571 / 0.636 / 0.857 / 170** |

## Headlines
- **Best on gold:** MiniLM-L6 + logreg + SAL — rules+head 0.571 (limitation 0.636, future_work 0.857), but keeps 170 (lower precision). bge-small+SAL is the precise alternative (0.429, keeps 97).
- **SAL fixed limitation** for the frozen heads (bge-small gold limitation 0.455 → 0.545).
- **Every full fine-tune overfits:** PubMedBERT/SciBERT/DistilBERT top validation (up to 0.890) yet lose on gold (≤0.327).
- **Validation ≠ task:** per-dataset/overall validation is anti-correlated with gold recall. Decide on gold.
- Not run: SPECTER2 (legacy .bin; needs safetensors conversion); regularized FT (LoRA / freeze-encoder+head).

Source files: frozen_result, bert_result, scibert_result, frozen_sal, pubmedbert_sal, frozen_bgebase, frozen_e5base, frozen_mpnet, frozen_minilm (.json).
