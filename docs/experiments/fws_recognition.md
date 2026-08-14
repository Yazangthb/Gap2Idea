# FWS future-work recognition — reproduction, leakage audit, and honest re-eval

Benchmark of the extraction funnel on the *other* gap type (future work), against:

> Zhang, Xiang, Hao, Li, Qian, Wang. *Automatic Recognition and Classification of Future
> Work Sentences from Academic Articles in a Specific Domain.* J. Informetrics, 2022.
> Data + code: [xiangyi-njust/FWS](https://github.com/xiangyi-njust/FWS).

Their reported recognition SOTA: **BernoulliNB, Macro-F1 = 0.9073** (beats their BERT/BiLSTM/
TextCNN). Corpus: 64,893 sentences from 9,013 ACL papers, 13.9 % future-work, section labels
shipped. Recognition is near-lexical ("in future work", "we plan to", "remains to be").

## TL;DR

Their 0.907 **reproduces exactly** — and is an **evaluation-leakage artifact**. Their
`SelectKBest(chi2)` is fit on the *full* dataset before 10-fold CV; refitting it inside each
fold drops the *same* model to **0.825**. Under honest evaluation the ceiling is ~0.83–0.88,
and a small fine-tuned encoder (bge-small, 33 M) is the best model — not their NB, not the
110 M biomedical/scientific transformers.

## 1. Reproduction + leakage audit (their protocol: balanced 1:1, lemmatised BoW, chi²-14k, 10-fold CV)

| Pipeline | Macro-F1 |
|---|---|
| their reported | 0.9073 |
| **LEAKY** — chi² fit on full data, then CV (their code) | **0.9072** (std 0.005) |
| **PROPER** — chi² refit inside each fold | **0.8252** (std 0.012) |
| **leakage inflation** | **+0.082** |

The leaky pipeline reproduces their number to three decimals; removing the leak costs 0.082.
The selector chooses the top-14k features using the test folds' labels — classic feature-
selection leakage. (Their protocol is also balanced 1:1, discarding 84 % of real negatives,
and CV-splits sentences with no paper grouping, so templated FW sentences leak across folds.)

## 2. Unified comparison — balanced 1:1, clean single held-out split (chi² fit on train only)

| Model | Cost | pos-F1 | Macro-F1 |
|---|---|---|---|
| shipped cue+head (zero-shot, no training) | ~free | 0.672 | 0.731 |
| TF-IDF word + logreg | ~free | 0.824 | 0.827 |
| BernoulliNB (1–4-gram, chi²) — *their model* | ~free | 0.822 | 0.830 |
| SciBERT (110 M) fine-tuned | GPU | 0.864 | 0.864 |
| PubMedBERT (110 M) fine-tuned | GPU | 0.875 | 0.877 |
| **bge-small (33 M) fine-tuned** | GPU | 0.875 | **0.879** |

On a clean split the encoders beat the cheap lexical models by ~0.05, and **bge-small (33 M)
tops both 110 M transformers** — the "data not model" pattern, now on its third dataset/domain
(after RCT and the FWS full-stream). PubMedBERT's biomedical pretraining buys nothing on NLP
text yet still matches SciBERT; capacity and domain both wash out after fine-tuning.

## 3. Honest held-out, imbalanced (14 %), paper-id split — the deployable setting

| Model | stage | pos-F1 | Macro-F1 |
|---|---|---|---|
| bge-small ft | detector | 0.741 | 0.850 |
| SciBERT ft | detector | 0.746 | 0.853 |
| PubMedBERT ft | detector | 0.750 | 0.853 |
| TF-IDF word+char SVM (cheap) | — | 0.736 | 0.847 |
| BernoulliNB (cheap) | — | 0.685 | 0.819 |
| shipped cue+head | +Stage C | 0.624 | 0.786 |

At realistic prevalence macro-F1 is dominated by the easy 86 % negative class, so everything
compresses to ~0.82–0.85; cheap ≈ expensive here. **Stage C is neutral-to-slightly-negative at
this prevalence** (e.g. bge 0.850 → 0.849), consistent with the RCT finding: Stage C is a
low-prevalence precision tool, not a high-prevalence one.

## Stage C error analysis + optimization (shipped zero-shot detector, 699 predicted-pos)

Inspecting what `validate_fws` keeps/drops on the shipped detector's predictions:

| Stage-C prompt | kept-FP | dropped-TP | P | R | pos-F1 | Macro-F1 |
|---|---|---|---|---|---|---|
| validate_fws (initial) | 139 | 18 | 0.768 | 0.526 | 0.624 | 0.786 |
| **keep-biased (shipped)** | 141 | 7 | 0.770 | 0.538 | **0.633** | **0.791** |
| drop-strict | 41 | 176 | 0.880 | 0.345 | 0.496 | 0.720 |

- The **139 kept "false positives" are overwhelmingly real future-work the gold mislabels as 0**
  ("*a thorough investigation ... remains future work*", "*further research is required*",
  "*it would be interesting to try ...*" all labelled negative). Precision is **label-capped**,
  not prompt-capped — same as the RCT Stage-C analysis.
- **`drop-strict` proves it**: chasing precision (kept-FP 139→41, P 0.77→0.88) drops 176 *real*
  future-work sentences, collapsing recall 0.53→0.35 and F1 to 0.496.
- **`keep-biased` is the win**: explicitly keeping weak/implicit own-future-work (hopes about
  one's own resource, "currently experimenting", "we can further improve") cut wrongly-dropped
  TPs 18→7 for +2 FPs → pos-F1 0.624→**0.633**, macro 0.786→**0.791**. Shipped as the
  `validate_fws` default. (Caveat: selected on the test predictions — a dev-split reselection
  would firm this up; the gain is small and the change is principled, not fitted to noise.)

## Findings

1. **The published SOTA reproduces but is a leakage artifact** (+0.082 from chi²-on-full-data).
   Honest number for their own model: ~0.825.
2. **bge-small (33 M) is the best model on a clean split** (0.879), beating 110 M PubMedBERT /
   SciBERT — third confirmation of *data not model*.
3. **Cheap ≈ expensive only at realistic prevalence** (macro dominated by negatives); on a clean
   balanced split the encoders genuinely win.
4. **Stage C confirmed as a low-prevalence precision tool** across a second dataset and domain.
5. Our shipped zero-shot head transfers to the new gap type (future work) but is recall-limited
   (0.53), so it needs in-domain training to be competitive — same as RCT.

## Reproduce

```bash
# honest held-out (imbalanced, paper-id split): zero-shot shipped, or --train for encoders
python scripts/bench/bench_fws.py                 # zero-shot shipped pipeline
python scripts/bench/bench_fws.py --train         # fine-tune SciBERT/PubMedBERT/bge
# cheap lexical models, honest held-out
python scripts/bench/bench_fws_cheap.py
# their protocol + all models in one table (balanced, clean split); needs V100 for encoders
python scripts/bench/bench_fws_balanced.py
```
