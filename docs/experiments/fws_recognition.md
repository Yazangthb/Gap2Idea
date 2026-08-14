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
