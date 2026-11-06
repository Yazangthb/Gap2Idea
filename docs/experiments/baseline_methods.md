# Task 5b: registered methods and evaluation choices

These choices were written before the new full-test experiment. This document is
the protocol, not a result table. Each run additionally writes exact source/input
hashes, software versions, model revisions and `preregistered_methods.json` before
fitting, then `frozen_settings.json` for every method **before any test scoring**.
Change a method in a new run and document why; do not choose changes from test F1.

## Unified held-out comparison

| ID | Registered implementation | Scope / qualification |
|---|---|---|
| `tfidf_lr` | Word TF-IDF (1,2), min_df=2, sublinear_tf, max_features=200000; logistic regression C=4, balanced class weights, max_iter=2000 | SAL limitation, FWS future-work, arXiv any-gap; fit on training papers only |
| `binary_nb` | Binary word counts (1,4), min_df=2, max_features=200000; BernoulliNB alpha=1 | Same targets; separate from chi-square-selected Zhang adaptation |
| `kilicoglu_rules` | Python port of top-section gate, limitation-section/paragraph spans, introductory/citation/contrast rejection; fixed threshold .5 | Limitation only; requires exhaustive aligned paragraph/section metadata; regex tokenization approximates original Java linguistic preprocessing |
| `kilicoglu_section_proxy` | Same lexical rules, but each contiguous section substitutes for a paragraph | Explicit approximation for SAL raw section_name or arXiv recovered LaTeX sections. Must retain this name/qualification in reporting. Not used for FWS |
| `modernbert_nli` | `tasksource/ModernBERT-large-nli` at `ca476cb923a8637073d4ceb0f19f7fc236e260d4`; original broad hypothesis below; length 256; entailment/(entailment+contradiction) | NLI-only, no LLM, no section/junk gate. Validation selects threshold. Not assumed to reproduce an earlier mixed-gold headline |
| `modernbert_nli_typed` | Same pinned NLI model and score; max over dataset-specific short hypotheses below | Separately registered optional variant; report all attempted variants, not just test winner |
| `scibert_ft` | `allenai/scibert_scivocab_uncased`; binary head; fixed 3 epochs, AdamW lr=2e-5, weight_decay=.01, clip norm 1, length 256, unweighted cross-entropy; local remaining-run batch 1 | In-domain train papers only; final epoch, then validation threshold. Saves safetensors and resolved base-model revision |
| `biomedbert_ft` | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`; same fine-tuning protocol | Generic fine-tune, **not Lan EDA/PBA**; supports SAL/arXiv and optional FWS control |
| `zhang_nb` | Raw-text TF-IDF (1,4), min_df=3, sublinear_tf, max_features=200000; train-only chi2 top 14000 (all if fewer); BernoulliNB alpha=1e-5 | FWS held-out adaptation of the model family. No lemmatization/balancing in this row; not an exact reproduction of the published balanced sentence-CV protocol |

The feature cap is a registered resource limit. A run with a different cap has a
different configuration and must retain it in its report. Encoder revisions can
be pinned explicitly with `--encoder-revision`; otherwise the resolved revision
is recorded. GPU execution is seeded but not promised to be bitwise deterministic
across devices/software. Use separate run directories for each training seed.

Encoder resource registration, 2026-11-06, before any full encoder test run:
the remaining-run launcher uses the pinned base-model commits in
[task5b_remaining_runs.md](task5b_remaining_runs.md), encoder batch 1 on the local
6 GB GPU, CUDA fp16 autocast/GradScaler with fp32 parameters, non-reentrant
gradient checkpointing, and AdamW `foreach=False`. CPU uses fp32. Initialization
is prepared on CPU in a separate process and saved as safetensors before GPU
training. New base downloads/initialization caches use `data/task5b/hf_cache/`
on the project drive. These changes address observed resource failures, not test
performance; all choices and base metadata are frozen per run. Completed SAL/FWS
baseline results are unchanged. SAL legacy encoder imports remain qualified as
cached decisions with unverified original training/threshold-selection history.

### NLI hypotheses

Original (verbatim from this project's `nli_score_remote.py`):

> This sentence points out a limitation, weakness, shortcoming, open research problem, or future-work direction of the research described.

Typed:

- SAL: `This is a limitation.`
- FWS: `This is future work.`
- arXiv: maximum of `This is a limitation.`, `This is future work.`, `This is a research gap.`

Three-way label indices are resolved from model metadata. Neutral is excluded in
the entailment-versus-contradiction score, following the reference implementation.
Model input is the sentence as premise and hypothesis as the paired second text.
Inference uses eager attention and disables ModernBERT reference compilation so
Windows does not require Triton. CUDA uses float16; CPU uses float32. These runtime
choices are recorded in the method settings.
On Windows, parameters are decoded tensor by tensor with the safetensors parser
and copied into the model to avoid native crashes in the installed stack's
memory-mapped loader. Neural operations run in separate processes to release GPU
memory between stages. The checkpoint values and pinned revision are unchanged.
No fixed historical .75 / .669 / .354 threshold is borrowed across datasets.

## Split and target policy

- SAL: public original train/dev (120/40 papers), cached 40-paper test fixture
  (7968 sentences, 205 positives). Align raw test by paper, reading order and text
  after stripping outer whitespace only; use the **unchanged cached sentence** as
  model input. Verify 8882 raw rows minus 914 omitted **negative** fragments.
- Legacy FWS (`--fws-protocol legacy`): reproduce `stage_c_package/breadth_funnel.py` selection: exclude
  exact stripped texts in `data/scibert_prep/train.jsonl`, seeded paper shuffle
  (`Random(0)`), select eligible paper groups until at least 1500 rows. Require
  **209 papers / 1506 sentences**. Train on original CSV papers outside those test
  IDs; reserve 10% of remaining sorted/shuffled paper IDs for validation (seed 0).
- Current FWS default, registered 2026-11-06 before full FWS scoring:
  `fws_current_gold_v1` reconstructs ALL sentences of the 162 uploaded gold TRAIN
  papers, lowercases both exclusion text and every FWS input, drops matching
  strings from the complete FWS corpus, then applies the same seeded paper-group
  sampling algorithm. It does not need the legacy training file. The locked test
  also has 209 papers / 1506 sentences / 220 positives, but no claim of historical
  membership identity is made. See [the detailed current protocol](task5b_fws_current_gold.md).
  This FWS-only revision does not change the completed SAL run or its inputs.
- arXiv: filter official mixed-gold train/val/test splits to `arxiv_cs` only;
  require **28 test papers / 10388 sentences**. Train paper count is whatever the
  arXiv subset contains; the proposal's 162 mixed-domain training papers is not
  silently interpreted as 162 arXiv papers. Prefer original complete Stage-B
  exports; otherwise verify every positive index/text against canonical splitting
  of original full text. Unannotated sentences become `none` only under the
  finalized corpus's positive-only annotation convention.
- Paper IDs must be disjoint across splits; duplicates/mismatches fail. Exact
  cross-split sentence-string overlaps are counted in manifests, not removed
  silently (generic headers recur). No guarantee of independent sentence wording.
- arXiv supervised/NLI rows use limitation OR future_work OR research_gap. Rules
  use limitation only, including treating future_work/research_gap as negatives.
  These are different tasks: do not rank their F1 values as a common-target table.
- Positive-only arXiv gold `section` fields must never serve as model features.
  Section rules use original LaTeX blocks or independently aligned annotations
  for **all** sentences. Missing/unmapped sections stay unknown and negative at
  the rule gate; do not drop them from the denominator.

## Selection and statistics

Fit vocabulary, IDF, chi-square selection and learned weights on **train only**.
For each learned/scored method select a threshold from .05 through .99 in .01
steps on validation only. Optimize positive F1 for SAL/arXiv and macro F1 for FWS.
Ties prefer the threshold closest to .5, then the larger threshold. Fixed rules
remain at .5. Default neural batch size 8; use 4 on a 6 GB GPU if necessary.
No test-selected checkpoint, best-test seed, or test-selected NLI hypothesis.

Report positive precision/recall/F1, confusion counts and macro F1 (particularly
for FWS). Zero-denominator metrics are 0. CI uses 2000 paper-cluster bootstrap
replicates, NumPy seed 0, pooled sentence confusion counts, percentile [2.5,97.5].
These intervals condition on the trained model and threshold; they do not measure
training-seed or threshold-selection uncertainty. Paired differences use exactly
the same paper resamples after checking every paper/index, sentence hash and gold
target. No paired CI can be computed from a scalar headline F1 or candidate-only
verdict list. Record missing paired evidence as N/A.

## Secondary / externally supplied methods

`audit-zhang` separately saves both fold-local and deliberately global-fitted
preprocessing/chi2 diagnostics on a 1:1 balanced, sentence-stratified CV pool.
It requires original aligned lemmatized text. This protocol differs from the
209-paper unified test. Its contrast combines vocabulary/IDF and supervised
feature-selection exposure; do not call it an isolated chi2 effect or promise
that it will exactly recreate historical .907/.825 scores. Both diagnostic
variants are saved, and neither is used to pick the held-out model.

Lan's published PubMedBERT + EDA/PBA checkpoint/training implementation is absent
from this checkout and the fetched reference branch. Generic BiomedBERT is not
relabelled as that method. Obtain the original augmentation recipe/code and
training-only augmented data; train with unchanged validation/test sets; then
use `score-external` to import complete frozen predictions with a provenance
protocol. That adapter also accepts existing SAL encoder vectors and funnel
results after their owners export exact, complete paper/index/text identities.

## Primary implementation references

- [Kilicoglu original Java rule implementation](https://github.com/kilicogluh/limitation-recognizer/blob/master/src/gov/nih/nlm/limitations/RuleBasedLimitationSentenceRecognizer.java) and [filters/helpers](https://github.com/kilicogluh/limitation-recognizer/blob/master/src/gov/nih/nlm/limitations/Utils.java).
- [Pinned ModernBERT NLI model](https://huggingface.co/tasksource/ModernBERT-large-nli/tree/ca476cb923a8637073d4ceb0f19f7fc236e260d4).
- [SAL source](https://github.com/MengfeiLan/SAL_Type_Classification/tree/checkpoint/data), [FWS source](https://github.com/xiangyi-njust/FWS).
- Reference code inspected at `origin/chore/repo-reorg` commit
  `f34cc7daa3c739832c0c9b4db0ea4ad80b654567`: `nli_score_remote.py`,
  `nli_variants.py`, `stage_c_nli_qwen.py`, `gold_label_full.py`,
  `export_stageb_data.py`, `build_arxiv_latex_blocks.py`, `bench_fws_balanced.py`.
  This branch was fetched for reference; unrelated history was not merged.

## Verified completed results (audit 2026-11-06)

These are the nine completed method/dataset evaluations, verified against saved
predictions and manifests without rerunning models. All CIs resample papers
(2000 replicates). FWS uses `fws_current_gold_v1`; its membership/casing must be
matched before pairing with a funnel run. See [the completion audit](task5b_status.md)
for missing methods, protocol qualifications and remaining tasks.

| Dataset | Method | P | R | Positive F1 [95% CI] | Macro F1 [95% CI] | F1 difference from funnel |
|---|---|---:|---:|---:|---:|---|
| SAL | tfidf_lr | 0.473 | 0.380 | 0.422 [0.355, 0.488] | 0.704 [0.670, 0.738] | N/A: complete matching funnel predictions needed |
| SAL | binary_nb | 0.000 | 0.000 | 0.000 [0.000, 0.000] | 0.493 [0.492, 0.494] | N/A: complete matching funnel predictions needed |
| SAL | kilicoglu_section_proxy | 0.467 | 0.478 | 0.472 [0.363, 0.559] | 0.729 [0.675, 0.774] | N/A: complete matching funnel predictions needed |
| SAL | modernbert_nli | 0.331 | 0.585 | 0.423 [0.377, 0.470] | 0.701 [0.677, 0.725] | N/A: complete matching funnel predictions needed |
| FWS (current gold) | tfidf_lr | 0.802 | 0.700 | 0.748 [0.689, 0.800] | 0.854 [0.820, 0.883] | N/A: complete matching funnel predictions needed |
| FWS (current gold) | binary_nb | 0.808 | 0.632 | 0.709 [0.651, 0.762] | 0.833 [0.801, 0.863] | N/A: complete matching funnel predictions needed |
| FWS (current gold) | modernbert_nli | 0.387 | 0.495 | 0.434 [0.370, 0.492] | 0.661 [0.623, 0.695] | N/A: complete matching funnel predictions needed |
| FWS (current gold) | modernbert_nli_typed | 0.711 | 0.705 | 0.708 [0.649, 0.761] | 0.829 [0.795, 0.860] | N/A: complete matching funnel predictions needed |
| FWS (current gold) | zhang_nb | 0.769 | 0.682 | 0.723 [0.663, 0.778] | 0.839 [0.805, 0.870] | N/A: complete matching funnel predictions needed |

The SAL rule row is a section-proxy approximation; the FWS Zhang row is a
held-out raw-text adaptation. Neither is an exact published-method reproduction.
The proposal's final completion condition is **not met** until missing method/
dataset runs and paired funnel differences are resolved or explicitly scoped out.
