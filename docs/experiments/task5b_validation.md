# Task 5b implementation validation - 2026-11-05

## Remaining-run preparation update, 2026-11-06

The remaining-run launcher is ready; see
[task5b_remaining_runs.md](task5b_remaining_runs.md). It checks frozen data,
registers a suite plan, executes independent jobs sequentially, verifies completed
jobs before skipping them, and refuses to overwrite incomplete directories. It
also imports the original cached SAL encoder vectors with explicit unverified-
history qualifications, and exports label-free full test requests for the funnel.

**21 tests passed** after adding the launcher and CPU checkpoint preparation.
The actual pinned SciBERT and BiomedBERT models also completed a GPU smoke run at
`runs/task5b/encoder-smoke-20261106-v4/`: training, safetensors save/load,
validation threshold selection, prediction export, paper CIs and completion
markers. Each used up to 8 long training sentences and disjoint subsets of arXiv
validation papers for temporary validation/evaluation, with one training epoch.
The official arXiv/FWS tests were not scored. Checks confirmed changes to BOTH
classifier-head weights and encoder attention weights, rather than merely saving
an unchanged pretrained model. Smoke scores are not research results.

Resource failures discovered before success were preserved as incomplete smoke
directories. The default C: model cache initially ran out of space; the partial
SciBERT download created by this check was moved to the project's E: encoder
cache without deleting other user files. Direct GPU training also failed with
the original loading/resource setup. The validated preset prepares a seeded
classification initialization on CPU in its own process, loads safetensors for
GPU training, and uses batch 1, autocast/GradScaler, gradient checkpointing and
non-foreach AdamW. New encoder base/initialization caches stay on the project
drive. The current two base caches were validated successfully; original public
`.bin` loading uses restricted weights-only loading on the newer torch build.

These choices were set before any full new encoder test results. No full
remaining job was started and nothing was pushed. The completed SAL/FWS runs
were not changed. The label-free reference request folders contain exactly
7968 SAL, 1506 FWS and 10388 arXiv sentences, without gold labels.

**Latest audit, 2026-11-06:** the completed SAL and full FWS runs have now both
passed all saved-evidence checks, and the 17 offline tests passed again. All nine
method/dataset results reproduced from saved predictions, including validation
threshold selection and 2000-resample confidence intervals. See
[task5b_status.md](task5b_status.md) and
[task5b_audit_20261106.json](task5b_audit_20261106.json). The dated implementation
and preparation notes below describe earlier checkpoints, not current completion.

Branch: `experiment/task-5b-comparison-methods`, based on `f0f242f`.
These checks validate execution and protocol safeguards. They are **not full
benchmark results** and provide no evidence that one research method wins.

This document records the implementation-stage checks. A later full SAL run
completed successfully and is documented in [task5b_sal_results.md](task5b_sal_results.md).

## Update: current-gold FWS preparation, 2026-11-06

The uploaded corpus files were imported from `85523b4` as `a030d0d` without
merging unrelated manuscript/experiment changes. All 4462 positive labels aligned
with the reconstructed arXiv/SAL sentence indices. The full corpus split is
162/34/34 papers; train reconstructs 45777 sentences, including negatives.

The new `fws_current_gold_v1` preparation passed: FWS train 7924 papers / 57198
sentences / 7985 positives; validation 880 / 6190 / 804; test 209 / 1506 / 220.
Only current gold TRAIN sentences provide exclusions. All FWS inputs are
lowercased to eliminate the source casing artifact. Data and membership are
locked under `data/task5b/prepared-current-gold-v1/fws/`.

All **17 offline tests** passed, including new checks for reconstructing unlisted
negatives and verifying that validation-only gold text is not used for FWS
exclusions. A real FWS smoke run completed TF-IDF/LR, binary NB, train-only
chi2/NB and pinned ModernBERT GPU inference, with frozen thresholds, saved
predictions, 200-resample paper CIs and a paired comparison:

```powershell
python scripts/bench/task5b.py sanity --dataset fws --data-dir data/task5b/prepared-current-gold-v1 --output-dir runs/task5b/fws-current-gold-smoke-20261106 --with-nli --train-papers 24 --dev-papers 24 --batch-size 4
```

Smoke subsets: 24 train papers / 208 sentences / 40 positives; 12 validation
papers / 119 sentences / 11 positives; 12 separate validation papers / 102
temporary evaluation sentences / 19 positives. The official FWS test JSONL was
not opened or scored by the smoke command. These are execution checks, not paper
results. Full FWS scoring remains for the user to start with the
[FWS launcher](task5b_fws_current_gold.md). No SAL result artifact was changed.

The older missing-input statements below describe the original 2026-11-05 state;
current-gold FWS no longer requires the old `scibert_prep/train.jsonl` file, and
the arXiv text/label/split inputs have now been restored. Paragraph structure,
complete funnel predictions and secondary augmentation/lemmatization artifacts
remain separate requirements where applicable.

## Passed

- **15 offline tests**, command:
  `python -m unittest discover -s tests -p test_task5b.py -v`.
  Coverage includes paper leakage/count checks, strict label parsing, structure
  identity, LaTeX-to-original-text section alignment, validation threshold ties,
  NLI label permutation/neutral handling, rule filters/paragraph reset,
  train-only vocabulary fitting, artifact tampering, test-rescore prevention,
  paired identity and truncated-export rejection, external-prediction import,
  both diagnostic CV variants, and neural weight/training round trips.
- **Real SAL train/dev GPU smoke**, command:

  ```powershell
  python scripts/bench/task5b.py sanity --output-dir runs/task5b/sanity-20261105-final --with-nli --train-papers 4 --dev-papers 4 --batch-size 4
  ```

  Training: 4 papers / 831 sentences / 24 positives. Validation: 2 original dev
  papers / 351 sentences / 9 positives. Temporary smoke evaluation: 2 other dev
  papers / 288 sentences / 5 positives. All paper groups are disjoint.
  TF-IDF/LR, binary NB, section-proxy rules and the **actual pinned ModernBERT**
  completed fit/validation, freeze, scoring, prediction export, paper-cluster CI,
  report generation and paired comparison. No official SAL test predictions were
  generated. Every manifest is marked `mode=smoke`; 200 bootstrap replicates were
  used for this small execution check. The tiny training sample gives weak scores
  for several methods; they must not be reported as benchmark findings.
- **Neural serialization/training checks:** a locally initialized tiny BERT was
  trained for one CPU epoch, saved as safetensors and reloaded for finite binary
  probabilities. A separate tiny-model test checks tensor-wise loading against
  original weights and logits with zero tolerance. These exercise the encoder
  code path; they do not validate full SciBERT/BiomedBERT training performance.
- **Full SAL data preparation:** verified 120/40/40 papers; 25676/8650/7968 rows;
  567/180/205 positives. Confirmed 914 negative raw-test omissions and unchanged
  canonical cached test texts/indices. Public data commits:
  SAL `76f8febad929739cb35a2912117f4812ea8c88a9`,
  FWS `8628e7c4b654ea9588b78290510135ee534d8c76`.
- `git diff --check` passed. Manuscript and previous experiment result artifacts
  were not edited.

## Runtime issues found and fixed

1. ModernBERT default reference compilation attempted to use missing Triton on
   Windows. The runner now uses `reference_compile=False` and eager attention.
2. The installed stack also produced native access violations while slicing
   large memory-mapped safetensors storage. Process isolation alone did not fix
   it. Windows loading now decodes/copies one tensor at a time with the official
   safetensors parser. Complete final smoke execution passed after this change.
   Neural subprocesses additionally release CUDA state between operations.

Failed smoke attempts are preserved under `runs/task5b/smoke-20261105*` as
diagnostics; they are incomplete and must not enter research result tables.
The completed final run is `runs/task5b/sanity-20261105-final/` and includes
`run/test_complete.json`, `run/report.md` and `paired_smoke.json`.

Tested environment: Windows, Python 3.12, torch 2.8.0+cu129, transformers 4.53.3,
safetensors 0.6.2, NVIDIA RTX 3060 Laptop GPU (6 GB). Exact package/source/input
hashes are preserved in each run's `run.json` / manifests / frozen settings.

## Still unvalidated / unavailable

- No full official benchmark was scored during implementation.
- Exact FWS 209-paper and arXiv 28-paper runs await the private inputs listed in
  [the runbook](task5b_runbook.md#2-collect-missing-private-artifacts-from-the-team).
  Their adapters have offline checks, but full-corpus alignment still must pass.
- Paragraph-faithful rule execution awaits complete aligned structure. The tested
  proxy is explicitly an approximation of the original structural method.
- Full SciBERT/BiomedBERT checkpoint downloads and multi-epoch training were not
  performed. Tiny-model smoke validation is not a replacement for those runs.
- Original Lan EDA/PBA training code/augmentation provenance is missing. Only an
  external prediction-import path is ready; no reproduction result is claimed.
- Zhang's original aligned lemmatized corpus is missing. The diagnostic code was
  tested on a synthetic fixture; historical .907/.825 results are not reproduced
  or asserted here.
- Complete funnel predictions are missing for the proposed paired comparisons;
  their confidence intervals remain N/A until those artifacts are imported.
