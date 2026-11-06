# Task 5b implementation validation - 2026-11-05

Branch: `experiment/task-5b-comparison-methods`, based on `f0f242f`.
These checks validate execution and protocol safeguards. They are **not full
benchmark results** and provide no evidence that one research method wins.

This document records the implementation-stage checks. A later full SAL run
completed successfully and is documented in [task5b_sal_results.md](task5b_sal_results.md).

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
