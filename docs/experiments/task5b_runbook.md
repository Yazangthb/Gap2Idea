# Task 5b: runbook and evidence checklist

Implementation of `Task_5b_Comparison_methods.pdf`, based on the current
minimal-derived checkout. Start with methods 1-3; optional fine-tunes and Zhang
diagnostics use the same locked-data interfaces. The manuscript is unchanged.

## Status and scope

| Proposal item | Code available | What is needed for a full run |
|---|---|---|
| 1: TF-IDF + LR; binary ngram + NB | `tfidf_lr`, `binary_nb` | Prepared SAL/FWS/arXiv datasets |
| 2: Kilicoglu section-gated rules | `kilicoglu_rules`; separately named `kilicoglu_section_proxy` approximation | Original aligned paragraph/section metadata for structural mode; SAL section_name suffices for proxy; arXiv proxy additionally needs LaTeX blocks + original flat texts; not a FWS method |
| 3: ModernBERT NLI alone | `modernbert_nli`, optional preregistered `modernbert_nli_typed` | Hugging Face model download once; GPU recommended; no LLM/API key |
| 4: In-domain SciBERT / BiomedBERT | `scibert_ft`, `biomedbert_ft`; saved safetensors | Full data and GPU time; generic fine-tunes, not prior checkpoint replicas |
| 5a: Lan PubMedBERT + EDA/PBA | External frozen-prediction adapter only | Original training/augmentation recipe (`h100_allmodels.py` or equivalent), its dependencies/config and train-only augmentation provenance. **Not yet a faithful runnable reproduction** |
| 5b: Zhang NB | `zhang_nb` held-out raw-text adaptation; `audit-zhang` separate CV diagnostic | Full FWS split; original aligned lemmatized texts for diagnostic reproduction. No claim of exact historical numbers |
| Same-paper funnel comparison | `score-external`, `compare` | Complete aligned predictions for each target/dataset, with model/prompt/candidate/threshold provenance. Aggregates alone are insufficient |

Read [baseline_methods.md](baseline_methods.md) for fixed hyperparameters,
threshold/target rules, statistical assumptions and deviations from historical
experiments. Report every attempted/skipped method, including negative results.
See [task5b_validation.md](task5b_validation.md) for the checks actually completed
and the limits of that validation.
The later completed full SAL run and versioned evidence are documented in
[task5b_sal_results.md](task5b_sal_results.md); FWS and arXiv results remain pending.

## 1. Setup (PowerShell)

```powershell
Set-Location 'E:\vs\Gap2Idea - main code\Gap2Idea'
python -m pip install -r requirements-task5b.txt
python scripts/bench/task5b.py --help
python scripts/bench/task5b.py fetch-public --datasets sal,fws
python scripts/bench/task5b.py preflight
```

Use an existing CUDA-enabled PyTorch installation or install the build appropriate
for the machine before installing requirements. This suite does not require the
full pipeline's `sentence-transformers` dependency or any `.env` credentials.
Public downloads resolve a Git commit and record URLs/hashes under
`data/task5b/downloads/`. Existing input files are preserved. ModernBERT's initial
download is approximately 1.6 GB; subsequent loads use the Hugging Face cache.

## 2. Collect missing private artifacts from the team

The public SAL CSVs and FWS CSV can be downloaded by step 1. They are kept under
ignored `data/`, along with generated data, model weights and run outputs.

Ask the experiment owner for these files **with their original revisions**:

1. **FWS selection source:** `data/scibert_prep/train.jsonl` used by
   `stage_c_package/breadth_funnel.py`. It determines the exact excluded training
   texts and therefore the 209-paper test membership. A newly trained/recreated
   file may select a different test set even with the same seed.
2. **arXiv finalized corpus:** `data/gold/gap_gold_final.jsonl`,
   `data/gold/splits.json`, `data/paper_texts.jsonl`. Alternatively supply complete
   original `export_stageb_data.py` exports under
   `data/gold/task5b_arxiv/{train,val,test}.jsonl`; schema is
   `paper_id, idx, sentence, label, domain` (`domain=arxiv_cs` for these papers;
   negatives explicitly `label=none`). Complete exports include **all** sentences,
   not only positive gold rows. Prefer these if segmentation ever changed.
3. **arXiv section recovery:** `data/gold/arxiv_latex_blocks.jsonl`. Its blocks
   must locate in the same original `data/paper_texts.jsonl`; top-level sections
   enable the proxy extension, not paragraph-faithful original rules. Papers with
   empty/unavailable blocks retain all their sentences with an unknown section.
4. **Original paragraph structure (if available):** a sentence-aligned JSONL at
   `data/task5b/structure/sal.jsonl` / `arxiv.jsonl`, one record for **every**
   train/val/test row, with `paper_id, idx, text_sha256, section, top_section,
   section_id, paragraph_id`. Use canonical Task 5b indices and exact UTF-8 text
   hashes, not raw SAL row numbers. For SAL only, an initial `prepare` can create
   the canonical export used to align recovered structure; prepare a **new output
   directory** once the structure is added. Never derive boundaries from labels.
5. **Funnel evidence:** complete 0/1 predictions aligned to canonical test rows
   for SAL Qwen, arXiv Qwen and FWS (Qwen or the intended Yandex reference), plus
   model/prompt versions and candidate definitions. Candidate-only verdicts need
   the full candidate mask and a documented full-corpus reconstruction upstream.
6. **Secondary reproductions:** original SAL EDA/PBA training/augmentation code
   and configuration, plus aligned FWS lemmatization outputs. Neither a headline
   number nor a generic PubMedBERT checkpoint fills this gap.

No files are guessed from similarly named historical splits. Count checks are
necessary but not sufficient evidence: input hashes and explicit saved test
membership are preserved in addition to counts.

## 3. Prepare and lock datasets

```powershell
# SAL is ready locally; already prepared during implementation.
# Run this only on a fresh machine/output directory:
python scripts/bench/task5b.py prepare --datasets sal

# Once private FWS/arXiv artifacts are restored:
python scripts/bench/task5b.py prepare --datasets fws,arxiv
```

The command refuses to overwrite existing prepared data or accept another test
size: SAL **40/7968**, FWS **209/1506**, arXiv **28/10388** (papers/sentences).
Manifests save input hashes, statistics, exact-text overlap counts and prepared
file hashes. All sentence IDs, texts, labels, source indices and test membership
are explicit in `{train,val,test}.jsonl`. arXiv raw reconstruction fails on
positive-text/index drift rather than fuzzy-matching labels. If any count differs,
recover the correct input revision; do not lower the check to get a result.

Existing local SAL preparation verified 120/40/40 disjoint papers, 25676/8650/7968
sentences and 567/180/205 positives. Generic exact sentence strings recur across
papers (train/val 186; train/test 63; val/test 26); the manifest reports this.
No official test predictions were generated as part of the implementation smoke.

## 4. Run the full first-priority comparison

The SAL data is prepared in this checkout. This is the immediate full-test command:

```powershell
python scripts/bench/task5b.py run --datasets sal --methods tfidf_lr,binary_nb,kilicoglu_section_proxy,modernbert_nli --batch-size 4 --bootstrap 2000 --run-dir runs/task5b/sal-full-01
```

Once all three datasets are prepared:

```powershell
python scripts/bench/task5b.py run --datasets sal,fws,arxiv --methods tfidf_lr,binary_nb,kilicoglu_rules,kilicoglu_section_proxy,modernbert_nli --batch-size 4 --bootstrap 2000 --run-dir runs/task5b/primary-full-01
```

Each requested dataset is checked before training starts. Unsupported or
structurally unavailable rules are explicitly skipped with a saved reason. The
arXiv section proxy requires LaTeX-derived sections; it never consumes positive-
only gold sections. Use **a new run directory** for a new configuration. Both
commands fit/lock first, then score the official test once. CPU is supported
(`--device cpu`) but NLI is substantially slower. Lower batch size for GPU OOM;
do not change the method after inspecting test scores.

To inspect frozen settings **before** manually starting test evaluation:

```powershell
python scripts/bench/task5b.py fit --dataset sal --methods tfidf_lr,binary_nb,modernbert_nli --batch-size 4 --run-dir runs/task5b/sal-review-01
# Inspect preregistered_methods.json, fit_status.json and each frozen_settings.json.
python scripts/bench/task5b.py evaluate --run-dir runs/task5b/sal-review-01 --bootstrap 2000
```

Test evaluation refuses a second pass in the same run. If scoring is interrupted,
preserve that directory/log and document the failure before starting a fresh run.
Fit and evaluate must use the same experiment sources/software and unchanged
prepared data/model artifacts. Reports can be regenerated without rescoring:

```powershell
python scripts/bench/task5b.py report --run-dir runs/task5b/primary-full-01/sal
```

## 5. Optional encoders and Zhang adaptation

```powershell
python scripts/bench/task5b.py run --datasets arxiv --methods scibert_ft,biomedbert_ft --epochs 3 --batch-size 4 --run-dir runs/task5b/arxiv-encoders-01
python scripts/bench/task5b.py run --datasets fws --methods scibert_ft,zhang_nb --epochs 3 --batch-size 4 --run-dir runs/task5b/fws-secondary-01
```

SAL in-domain encoder reruns are also supported with `--datasets sal`. Existing
SAL cached encoder decisions can instead be exported and scored externally, but
their original training/threshold provenance must remain attached. They are not
assumed equivalent to new fine-tunes or to a section-gated A+B funnel.

The separate optional Zhang diagnostic expects an aligned lemmatized JSONL with
`raw_idx` (zero-based original FWS CSV row), `text_sha256` of original raw text,
and `lemmatized_text`, for every original CSV row:

```powershell
python scripts/bench/task5b.py audit-zhang --lemmatized-jsonl data/task5b/fws_lemmas.jsonl --folds 10 --output-dir runs/task5b/zhang-diagnostic-01
```

This is a **balanced sentence-CV diagnostic**, not the common held-out test.
Both properly fold-fitted and deliberately global-fitted predictions are saved.
Historical .907/.825 values are not copied into new result files. Exact historical
reproduction additionally needs the original preprocessing/sampling/fold details.

## 6. Import the funnel / Lan / saved encoder predictions and pair them

Frozen external prediction JSONL requires one record for **each** canonical test
row, including rejected/negative sentences:

```json
{"paper_id":"PMC116603","idx":0,"text_sha256":"<canonical SHA256>","y_pred":0}
```

`sentence` with the exact canonical text can replace `text_sha256`. A provenance
JSON must include `method`, `training_data`, `threshold_selection`,
`model_or_code_revision`, and `limitations`. Use a meaningful method ID such as
`qwen_funnel_sal40` and record any previous test-guided selection under limitations.

```powershell
python scripts/bench/task5b.py score-external --dataset sal --method qwen_funnel_sal40 --target limitation --predictions data/task5b/qwen_sal40_full.jsonl --protocol data/task5b/qwen_sal40_protocol.json --output-dir runs/task5b/qwen-sal40-import-01
python scripts/bench/task5b.py compare --predictions runs/task5b/sal-full-01/sal/tfidf_lr/test_predictions.jsonl --reference runs/task5b/qwen-sal40-import-01/test_predictions.jsonl --output runs/task5b/tfidf-minus-qwen-sal40.json
```

`compare` rejects partial, duplicate, different-text or different-gold-target rows.
It reports method-minus-reference deltas and paired paper-bootstrap CIs. The
arXiv limitation-rule target needs a limitation-labelled reference evaluation;
do not compare its F1 directly with any-gap ground truth. Without complete funnel
predictions, paired comparisons remain **N/A**, with no inferred superiority.

## 7. Outputs and smoke checks

Each dataset run produces:

- `run.json`, `input_manifest.json`, `preregistered_methods.json`, `fit_status.json`.
- Per method: checkpoint/joblib where applicable; `validation_predictions.jsonl`,
  `frozen_settings.json`, `test_predictions.jsonl`, `metrics.json`.
- `fit_complete.json`, `test_started.json`, `test_complete.json`, `results.json`,
  `report.md`. Interrupted runs lack a completion marker and are not full results.

Predictions include paper/index, exact text/hash, gold type, binary target, score,
decision, threshold and frozen-settings hash. Results contain P/R/F1/macro F1,
confusion counts, paper CIs and prediction hashes. Preserve entire runs, not just
the Markdown table. `data/` and `runs/task5b/` are ignored by Git; archive/share
them separately with their hashes after checking corpus redistribution terms.

`compare` accepts completed evaluated outputs with their adjacent `metrics.json`
and checks prediction hashes; pass raw team predictions through `score-external`
first. This prevents comparing two equally truncated exports by accident.

```powershell
python -m unittest discover -s tests -p test_task5b.py -v
python scripts/bench/task5b.py sanity --with-nli --train-papers 4 --dev-papers 4 --batch-size 4 --output-dir runs/task5b/smoke-new-id
```

The smoke uses real public SAL **train/dev papers only**. It uses part of dev as a
temporary smoke evaluation pool, labels its manifest `mode=smoke`, and never uses
the official cached test fixture for training or predictions. Its scores are
diagnostics, **not research results**. It exercises the actual pinned ModernBERT
when `--with-nli` is supplied; otherwise it tests classical/proxy methods only.
Fine-tunes and private-corpus end-to-end runs require separate full validation.

People exposed to gold labels while doing this work should not serve as blinded
raters in the separate human-validation study described in the proposal.
