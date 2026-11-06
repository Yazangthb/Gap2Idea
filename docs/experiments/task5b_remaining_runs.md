# Task 5b: remaining run plan

Prepared 2026-11-06. Keep the completed SAL and updated FWS runs. Run the jobs
below locally; **nothing in this launcher pushes Git**. The complete proposal
still requires several external artifacts, listed after the runnable jobs.

## Run everything currently supported by our data

```powershell
Set-Location 'E:\vs\Gap2Idea - main code\Gap2Idea'
python scripts/bench/run_task5b_remaining.py --check
python scripts/bench/run_task5b_remaining.py --run
```

The first command checks data readiness without downloading models or scoring
tests. The second executes these independent jobs sequentially:

| Job / `--stage` | What it produces | Data | Status before running |
|---|---|---|---|
| `arxiv-baselines` | TF-IDF/LR, binary NB, broad NLI, typed NLI | Prepared 134/28/28-paper arXiv split | Ready |
| `arxiv-scibert` | Full in-domain binary SciBERT fine-tune | Same arXiv split | Ready; GPU training |
| `arxiv-biomedbert` | Full in-domain binary BiomedBERT fine-tune | Same arXiv split | Ready; GPU training |
| `fws-scibert` | Full in-domain binary SciBERT fine-tune | Frozen `fws_current_gold_v1` split | Ready; GPU training |
| `sal-cached-encoders` | P/R/F1, paper CIs and canonical predictions for the saved SciBERT/PubMedBERT vectors | Existing SAL fixture and frozen SAL test | Ready; CPU import/evaluation only |

No SAL/FWS method already completed is retrained. The cached SAL rows are named
`legacy_scibert_ft` and `legacy_pubmedbert_ft`: original binary decisions are
preserved, and unverified original training/threshold history is disclosed. These
rows are full-stream encoders, **not section-gated A+B**, new controlled fine-tunes,
or the Lan EDA/PBA method. Do not relabel them as those methods.

Results go under:

```text
runs/task5b/remaining-v1/
  suite_plan.json
  arxiv-baselines/arxiv/report.md
  arxiv-scibert/arxiv/report.md
  arxiv-biomedbert/arxiv/report.md
  fws-scibert/fws/report.md
  sal-cached-encoders/legacy_scibert_ft/metrics.json
  sal-cached-encoders/legacy_pubmedbert_ft/metrics.json
```

Every trained method saves model weights where applicable, validation/test
predictions, frozen settings, metrics, paper CIs and run manifests. The cached
SAL import stores external provenance and prediction hashes. Keep entire run
directories for the final audit and later publication.

## Run one job at a time

```powershell
# Start here if you want the arXiv baseline results first:
python scripts/bench/run_task5b_remaining.py --run --stage arxiv-baselines

python scripts/bench/run_task5b_remaining.py --run --stage arxiv-scibert
python scripts/bench/run_task5b_remaining.py --run --stage arxiv-biomedbert
python scripts/bench/run_task5b_remaining.py --run --stage fws-scibert
python scripts/bench/run_task5b_remaining.py --run --stage sal-cached-encoders
```

Completed jobs are skipped only after checking their saved evidence and frozen
data. Incomplete directories are not overwritten or automatically rescored. If a
job fails, preserve its log/directory, fix the cause, and retry that specific job
with a new suite root, for example:

```powershell
python scripts/bench/run_task5b_remaining.py --run --stage fws-scibert --run-root runs/task5b/remaining-retry-01
```

`suite_plan.json` freezes source hashes, launcher hash, jobs, model revisions,
training settings and resource options before any stage starts. Do not edit the
experiment source or change those options between jobs in the same suite. Use a
new root for a deliberately changed protocol; do not choose changes from test
scores. Repeating `--run` is safe for completed, unchanged jobs and stops on an
incomplete job instead of repeating its test pass.

## Encoder protocol and resource use

Encoder jobs are separate full fine-tunes, not faithful reproductions of an
unreleased author checkpoint. All layers train for 3 fixed epochs with AdamW,
learning rate 2e-5, weight decay .01, clip norm 1, seed 0, max length 256; the final
epoch is used and validation selects the decision threshold. The launcher pins:

- SciBERT: `24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1`.
- BiomedBERT: `e1354b7a3a09615f6aba48dfad4b7a613eef7062`.

For the local 6 GB GPU, the launcher defaults to **encoder batch size 1** with
CUDA fp16 autocast/GradScaler, fp32 parameters, gradient checkpointing and
non-foreach AdamW. NLI baseline inference remains batch size 4. CPU uses fp32.
These choices are registered before full encoder test scoring. Training progress
is printed every 100 batches. Full fine-tunes take substantially longer than the
classical baselines; let them finish and avoid concurrent GPU jobs.

To use a different resource configuration on a larger GPU, choose a fresh root:

```powershell
python scripts/bench/run_task5b_remaining.py --run --encoder-batch-size 8 --run-root runs/task5b/remaining-largegpu-01
```

Record the batch-size change as a different configuration. `--device cpu` is
supported if needed, but full training is much slower.

The system-drive Hugging Face cache was nearly full during preparation. New
encoder downloads and a seeded safetensors initialization cache now live under
`data/task5b/hf_cache/` on E:. CPU preparation runs in its own process; GPU
training loads the resulting safetensors initialization, avoiding retained .bin
storage/mapping memory. Existing NLI cache and old experiment results are not
cleared or moved. Allow several GB for the two base models, initialization caches
and three final checkpoints. Only model checkpoints are saved, not giant optimizer
states. No automatic cache cleanup or result deletion is performed.

## Funnel comparisons: request the missing decisions

Label-free exact test inputs have been prepared under:

```text
data/task5b/funnel_requests_v1/sal/
data/task5b/funnel_requests_v1/fws/
data/task5b/funnel_requests_v1/arxiv/
```

Each contains `sentences.jsonl` (all test rows, original IDs/hashes and context),
`request.json`, and `protocol_TEMPLATE.json`. Send these folders to the teammate
who owns the final frozen funnel. Ask for:

1. One prediction for **every** supplied sentence, including gate-rejected
   negatives: `paper_id`, `idx`, `text_sha256`, integer `y_pred` (0/1).
2. A completed `protocol.json` with actual checkpoint/training data, validation
   threshold/prompt-selection history, Stage-C model/prompt revision and known
   limitations. Do not submit the unfilled template as real provenance.
3. Confirmation that the frozen funnel used these exact inputs. SAL includes both
   raw `section` and canonical `coarse_section`; FWS inputs are lowercased. Never
   infer predictions from an aggregate F1 or combine an unrelated candidate mask
   with a verdict file.

The exporter can be rerun to a new folder if needed, without disclosing labels:

```powershell
python scripts/bench/run_task5b_remaining.py --export-reference-requests data/task5b/funnel_requests_v2
```

Once the files return, import and pair them. FWS example:

```powershell
python scripts/bench/task5b.py score-external --dataset fws --data-dir data/task5b/prepared-current-gold-v1 --method frozen_funnel_fws --target future_work --predictions data/task5b/funnel_requests_v1/fws/predictions.jsonl --protocol data/task5b/funnel_requests_v1/fws/protocol.json --output-dir runs/task5b/funnel-fws-v1
python scripts/bench/task5b.py compare --predictions runs/task5b/fws-current-gold-full-01/fws/tfidf_lr/test_predictions.jsonl --reference runs/task5b/funnel-fws-v1/test_predictions.jsonl --output runs/task5b/paired-fws-tfidf-vs-funnel.json
```

Repeat the comparison for every attempted method, not only the best result.
For SAL use `--data-dir data/task5b/prepared --dataset sal --target limitation`.
For arXiv use `--data-dir data/task5b/prepared-current-gold-v1 --dataset arxiv
--target any_gap`. A separate limitation-only reference evaluation is needed for
arXiv rule scores; their gold target differs from any-gap detection.

## Other proposal items still requiring teammate artifacts

| Item | Ask the team for | What to do next |
|---|---|---|
| Lan 2024 EDA/PBA | `h100_allmodels.py` or original complete recipe, augmentation implementation/config and train-only augmented data provenance | This reproduction is not implemented by a generic BiomedBERT fine-tune. Once the recipe is available, implement/validate it and import its frozen predictions. Otherwise disclose inability and a qualified closest baseline |
| Zhang original balanced CV | Original aligned lemmatized text and exact sampling/preprocessing/fold details | Existing `audit-zhang` saves both fold-local and deliberately global-fitted diagnostic results; it is not the current held-out adaptation or a certificate of exact .907/.825 reproduction |
| arXiv rule extension | `data/gold/arxiv_latex_blocks.jsonl` for the original flat texts, including explicit empty status for unavailable papers | Prepare a **new** section-aware arXiv directory, then run `kilicoglu_section_proxy` as a limitation-only extension |
| Structurally faithful SAL rules | Exhaustive independent paragraph/top-section metadata at `data/task5b/structure/sal.jsonl`, with original paper/index/text hashes | Prepare a **new** SAL directory and run `kilicoglu_rules`. Current section proxy stays a separately named approximation |

Conditional commands (only after the named files arrive):

```powershell
# arXiv LaTeX-section extension:
python scripts/bench/task5b.py prepare --datasets arxiv --output-dir data/task5b/prepared-arxiv-sections-v1
python scripts/bench/task5b.py run --datasets arxiv --data-dir data/task5b/prepared-arxiv-sections-v1 --methods kilicoglu_section_proxy --run-dir runs/task5b/arxiv-section-proxy-v1

# SAL rules with complete independent structure:
python scripts/bench/task5b.py prepare --datasets sal --output-dir data/task5b/prepared-sal-structure-v1
python scripts/bench/task5b.py run --datasets sal --data-dir data/task5b/prepared-sal-structure-v1 --methods kilicoglu_rules --run-dir runs/task5b/sal-structured-rules-v1

# Separate Zhang CV diagnostic; do not add its leaky score to the held-out table:
python scripts/bench/task5b.py audit-zhang --lemmatized-jsonl data/task5b/fws_lemmas.jsonl --folds 10 --output-dir runs/task5b/zhang-diagnostic-01
```

Lemmas must be JSONL with `raw_idx`, `text_sha256` of each original FWS CSV text,
and `lemmatized_text`, in full original CSV order. Section/paragraph annotations
must cover every canonical train/val/test row and must not be inferred from
positive-only gold section fields. See the main runbook for their exact schema.

## When the runs finish

Ask for another completion audit. We should merge all new rows into the unified
table, verify complete model/prediction/settings evidence and paired differences,
then push the full result package together. The five runnable jobs alone do not
resolve the external-artifact requirements; record any unreproducible methods and
explicit agreed scope before calling the entire proposal complete.
