# Task 5b: server handoff

Branch: **Task_5b**. The completed SAL/FWS baselines and three completed arXiv
jobs are published with full evidence. Large trained checkpoints and exact frozen
prepared datasets are transported in checksum-verified, sub-100-MiB archive
parts under `artifacts/task5b/server-handoff-20261106/`. No Git LFS setup or manual
private-data transfer is required for the two remaining data-ready jobs.

## Completed work to retain

- SAL: four baselines, `runs/task5b/sal-full-01/sal/`.
- Updated FWS: five baselines, `runs/task5b/fws-current-gold-full-01/fws/`.
- arXiv classical/NLI: four methods, `runs/task5b/remaining-v1/arxiv-baselines/arxiv/`.
- arXiv SciBERT: `runs/task5b/remaining-v1/arxiv-scibert/arxiv/`.
- arXiv BiomedBERT: `runs/task5b/remaining-v1/arxiv-biomedbert/arxiv/`.

The server script verifies these 15 completed method/dataset results and never
trains or scores them again. JSON/JSONL, scores, classical models, metadata and
reports are ordinary tracked run files. The two large encoder `.safetensors`
files are restored to their original paths from the archive parts. Original
Windows manifests retain their historical paths; the verifier normalises only
relative artifact separators when locating files, never rewrites hashed evidence.

The two arXiv encoder runs are completed **negative/degenerate results**: their
validation probabilities are nearly constant (SciBERT about .00236; BiomedBERT
about .00286), validation-selected thresholds are .5, and positive test F1 is 0.
They are preserved rather than hidden. They need validation-side training
diagnostics before being treated as useful competitive encoder baselines; this
handoff does not tune or rerun their already-scored tests. The generic binary
fine-tune differs from the original authors' class-weighted recipe. Do not infer
that these results establish that the model families are intrinsically weak.

## What will run on the server

1. **FWS SciBERT**, all three epochs from the pinned initialisation.
2. **Cached SAL SciBERT/PubMedBERT evaluation**, preserving original binary
   decisions and disclosing unverified original training/threshold history.

The stopped FWS trainer saved no model/optimizer checkpoint before interruption.
Its batch/epoch progress cannot be recovered. Only this unfinished training stage
must restart; completed arXiv work is retained. The local partial folder is
preserved, excluded from the published completed package, and not copied into the
server's new run root. The current trainer checkpoints the final model, not
mid-epoch optimizer state, so run it in `tmux` or with `nohup` to survive SSH
disconnects.

## Setup and run (Linux server)

Use Python 3.12 and a working CUDA-enabled PyTorch >=2.6 installation compatible
with the server. The requirements file keeps the Transformers major version
below 5 and records actual versions in each run; it does not configure GPU drivers.

```bash
git clone --branch Task_5b https://github.com/Yazangthb/Gap2Idea.git
cd Gap2Idea
# For an existing clone on this branch: git pull --ff-only origin Task_5b

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-task5b.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Restore exact data/checkpoints and verify completed evidence; no training:
python scripts/bench/run_task5b_server.py --check

# Run only the unfinished stages:
python -u scripts/bench/run_task5b_server.py --run
```

Do not use `--run` with a CPU-only installation unless you intentionally accept
slow CPU training. The default encoder batch is 1, preserving the registered
local remaining-run batch size. On a larger GPU a different batch size is allowed
only as a separately recorded resource/training configuration:

```bash
python -u scripts/bench/run_task5b_server.py --run --encoder-batch-size 8 --run-root runs/task5b/server-largegpu-v1
```

To keep the default run alive after logout:

```bash
nohup python -u scripts/bench/run_task5b_server.py --run > task5b-server.log 2>&1 &
tail -f task5b-server.log
```

New output is separate from the old Windows suite:

```text
runs/task5b/server-v1/fws-scibert/fws/report.md
runs/task5b/server-v1/sal-cached-encoders/legacy_scibert_ft/metrics.json
runs/task5b/server-v1/sal-cached-encoders/legacy_pubmedbert_ft/metrics.json
```

The exact prepared data are restored from the archive; do not regenerate a
different FWS split or copy the obsolete `scibert_prep/train.jsonl` into this run.
Linux line endings otherwise change byte hashes, which is why the frozen files
are archived and published evidence has explicit Git attributes.

Restoration refuses to overwrite any existing file whose hash differs. Completed
server stages are verified and skipped. An incomplete server run needs inspection
and a new `--run-root` for a fresh attempt; it is not silently overwritten.

## Packaging and reproducibility

`build_task5b_handoff.py` enumerates only completed runs and the three prepared
datasets. Its manifest records each archive part, complete archive and restored
file SHA256/size. Download caches, `.env`, live/incomplete FWS training and failed
smoke directories are excluded. The server installer validates member paths and
hashes and extracts ordinary files only, with no symlink/path traversal support.
Allow several GB for the clone, restored checkpoints/data and new encoder caches.

The published trained checkpoints derive from the pinned SciBERT and BiomedBERT
models documented in `baseline_methods.md`; preserve their original model-card
and licence attribution when redistributing or reusing them. Their provenance
and base revisions remain in each frozen-settings file.

## Still outside this two-job handoff

Finishing these jobs completes the data-ready local run list, not every original
proposal item. Exact Lan EDA/PBA reproduction, original Zhang lemmatised-CV
reproduction, independent rule structure and matched full funnel predictions
still need the external files listed in `task5b_remaining_runs.md`. Do not replace
those missing comparisons with historical aggregate scores.
