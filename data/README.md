# `data/` — benchmark inputs and outputs

This directory holds every benchmark we built around the Gap2Idea pipeline,
together with the artefacts each one produced. Code lives in
[`src/gap2idea/pipeline/`](../src/gap2idea/pipeline/); the entry points are
exposed as `gap2idea …` CLI subcommands (defined in
[`src/gap2idea/cli.py`](../src/gap2idea/cli.py)).

## What lives where

| Subdirectory | Built by | One-line summary |
|---|---|---|
| `bench/` | `gap2idea bench-extraction` (v1 baseline) | Extraction-quality bench, **N=10**, plain text only |
| `bench_v2a/` | `gap2idea bench-extraction --skip-llm` (v2 vocab only) | Same 10 papers, expanded keyword vocab on plain text |
| `bench_v2b/` | `gap2idea bench-extraction --use-pdf` | Same 10 papers, PyMuPDF style-aware extraction |
| `bench_ablation.md`, `bench_ablation_plots/` | `scripts/compare_extraction_bench.py` + `gap2idea bench-ablation-plots` | v1 vs v2a vs v2b + oracle ablation tables and figures |
| `bench_n100/` | `gap2idea bench-extraction --n 100 --use-pdf` | Full extraction bench at **N=100** with the production v2 pipeline |
| `clustering_bench/` | `gap2idea bench-clustering` (initial smoke run) | Plumbing-verification clustering bench on the N=10 gap corpus |
| `clustering_bench_n100/` | `gap2idea bench-clustering --gaps-tsv data/bench_n100/gaps.tsv` | Real clustering bench on the **161-gap, N=100-paper** corpus |
| `bench/raw/` (gitignored) | manual download | unarXive 2023 open-subset tarball (~4.8 GB) — the gold reference |
| `bench_v2b/pdfs/`, `bench_n100/pdfs/` (gitignored) | bench downloader | Per-paper arxiv PDFs |
| `pdf_test/` (gitignored) | manual | Single-paper PDF used by `scripts/test_extraction_v2.py` |

---

## The benchmarks

There are **two independent benchmarks** in the codebase.

### 1. Extraction-quality bench

Question it answers: *does our pipeline find the same future-work / limitations
content that the paper's authors themselves labelled?*

- Gold: author-titled sections (`Future Work`, `Open Problems`, `Limitations`,
  etc.) from the [unarXive 2023 open subset](https://zenodo.org/records/7752615).
- Stage-1 metric: ROUGE-1/2/L between our extracted section and the gold span.
- Stage-2 metric: cosine-similarity-based **recovery** of LLM gap sentences
  against gold-section sentences (`recovery_at_τ`) and a hallucination check
  against the full paper (`hallucination_at_τ`).
- Optional oracle: feed the gold section *directly* to the LLM and compare
  gap-to-gap to bound how good Stage 2 can possibly be.

### 2. Clustering-quality bench

Question it answers: *which (clusterer × embedding-backbone) combo produces
meaningful, stable, coherent groupings of the gap statements our pipeline
emits?*

- Grid: 5 clusterers × 4 sentence-transformer embedders.
- Intrinsic geometry: silhouette (cosine), Davies-Bouldin, Calinski-Harabasz.
- Topic coherence: **NPMI** computed inline from the input texts
  (gensim's `CoherenceModel` does not compile on Python 3.14, hence the
  inline implementation).
- Stability: **bootstrap-resampled mean Adjusted Rand Index** over 10
  resamples (80 % with replacement) versus the full-corpus labels.

---

## How to reproduce, end to end

Prerequisites (one-time):

```powershell
# Set up the venv
uv sync                                   # or: pip install -r requirements.txt
uv pip install bertopic                   # clustering bench dep, not in base reqs

# OpenRouter key for the LLM gap-extraction stage
copy .env.example .env
# add OPENROUTER_API_KEY=sk-or-... to .env

# Download the unarXive tarball used as the gold reference (~4.8 GB)
mkdir data\bench\raw
curl -L -o data\bench\raw\unarxive_open_subset.tar.xz `
  https://zenodo.org/records/7752615/files/unarXive_230324_open_subset.tar.xz
```

### Extraction-quality bench (full pipeline + ablation)

```powershell
# v1 baseline — plain-text path (no PDFs)
gap2idea bench-extraction --n 10 --out-dir data/bench

# v2b — PDF style-aware path (downloads arxiv PDFs)
gap2idea bench-extraction --n 10 --use-pdf --out-dir data/bench_v2b

# Optional: oracle comparison (feeds gold section straight to the LLM)
gap2idea bench-extraction --n 10 --use-pdf --oracle --out-dir data/bench_v2b

# Cross-variant ablation table + plots
python scripts/compare_extraction_bench.py        # writes data/bench_ablation.md
gap2idea bench-ablation-plots                     # writes data/bench_ablation_plots/

# Larger run (100 papers, ~10 min, ~$1 OpenRouter)
gap2idea bench-extraction --n 100 --use-pdf --out-dir data/bench_n100
```

Each bench writes the same artefact set into its `--out-dir`:

- `bench_papers.jsonl` — sampled unarXive records (gold sections + full text)
- `paper_texts.jsonl` — what we fed our pipeline (text and, in PDF mode, `blocks`)
- `sections_extracted.jsonl` — Stage-1 output
- `gaps.tsv` — Stage-2 verbatim gap sentences
- `metrics.tsv` — long-format per-paper metric table
- `REPORT.md` — aggregate mean ± std summary
- `plots/` — matplotlib PNGs

### Clustering-quality bench

```powershell
# Run on the 161-gap N=100 corpus
gap2idea bench-clustering `
  --gaps-tsv data/bench_n100/gaps.tsv `
  --out-dir data/clustering_bench_n100

# Or specify a subset of clusterers / embedders
gap2idea bench-clustering `
  --gaps-tsv data/bench_n100/gaps.tsv `
  --clusterers kmeans,agglomerative,hdbscan_umap `
  --embedders BAAI/bge-small-en-v1.5,all-mpnet-base-v2 `
  --n-bootstrap 10
```

Bench outputs land in `--out-dir`:

- `metrics.tsv` — one row per `(clusterer, embedder, metric)`
- `REPORT.md` — per-metric pivot tables
- `embeddings/` (gitignored) — cached sentence embeddings per backbone
- `plots/` — heatmap grid, stability bars, silhouette-vs-NPMI scatter

For the **2-D cluster scatters** (UMAP projection coloured by cluster):

```powershell
gap2idea bench-clustering-plots `
  --bench-dir data/clustering_bench_n100 `
  --gaps-tsv data/bench_n100/gaps.tsv `
  --showcase hdbscan_umap:BAAI/bge-small-en-v1.5
```

This regenerates everything plus the showcase cluster plot and a side-by-side
"all clusterers on one embedder" comparison grid.

---

## Headline numbers (latest results checked in)

### Extraction bench, N=10 papers (Stage-1 ROUGE-1 F)

| variant | rouge1_f | what it tests |
|---|---:|---|
| v1 (text + old regex) | 0.246 | baseline |
| v2a (text + new vocab) | 0.246 | vocab alone doesn't help on flat text |
| **v2b (PDF + style + new vocab)** | **0.459** | the real production improvement |

End-to-end Stage-2 recovery@τ=0.6: **0.40 → 0.65** moving from v1 to v2b.
Full ablation in [`bench_ablation.md`](bench_ablation.md).

### Clustering bench, N=161 gap sentences

Best per metric (after dropping NaN rows in the heatmap):

| metric | best cell | value |
|---|---|---:|
| silhouette (cosine) | hdbscan_umap × MiniLM | 0.13 |
| NPMI (topic coherence) | agglomerative × MiniLM | 0.59 |
| bootstrap mean ARI (stability) | kmeans × bge | 0.36 |

Honest caveat written up in
[`clustering_bench_n100/ABLATION_NOTE.md`](clustering_bench_n100/ABLATION_NOTE.md):
on this heterogeneous arXiv-wide corpus, NPMI is decent but silhouette and
stability are weak — typical of clustering ~150 short text snippets in
384–768d embeddings across an unfiltered subject mix.

---

## Things that are intentionally NOT in this directory

- The 4.8 GB unarXive tarball (`bench/raw/`)
- Downloaded arxiv PDFs (`*/pdfs/`)
- Cached sentence embeddings (`*/embeddings/`)
- The local `.env` with the OpenRouter key

All gitignored via [`.gitignore`](../.gitignore). Re-running the bench commands
above regenerates everything that is checked in.
