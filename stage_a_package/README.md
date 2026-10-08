# Stage A package

Organized deliverables for **Stage A** (the gap-extraction *localization* step) plus the corrected project audit.

## Files

| File | What it is |
|---|---|
| `stage_a_all_experiments.ipynb` | Runs **every Stage-A version** on the cached gold and shows the improvement: v1 (`slice_terminal_regions`) → v2 (`slice_with_midpaper_anchors`), with a comparison table, charts, and the list of gap sentences v2 rescues. |
| `stage_a_final.ipynb` | The **final Stage A** only (`slice_with_midpaper_anchors`): a clean reusable `run_stage_a()` step, a demo of the regions it keeps, and its final gold-set numbers. |
| `AUDIT_CORRECTED.md` | One-page corrected summary of the 5-Oct-2026 project audit (what checks out, what was wrong). |

## How to run the notebooks

They run **offline, CPU-only, no API keys**. Open either notebook and Run All.
Each one auto-detects the repo root (walks up to `pyproject.toml`), so it reads the cached data from the
parent repo no matter where this folder sits inside it.

**Data used (all cached in the repo):**
- `data/scibert_prep/gold_papers.jsonl` — 10 papers (full text + layout blocks), from arXiv / unarXive 2023 open subset (Zenodo 7752615).
- `data/bench_gap/gold_sentences_v2.tsv` — 49 gold gap sentences / 10 papers (built with GPT-4o, then verified — *silver*, not human gold).

## Result in one line

Adding mid-paper scope/assumption anchors (v1 → v2) raised localization@0.70 from **0.633 to 0.857**
while still dropping ~**84%** of each paper. (10-paper gold set — small-sample, good for the trend, not a large-scale claim.)
