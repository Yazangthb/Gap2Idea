# Gap2Idea — minimal (experiments + pipeline)

A trimmed branch of Gap2Idea containing **only** the research-gap extraction **experiments**
(the Stage A/B/C notebooks) and the **end-to-end pipeline** that runs them over a list of papers.

```
paper(s) → Stage A (localize) → Stage B (classify) → Stage C (LLM filter) → gaps
```

## Layout
```
scripts/run_extraction_pipeline.py   # the full pipeline (list of papers → gaps.tsv/json)
src/gap2idea/pipeline/               # the funnel code the pipeline + notebooks import
  gap_funnel.py      # Stage A (slice_section_aware, pdf_to_blocks) + Stage B (cue rules, EmbeddingGapHead)
  gap_llm_filter.py  # Stage C (LLMGapFilter — batched LLM precision filter)
  llm.py             # LLM client (Yandex / OpenRouter, OpenAI-compatible)
  sections.py, gap_prefilter.py
stage_a_package/     # Stage A notebooks (localization): section-aware slice + PDF front-end
stage_b_package/     # Stage B notebooks (classifier): cue rules + encoder head, model comparison
stage_c_package/     # Stage C notebooks (LLM filter) + the end-to-end funnel on real papers
```

## The pipeline
Takes a **list of papers** (PDFs or a texts `jsonl`) and runs the whole funnel, using the exact
components validated in the notebooks (`pdf_to_blocks` + `slice_section_aware` for Stage A, cue rules +
frozen `bge-small`+logreg head for Stage B, a Yandex LLM for Stage C).

```bash
# PDFs in a folder, Stage A+B only
python scripts/run_extraction_pipeline.py --pdfs data/pdfs --out gaps.tsv

# full funnel with Stage C (biomedical prompt)
python scripts/run_extraction_pipeline.py --pdfs data/pdfs --out gaps.tsv --stage-c --domain biomed

# from a texts jsonl ({"id","text"[,"blocks"]})
python scripts/run_extraction_pipeline.py --texts paper_texts.jsonl --out gaps.tsv --stage-c
```
Output columns: `id, gap_type, gap_sentence, section_type, source, confidence, paragraph_text`.

## Install
```bash
pip install -e .          # or: pip install -r requirements.txt
```
Python 3.10+. First Stage-B run trains a `bge-small`+logreg head from `data/scibert_prep/train.jsonl`
(~2 min, cached to `data/pipeline_head.joblib`).

## Data (not included)
The notebooks and the Stage-B head expect labeled data under `data/` (e.g.
`data/scibert_prep/train.jsonl`, gold sets, SAL/FWS/Sci-Sentence corpora). These are **not** shipped in
this minimal branch. Public corpora are fetched on demand by the benchmarks (SAL/FWS from GitHub, BioC
sections from the PMC API); the self-distilled training set must be supplied under `data/`.

## Stage C (LLM filter)
Set credentials in a `.env` (see `.env.example`):
```
LLM_PROVIDER=yandex
YANDEX_API_KEY=...
YANDEX_FOLDER_ID=...
```
Default model is `yandexgpt-5-lite` @ batch 20 (the probe winner). Without `--stage-c` the pipeline
stops after Stage A+B and needs no API.

## Key results (from the notebooks)
- **Stage A**: keeping Discussion+Conclusion retains **99%** of limitations on real SAL papers (0.861→0.988 on SAL, 5-fold CV).
- **Stage B**: fine-tuned PubMedBERT best in-domain (SAL full-funnel A+B **0.604**); frozen bge generalizes best cross-domain.
- **Stage C**: lifts precision ~0.30→0.60; full funnel **PubMedBERT A+B+C 0.701**. Precision vs SAL gold is a documented lower bound (span-based annotation under-labels; see `stage_c_package/stage_c_end_to_end_funnel.ipynb`).
