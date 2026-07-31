# Gap2Idea

End-to-end pipeline that mines **research gaps** (limitations, future work) from academic
papers, structures them into a graph, and synthesises **novel, evidence-grounded research
ideas** — each novelty-checked against Semantic Scholar and scored by an LLM judge panel.
Runs as a CLI, a Streamlit app, or an MCP server.

## Pipeline

```
select-papers → download-pdfs → extract-text ─► data/paper_texts.jsonl
      │
  gap extraction ─► data/gaps.tsv
      ├─ extract-gaps          LLM per paper (verbatim, strict JSON)        ~$4000 / 1M papers
      └─ extract-gaps-funnel   cheap Stage A/B/C funnel (fw + limitation)   ~$3–31 / 1M papers
      │
theme-mine     ─► gap graph: Leiden communities + bridge & frontier scoring
generate-ideas ─► ideas   (modes: bridge · within · method-gap · frontier · orchestrated)
evaluate-ideas ─► LLM judge panel + Semantic-Scholar novelty
export-ideas   ─► LaTeX / PDF / full-paper draft        serve-mcp ─► MCP server
```

Stages pass plain files (`data/` inputs, `artifacts/` outputs). Every LLM call is an
OpenRouter model slug — swap providers with `--model <provider/model>`.

## Install

```bash
pip install --upgrade uv
uv sync                 # creates .venv with all deps
cp .env.example .env    # then set OPENROUTER_API_KEY (S2_API_KEY recommended)
```

Then `source .venv/bin/activate` (or `.venv/Scripts/activate` on Windows), or prefix
commands with `uv run`. Docker: `docker build -t gap2idea . && docker run --env-file .env -p 8501:8501 gap2idea`.

## Usage

**Already have `data/gaps.tsv`:**
```bash
gap2idea theme-mine --gaps-tsv data/gaps.tsv      # embed → gap graph → bridge/frontier
gap2idea fetch-metadata
gap2idea generate-ideas --n-pairs 10
gap2idea evaluate-ideas --judges "anthropic/claude-sonnet-4,openai/gpt-4o"
gap2idea export-ideas --format library-pdf
streamlit run src/gap2idea/app/streamlit_app.py
```

**From scratch (arXiv):**
```bash
gap2idea select-papers --source s2 --query "graph neural networks" --n 100
gap2idea download-pdfs
gap2idea extract-text
gap2idea extract-gaps-funnel --mode hybrid --head data/gap_head.joblib   # cheap, no per-paper LLM
# → then theme-mine → generate-ideas → evaluate-ideas as above
```

**Multi-agent generation** (critic-revise loop + judge panel):
```bash
gap2idea generate-ideas --mode orchestrated --orchestrate-mode within \
  --judges "anthropic/claude-sonnet-4,openai/gpt-4o,google/gemini-2.5-flash"
```

Run `gap2idea <command> -h` for all flags.

## How it works

- **Gap extraction** — two engines, same `gaps.tsv` schema: an LLM extractor (verbatim,
  expensive) and a cheap funnel (structural slice → cue/embedding classifier → optional
  LLM filter) for corpus scale.
- **Theme mining** — gaps form a multi-relational graph; Leiden communities plus
  edge-betweenness *bridges* and *frontier* nodes seed novelty-by-recombination.
- **Idea generation** — grounded in retrieved evidence with anti-hallucination gates
  (verbatim evidence, evidence-overlap, named baseline + falsifiable prediction); optional
  multi-agent critic / revise / sanity loop.
- **Evaluation** — LLM-as-judge (novelty / specificity / feasibility / grounding), optional
  cross-provider panel with inter-judge agreement, Semantic-Scholar novelty, human-study form.
- **Output** — per-idea LaTeX (`minimal` / `standard` / `ieee`) or one consolidated PDF;
  `--full-paper` expands an idea into a paper plan. `serve-mcp` exposes the corpus to
  Claude Desktop / Cursor.

## Docs & layout

- **[docs/CODEBASE_MAP.md](docs/CODEBASE_MAP.md)** — file-by-file map + reading order. Start here to navigate the code.
- **[docs/experiments/](docs/experiments/)** — lab notebook + results registry (all numbers, reproduce steps).
- **[docs/gap_extraction_architecture.md](docs/gap_extraction_architecture.md)** — the extraction funnel design.
- **[scripts/README.md](scripts/README.md)** — experiment & ops scripts, grouped by purpose.

## Requirements

- Python ≥ 3.10
- `OPENROUTER_API_KEY` **(required)** — one key for OpenAI / Anthropic / Google / open-source models.
- `S2_API_KEY` (recommended) — lifts Semantic Scholar rate limits.
- Optional: `tectonic` or `pdflatex` on PATH for `export-ideas --format rendered-pdf`.

## License

MIT
