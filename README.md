# Gap2Idea

**Gap2Idea** is an end-to-end pipeline that mines *research gaps* (Limitations, Future Work, Open Problems) from academic papers, clusters them into themes, and synthesises **novel, evidence-grounded research ideas** by bridging pairs of themes. Every generated idea is fact-checked against Semantic Scholar for novelty and scored by an LLM-as-judge rubric.

## What's new (v0.3)

**Multi-agent + MCP + exports.** Everything from v0.2 plus:

- **Three idea-generation modes** — `bridge` (pair gap-clusters), `within` (synthesise one idea per cluster), `method-gap` (apply retrieved method-claims to a cluster).
- **Method library** — new `extract-methods` stage mines method-claim sentences from abstracts/intros, embedded for sweet-spot retrieval against gap clusters.
- **Multi-agent orchestrator** — `--mode orchestrated`: synthesiser drafts → critic agent inspects (with tool access for novelty + evidence overlap) → revisor edits → up to N iterations → judge panel scores.
- **Critic agent** — separate Anthropic-by-default model that picks specific weaknesses on novelty/specificity/feasibility/evidence-grounding and proposes concrete fixes.
- **Multi-judge panel** — `evaluate-ideas --judges <comma-list>` runs 3+ models in parallel and reports per-axis consensus + inter-judge agreement (1 − mean std / 4).
- **MCP server** — `gap2idea serve-mcp` exposes the corpus + ops over the Model Context Protocol so Claude Desktop / Cursor / any MCP client can query your literature directly.
- **LaTeX paper export** — per-idea `.tex` file with a starter template (title, RQ, method, evaluation, contribution, risks, bibliography stub, disclaimer banner). Compile with `pdflatex`.
- **PDF library export** — one consolidated PDF summary of the whole idea library, rendered with ReportLab (pure Python, no system deps).

**From v0.2, still here:**

- **Bridge-score pair selection** — peaks at moderate similarity, penalises paper-overlap, rewards gap-type complementarity.
- **LLM cluster labels** + **diverse evidence picking** + **S2 novelty check**.
- **Multi-tab Streamlit dashboard** with sidebar nav, per-idea LaTeX export, library PDF/TSV/MD export.

## Architecture

```
                       arxiv-metadata-oai-snapshot.json   OR   Semantic Scholar search
                                            │
                                            ▼
                              select-papers ─► data/papers_subset.tsv
                                            │
                                            ▼
                             download-pdfs ─► data/pdfs/*.pdf
                                            │
                                            ▼
                              extract-text ─► data/paper_texts.jsonl
                                            │
                                            ▼ (regex headings + window fallback)
                          extract-sections ─► data/sections_extracted.jsonl
                                            │
                                            ▼ (OpenAI structured outputs, verbatim only)
                              extract-gaps ─► data/gaps.tsv
                                            │
                          ┌─────────────────┴────────────────┐
                          ▼                                  ▼
              theme-mine (KMeans/HDBSCAN,           fetch-metadata
              LLM labels, bridge score)             (Semantic Scholar)
                          │                                  │
                          ▼                                  ▼
                  artifacts/cluster_pairs.tsv      artifacts/papers_metadata.tsv
                          │
                          ▼ (diverse evidence + novelty check)
                  generate-ideas ─► artifacts/{ideas.tsv, ideas_full.jsonl}
                          │
                          ▼ (LLM-as-judge, 4-axis rubric)
                evaluate-ideas ─► artifacts/{idea_eval.tsv, evaluation_report.md}
```

## Methodology

### 1. Gap extraction
We feed only the *Limitations / Future Work / Discussion* sections of each paper to the LLM, with strict JSON schema enforcement requiring **verbatim** sentences and paragraphs. This anchors every "gap" on textual provenance from the source paper, avoiding LLM hallucination of capabilities authors never claimed were missing.

All LLM calls are routed through **[OpenRouter](https://openrouter.ai/)** so you can swap providers (OpenAI · Anthropic · Google · Meta · open-source) by changing one CLI flag, e.g. `--model anthropic/claude-sonnet-4` or `--model google/gemini-2.5-flash`.

### 2. Theme mining
Gaps are embedded with `all-MiniLM-L6-v2`, then clustered with KMeans (silhouette-tuned k) for small corpora or HDBSCAN for large ones. Each cluster gets two labels: an LLM-produced theme name (human-readable) and a TF-IDF keyword list (interpretable).

### 3. Bridge-score pair selection
For every cluster pair we compute

```
bridge_score = peak(cosine_sim, 0.45)
             × (1 - paper_overlap_jaccard)
             × (0.5 + 0.5 × type_complementarity)
```

- `peak(s, 0.45)` is a triangular peak that maxes out at moderate cosine similarity (≈0.45) and decays to 0 at both 0 and 1. Pairs that are too similar produce restatements; pairs that are too far apart produce nonsense.
- `paper_overlap` is the Jaccard overlap of source-paper IDs between the two clusters. Penalised because pairs from the same papers can't claim cross-pollination.
- `type_complementarity` is the L1 distance between the two clusters' `gap_type` distributions, scaled to [0, 1]. A `limitation` cluster paired with a `future_work` cluster scores higher than two identical-type clusters.

### 4. Idea synthesis — three modes

Each mode shares: diverse-evidence sampling, strict-JSON LLM call, S2 novelty check, full TSV+JSONL provenance.

**`--mode bridge`** *(default)* — pair two gap-clusters in the bridge-score sweet spot, ask the LLM to combine them. Encodes "novelty by recombining related-but-distinct themes." Works well when two themes are *adjacent* (e.g. continual-learning gaps × OOD-detection gaps); produces poor ideas when the two themes are merely *related by topic* (e.g. "ML" × "medical research").

**`--mode within`** — for each gap-cluster, synthesise ONE idea from k=6+ diverse evidence rows in *that* cluster. A cluster represents a recurring research opportunity; the idea fills the opportunity. No pairing, no bridge formula. Simpler and more defensible when you can't justify the cross-pollination claim.

**`--mode method-gap`** — explicit "X solves Y" structure. We run a second extraction pass (`extract-methods`) to mine method-claim sentences from abstracts/introductions. For each gap-cluster, we retrieve the top-K method statements whose cosine similarity to the cluster centroid falls in the **sweet spot** [0.30, 0.70] (configurable). Then the LLM applies the retrieved methods to the cluster's gaps. Every idea has explicit provenance: "*Apply method M (paper P1) to address gap G (paper P2).*" Requires `gap2idea extract-methods` to be run first.

All three modes use the LLM (OpenRouter, default `openai/gpt-4.1-mini`) with a strict JSON schema requiring concrete `method_sketch`, `evaluation_plan` (named metric + baseline), and `evidence_used` (subset of input evidence with verbatim quotes).

### 5. Novelty validation
The idea's `title + research_question` is sent to Semantic Scholar's `/paper/search`. We embed the idea text and each hit's abstract with the same sentence-transformer and report:

- `novelty_score = 1 − max_cosine(idea, S2_hit_abstract)` ∈ [0, 1]
- `closest_paper` (paper ID, title, year, similarity) so reviewers can audit.

### 6. Evaluation
`evaluate-ideas` runs an LLM-as-judge that scores each idea 1-5 on **novelty / specificity / feasibility / evidence_grounding** with per-axis rationales. We separately compute a quantitative **`evidence_overlap`** — the fraction of `evidence_used` (paper_id, gap_sentence) pairs that actually appeared in the input — to catch hallucinated citations even when the judge is fooled.

To mitigate self-evaluation bias, the **default judge model is from a different provider than the generator**: generation uses `openai/gpt-4.1-mini` and judging uses `anthropic/claude-sonnet-4`. Override either side with `--model` / `--judge-model` on the CLI.

## Installation

Three supported ways: `uv` (recommended), `pip`, or Docker.

### With `uv` (recommended)

```bash
# one time
python -m pip install --upgrade uv

# from repo root
uv sync                                        # creates .venv with all deps
.venv/Scripts/activate                         # Windows; or `source .venv/bin/activate`
cp .env.example .env                           # then edit OPENROUTER_API_KEY
```

Or, if you don't want to activate the env, prefix commands with `uv run`:

```bash
uv run gap2idea theme-mine --gaps-tsv data/gaps.tsv
uv run streamlit run src/gap2idea/app/streamlit_app.py
```

### With `pip`

```bash
git clone <repo>
cd Gap2Idea
python -m venv .venv && .venv/Scripts/activate    # or source .venv/bin/activate
pip install -e .
cp .env.example .env                              # then edit OPENROUTER_API_KEY
```

### With Docker

Ships the entire UI + pipeline stack via the provided `Dockerfile` (uses `uv sync --frozen --no-dev` under the hood):

```bash
docker build -t gap2idea:latest -f Dockerfile .
docker run --rm -p 8501:8501 \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/artifacts:/app/artifacts" \
  --env-file .env \
  gap2idea:latest
```

On Windows PowerShell, replace `$(pwd)` with `${PWD}` and use backticks for line continuation.

### Required env vars

| Variable | Status | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | **Required** | All LLM calls. Get one at <https://openrouter.ai/keys>. |
| `S2_API_KEY` | Recommended | Lifts Semantic Scholar rate limit from ~100/5min unauth → much higher. |
| `OPENAI_API_KEY` | Legacy fallback | Honoured only if `OPENROUTER_API_KEY` is unset. |

### Choosing models

All LLM-calling subcommands take `--model <openrouter_slug>`. Examples:

| Use case | Suggested slug | Why |
|---|---|---|
| Fast, cheap, strict JSON | `openai/gpt-4.1-mini` (default) | Reliable structured outputs, low cost |
| Higher-quality generation | `openai/gpt-4o` or `anthropic/claude-sonnet-4` | Better reasoning, higher cost |
| Independent judge | `anthropic/claude-sonnet-4` (default judge) | Different provider from generator → less self-eval bias |
| Open-source / local-equivalent | `meta-llama/llama-3.3-70b-instruct` | Free-tier on OpenRouter |

Full model catalogue: <https://openrouter.ai/models>.

## Usage

### Option A — full pipeline from arXiv
```bash
gap2idea select-papers --source s2 --query "graph neural networks dynamic" --n 100
gap2idea download-pdfs
gap2idea run-all                       # extract-text → … → evaluate-ideas
streamlit run src/gap2idea/app/streamlit_app.py
```

### Option B — already have `data/gaps.tsv`
```bash
gap2idea theme-mine --gaps-tsv data/gaps.tsv
gap2idea fetch-metadata
gap2idea generate-ideas --n-pairs 15
gap2idea evaluate-ideas --judge-model gpt-4.1-mini
streamlit run src/gap2idea/app/streamlit_app.py
```

### Per-stage commands
```
gap2idea select-papers      # corpus selection (S2 search OR arxiv snapshot)
gap2idea download-pdfs      # parallel arxiv PDF download
gap2idea extract-text       # PyMuPDF text extraction
gap2idea extract-sections   # regex section parser + window fallback
gap2idea extract-gaps       # LLM gap extraction from limitations/future-work
gap2idea extract-methods    # LLM method-claim extraction from abstracts/intros
gap2idea theme-mine         # embed → cluster → label → bridge-score pairs
gap2idea fetch-metadata     # S2 enrichment for every paper id
gap2idea generate-ideas     # idea synthesis + novelty check (3 modes — see below)
gap2idea evaluate-ideas     # LLM-as-judge rubric + markdown report
gap2idea run-all            # extract-text through evaluate-ideas
```

### Idea generation modes

```bash
# Default — bridge two gap-clusters in the similarity sweet spot
gap2idea generate-ideas --mode bridge --n-pairs 10

# Synthesise one idea per cluster from its own gaps (no pairing)
gap2idea generate-ideas --mode within --n-pairs 10

# Apply retrieved methods to each gap-cluster (requires extract-methods first)
gap2idea extract-methods
gap2idea generate-ideas --mode method-gap --n-pairs 10 --sim-low 0.30 --sim-high 0.70

# Multi-agent: synthesiser + critic-revise loop + judge panel
gap2idea generate-ideas --mode orchestrated \
    --orchestrate-mode within \
    --critic-model anthropic/claude-sonnet-4 \
    --judges "anthropic/claude-sonnet-4,openai/gpt-4o,google/gemini-2.5-flash" \
    --max-critic-iter 2 --n-pairs 5
```

Output schema is unified (`mode` column distinguishes them). You can run all four sequentially and compare in the **Ideas** tab of the Streamlit app.

### Multi-judge evaluation

```bash
gap2idea evaluate-ideas \
    --judges "anthropic/claude-sonnet-4,openai/gpt-4o,google/gemini-2.5-flash"
```

Produces `artifacts/idea_eval.tsv` with consensus scores per axis + per-judge breakdown + an `agreement` score in [0, 1] indicating inter-judge consistency.

### Idea export

```bash
# One .tex per idea using a bundled template (minimal | standard | ieee)
gap2idea export-ideas --format latex --template standard

# Server-render every .tex to PDF (requires tectonic or pdflatex on PATH)
gap2idea export-ideas --format rendered-pdf --template ieee

# Single consolidated PDF summary — no LaTeX install needed (reportlab)
gap2idea export-ideas --format library-pdf

# Use your own template (Jinja2 LaTeX)
gap2idea export-ideas --format latex --template-file path/to/mine.tex.j2
```

**Three bundled templates**:

| Name | Best for | Notes |
|---|---|---|
| `minimal`  | A clean draft you'll restructure later | Stock `article` class, no exotic packages |
| `standard` *(default)* | NeurIPS-style starter | Single column, tcolorbox disclaimer |
| `ieee`     | IEEE conference look | Two-column via `multicol`, fits in stock TeX Live |

**Custom templates** — pass a Jinja2 `.tex.j2` file via `--template-file`. The
template receives every variable in `gap2idea.pipeline.export.TEMPLATE_VARIABLES`
plus an `escape_tex` filter (use it on every user-supplied string field).

**Streamlit equivalents** — the Ideas page has a template dropdown + a "Custom (upload)" file uploader, plus a per-card "📄 Download .tex" and "📑 Render & download PDF" (the latter only enabled when a LaTeX compiler is found).

### Optional: install a LaTeX compiler

The `rendered-pdf` format and the Streamlit "Render & download PDF" button both need either:
- **[Tectonic](https://tectonic-typesetting.github.io/)** — recommended, single binary, auto-fetches packages on first compile.
- **TeX Live / MiKTeX** — full suite, includes `pdflatex`.

If neither is installed, you can still use `--format latex` to produce `.tex` files and compile them yourself or via Overleaf.

### MCP server (Claude Desktop integration)

```bash
gap2idea serve-mcp                     # runs on stdio
```

Then add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "gap2idea": {
      "command": "gap2idea",
      "args": ["serve-mcp", "--root", "C:/path/to/Gap2Idea"]
    }
  }
}
```

Claude can now call `list_themes`, `get_theme`, `get_evidence`, `retrieve_methods`, `search_prior_art`, `score_novelty`, `check_evidence_overlap`, `list_ideas`, `get_idea`, and `save_idea` against your corpus.

## Project layout

```
src/gap2idea/
  cli.py                       # all subcommands
  config.py io.py utils.py
  tools.py                     # NEW: shared agentic tool surface (also used by MCP)
  mcp_server.py                # NEW: Model Context Protocol server (stdio)
  pipeline/
    arxiv_select.py            # S2 search + arxiv snapshot + PDF download
    pdf_text.py                # PyMuPDF
    sections.py                # Limitations / Future Work finder
    openai_gaps.py             # gap extraction (LLM strict JSON)
    openai_methods.py          # NEW: method-claim extraction
    theme_mining.py            # embed/cluster/label + bridge-score pairs
    semantic_scholar.py        # S2 Graph API client w/ 429 retry
    openai_ideas.py            # idea synthesis (3 modes) + novelty check
    agents.py                  # NEW: critic + revisor + critique-revise loop
    orchestrator.py            # NEW: full multi-agent end-to-end pipeline
    evaluation.py              # LLM-as-judge + multi-judge panel + report
    export.py                  # NEW: LaTeX + PDF rendering
    llm.py                     # OpenRouter client factory + JSON parser
  templates/
    idea_paper.tex.j2          # NEW: per-idea LaTeX starter template
  app/
    streamlit_app.py           # tabbed dashboard (with export buttons)
artifacts/                     # generated outputs (gitignored)
  exports/                     # NEW: LaTeX + PDF deliverables
data/                          # raw inputs (gitignored)
notebooks/                     # exploratory work
tests/                         # 100+ unit + mocked-integration tests
```

## Requirements

- Python ≥ 3.10
- **OpenRouter** API key (`OPENROUTER_API_KEY`) — unifies access to OpenAI / Anthropic / Google / open-source models behind one key.
- Optional: Semantic Scholar API key (`S2_API_KEY`) — strongly recommended for large corpora to avoid 429 rate limits.
- Optional: `OPENAI_API_KEY` is honoured as a fallback if `OPENROUTER_API_KEY` is unset.

## License

MIT
