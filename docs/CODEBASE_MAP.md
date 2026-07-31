# Gap2Idea — Codebase Map & Reading Guide

> Come back to this after time away. It gives the **mental model**, the **flow**,
> a **reading order**, a **file-by-file map**, and the **gotchas**.
> Deep-dives: extraction → [gap_extraction_architecture.md](gap_extraction_architecture.md);
> the research story & numbers → [experiments/experiment_log.md](experiments/experiment_log.md)
> + [experiments/results_registry.md](experiments/results_registry.md);
> honest status → [paper/00_full_system_review.md](paper/00_full_system_review.md).

## 1. Mental model (read this first)
- **One package, one CLI.** All code is `src/gap2idea/`. You drive it with
  `gap2idea <stage>` — every stage is a `cmd_*` function in `cli.py`.
- **Stages pass files, not objects.** Each stage reads a file and writes a file:
  raw inputs in `data/`, generated outputs in `artifacts/`. No database.
  `config.py:get_paths()` is the single source for those locations.
- **Every LLM call is an OpenRouter slug.** Swap any model with `--model
  <provider/model>`; `pipeline/llm.py` builds the client and tolerantly parses JSON.
- **The pipeline is a spine with one fork** (gap extraction) and a
  graph-structured middle. Learn the spine (`cli.py`) and you know the system.
- **There are TWO extraction engines** — the expensive per-paper LLM (`openai_gaps.py`)
  and the cheap funnel (`gap_funnel.py`). The funnel is the research contribution.

## 2. The flow (the spine)
```
select-papers ─► data/papers_subset.tsv          arxiv_select.py
download-pdfs ─► data/pdfs/*.pdf                  arxiv_select.py
extract-text  ─► data/paper_texts.jsonl           pdf_text.py   (id, text, blocks)
      │
  GAP EXTRACTION  (pick one engine → data/gaps.tsv)
      ├─ OLD:  extract-sections → extract-gaps     sections.py + openai_gaps.py  (~$4000/1M)
      └─ NEW:  extract-gaps-funnel                 gap_funnel.py (+gap_llm_filter.py)  (~$3–31/1M)
      │
theme-mine ─► artifacts/{gaps_with_clusters, cluster_pairs, gap_pairs, gap_frontier, gap_graph.gpickle}
      │        theme_mining.py + gap_graph.py   (--method graph is the default)
fetch-metadata ─► artifacts/papers_metadata.tsv   semantic_scholar.py
      │
generate-ideas ─► artifacts/{ideas.tsv, ideas_full.jsonl}
      │           openai_ideas.py (simple) OR orchestrator.py+agents.py+sanity.py (multi-agent)
evaluate-ideas ─► artifacts/{idea_eval.tsv, evaluation_report.md}   evaluation.py
export-ideas   ─► artifacts/exports/*.tex|*.pdf   export.py + paper_drafter.py
serve-mcp      ─► MCP server over the same corpus  mcp_server.py + tools.py
```
`run-all` chains extract-text → evaluate-ideas. **Note:** `run-all` currently uses
the OLD extraction path and omits `extract-methods`/`export` (known drift).

## 3. Recommended reading order (fastest way back in)
1. **`README.md`** — pitch + architecture diagram + usage. (5 min)
2. **`docs/paper/00_full_system_review.md`** — the honest "what's built vs
   validated" scorecard. Orients you to reality fast.
3. **`docs/experiments/experiment_log.md`** — the extraction research story
   (Phases 0–6) and current state.
4. **`src/gap2idea/cli.py`** — the whole pipeline in one file; skim every `cmd_*`.
5. **Follow one vertical slice end-to-end**, in this order:
   `gap_funnel.py` → `theme_mining.py` → `gap_graph.py` → `openai_ideas.py` →
   `agents.py` → `orchestrator.py` → `evaluation.py` → `export.py`.
6. **`tools.py` + `mcp_server.py`** — the shared agent tool surface (ties
   generation and the MCP integration together).
7. **Run it** on a tiny corpus and watch `data/` → `artifacts/` fill:
   `gap2idea theme-mine --gaps-tsv data/gaps.tsv` then `generate-ideas`, or open
   the Streamlit app.

## 4. File-by-file map

### Entry & plumbing
| File | Role |
|---|---|
| `cli.py` | all subcommands; `gap2idea` entry point. **Start here.** |
| `config.py` | `get_paths()` — where `data/` and `artifacts/` live |
| `io.py` | `read_tsv` / `write_tsv` |
| `utils.py` | logger, `retry`, `set_seed` |
| `pipeline/llm.py` | OpenRouter client factory + tolerant JSON parser |
| `pipeline/resource_compat.py` | dependency/version compatibility shims |

### Ingestion
| File | Role |
|---|---|
| `pipeline/arxiv_select.py` | corpus selection (S2 search / arxiv snapshot) + parallel PDF download |
| `pipeline/pdf_text.py` (+ `pdf_text_v2.py`) | PyMuPDF → `{id, text, blocks}`; **origin of the two-column scrambling** |
| `pipeline/semantic_scholar.py` | S2 Graph API client (novelty, metadata, prior art), 429 retry |

### Gap extraction (the fork)
| File | Role |
|---|---|
| `pipeline/sections.py` | regex Limitations/Future-Work section finder (legacy) |
| `pipeline/openai_gaps.py` | LLM gap extraction, strict JSON, ≤2/paper (expensive path) |
| `pipeline/gap_funnel.py` | **the funnel** — Stage A slice + Stage B (cue rules + embedding/BERT head) |
| `pipeline/gap_prefilter.py` | shared text normalize + sentence split (used by funnel) |
| `pipeline/gap_llm_filter.py` | **Stage C** LLM precision filter over ~6 survivors/paper |
| `pipeline/openai_methods.py` | method-claim extraction (feeds `--mode method-gap`) |

### Structuring (embed → graph)
| File | Role |
|---|---|
| `pipeline/theme_mining.py` | embed gaps → cluster → LLM labels → **bridge-score** pairs |
| `pipeline/gap_graph.py` | **multi-relational gap graph** (semantic/paper/section/method edges; Leiden communities; bridge + frontier scoring) — default engine |

### Idea generation
| File | Role |
|---|---|
| `pipeline/openai_ideas.py` | idea synthesis: `bridge` / `within` / `method-gap` / `frontier`; diverse evidence + novelty check |
| `pipeline/agents.py` | Synthesiser / Critic / JudgePanel agent classes (use `tools.py`) |
| `pipeline/orchestrator.py` | end-to-end multi-agent: synth → critic-revise → sanity → panel |
| `pipeline/sanity.py` | experimental-sanity stage: 8 agents / 4 phases, **sandboxed code run** |
| `tools.py` | shared agentic tool surface (READ / RETRIEVAL / CHECK / WRITE) |

### Evaluation & output
| File | Role |
|---|---|
| `pipeline/evaluation.py` | LLM-as-judge (4 axes) single + panel + `evidence_overlap` |
| `pipeline/export.py` | LaTeX (3 templates) + rendered PDF + reportlab library PDF |
| `pipeline/paper_drafter.py` | expand one idea → full paper *plan* (`--full-paper`) |
| `templates/idea_paper_{minimal,standard,ieee}.tex.j2` | bundled LaTeX templates |

### Deployment & UI
| File | Role |
|---|---|
| `mcp_server.py` | FastMCP server wrapping `tools.py` (Claude Desktop / Cursor) |
| `app/streamlit_app.py` | tabbed dashboard (ideas, exports, drafts) |
| `Dockerfile` (repo root) | container; deployed on Cloud Run |

### Benchmarks (in-package)
| File | Role |
|---|---|
| `pipeline/extraction_bench.py` (+ `_plots`, `_ablation_plots`) | extraction vs unarXive gold |
| `pipeline/clustering_bench.py` (+ `_plots`) | clusterer × embedder grid |
| `pipeline/_versions/*` | archived module versions (funnel v1, llm_filter v4) |

## 5. Where things live (data & docs)
| Path | What |
|---|---|
| `data/` | inputs + artifacts: `bench/`, **`bench_gap/`** (live 19-gap gold), `bench_gold/`, `dataset_v*`, `gap_head*.joblib` |
| `runs/{ai,ml,math}[_v1]/` | per-domain corpus runs (each mirrors the full `data/` pipeline) |
| `artifacts/` | generated pipeline outputs (gitignored) |
| `scripts/` | ~77 experiment/one-off scripts (funnel training, benches, dataset builds) |
| `docs/experiments/` | the lab notebook — **source of truth for numbers** |
| `docs/paper/` | paper drafts + full-system review |
| `tests/` | unit + mocked-integration tests |

## 6. Gotchas (things that will trip you after a break)
- **Two extraction engines**, and `run-all` still calls the OLD one (`extract-gaps`),
  not `extract-gaps-funnel`. Wire the funnel in when you make it the default.
- **`theme-mine --method graph` is the default** (not kmeans). It writes the same
  artifact filenames as the legacy path, so downstream is drop-in — but also emits
  `gap_pairs.tsv`, `gap_frontier.tsv`, `gap_graph.gpickle`.
- **Scrambled PDF text** → the code matches by **token-containment**, never substring.
- **Evaluation is code-complete but not yet run credibly** — a `fake_responses.csv`
  exists and must be deleted before any submission (see the readiness report).
- **Nothing since May 30 is committed** — 164 untracked files. `git commit` early.
- **Version drift:** `pyproject.toml` 0.2.0 vs README "v0.4" — pick one.
- **API keys:** `OPENROUTER_API_KEY` (required), `S2_API_KEY` (recommended) in `.env`.

## 7. Run it
```bash
# already have gaps? structure → generate → evaluate:
gap2idea theme-mine --gaps-tsv data/gaps.tsv
gap2idea fetch-metadata
gap2idea generate-ideas --n-pairs 10
gap2idea evaluate-ideas --judges "anthropic/claude-sonnet-4,openai/gpt-4o"

# cheap extraction from paper_texts.jsonl (no per-paper LLM):
gap2idea extract-gaps-funnel --mode hybrid --head data/gap_head.joblib

# dashboard:
streamlit run src/gap2idea/app/streamlit_app.py
```
