# `code/` — environment & dataset collection

Code for the Stage-03 **Environment setup** and **Dataset collection** tasks of
*Scalable Extraction of Author-Stated Research Gaps in the Scientific Literature: Toward Automated Idea
Generation* (focus: **AI & Data Science**).

This is the data-collection slice of the full project. Extraction, clustering, idea generation, PDF→text
parsing, the Streamlit app, and the MCP server live in the full repo and are **not** included here (yet):
<https://github.com/Yazangthb/Gap2Idea>.

## Contents
```
code/
├── pyproject.toml, uv.lock, .env.example   # environment setup
├── src/gap2idea/
│   ├── cli.py                              # `gap2idea` command-line entry
│   ├── config.py                           # data/ path resolution
│   ├── io.py                               # TSV read/write
│   ├── utils.py                            # logging + seed
│   └── pipeline/
│       ├── semantic_scholar.py             # Semantic Scholar API client
│       ├── arxiv_select.py                 # corpus selection + PDF download
│       ├── gap_prefilter.py                # sentence split / normalize (rule-based)
│       ├── sections.py                     # section-heading regexes
│       └── gap_funnel.py                   # future-work / limitation cue rules
├── scripts/dataset/
│   └── harvest_acl_limitations.py          # build ACL gold gap corpora (from LimGen)
└── notebooks/                              # arxiv + Semantic Scholar collection notebooks
```
Collected + gold data lives in [`../data/`](../data/README.md) (`papers_subset.tsv`, `acl_limitations.tsv`, `acl_futurework.tsv`).

## Environment setup
Python **3.10+**. Package `gap2idea` (src layout; CLI entry `gap2idea`).

```bash
cd code
uv sync                 # exact, reproducible env from uv.lock  (or: pip install -e .)
```
Optional API keys — copy `code/.env.example` → `.env` at the **repo root**:
- `S2_API_KEY` — Semantic Scholar, raises the rate limit (not required for small runs).
- `OPENROUTER_API_KEY` — only for the later LLM stages (not needed for data collection).

## Dataset collection — what the commands do
| Command | Does | Writes |
|---|---|---|
| `gap2idea select-papers` | pick a corpus — **default `--source arxiv`** (keyless arXiv API) on `--query`; also `--source s2` (Semantic Scholar) or `--source snapshot` (local arXiv dump, `--cats cs.LG,stat.ML --min-year 2021`) | `data/papers_subset.tsv` |
| `gap2idea download-pdfs` | download those papers' PDFs from arXiv (parallel, skips cached) | `data/pdfs/` *(gitignored)* |
| `gap2idea fetch-metadata` | enrich paper IDs with Semantic Scholar metadata | TSV |

## Run it — simple
Run **from the repo root** so `data/…` resolves to `em.dcs/data`.

**Option A — full env (reproducible):**
```bash
cd code && uv sync && cd ..
gap2idea select-papers --query "research gap extraction" --n 20   # -> data/papers_subset.tsv
gap2idea download-pdfs                                            # optional -> data/pdfs/
```

**Option B — lightweight (data collection only; no torch/heavy deps):**
```bash
pip install pandas numpy requests python-dotenv
PYTHONPATH=code/src python -m gap2idea.cli select-papers --query "research gap extraction" --n 20
```
Both write `data/papers_subset.tsv`. The default **`--source arxiv`** uses the public arXiv API —
**no key and no large download**. Alternatives: `--source s2` (Semantic Scholar; rate-limited
without an `S2_API_KEY`, often 429/403) and `--source snapshot` (needs the ~4.8 GB arXiv metadata
dump, not included).

## Benchmark gold-data pipeline
`scripts/dataset/harvest_acl_limitations.py` builds the gold **author-stated-gap** corpora
(`data/acl_limitations.tsv`, `data/acl_futurework.tsv`) used to evaluate/train gap extraction. It
harvests the **LimGen** dataset (ACL papers' mandated *Limitations* sections; Faizullah et al.,
ECML-PKDD 2024, CC BY 4.0) — keyless, downloads from GitHub:
```bash
python code/scripts/dataset/harvest_acl_limitations.py     # run from repo root
```
It reuses the rule-based helpers in `pipeline/{gap_prefilter,sections,gap_funnel}.py` and a leakage
guard against `data/bench_gap/gold_sentences.tsv`. Provenance & columns:
[`../data/README.md`](../data/README.md).
