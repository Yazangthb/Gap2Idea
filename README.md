# Gap2Idea

Pipeline + Streamlit UI that clusters research gaps, labels the resulting themes, and expands promising combinations into concrete research ideas.

## Quick start (local dev)

1. **Install uv** (one time):
   ```bash
   python -m pip install --upgrade pip
   python -m pip install --upgrade uv
   ```

Verify:

```bash
uv --version
```

2. **Create and sync the environment** (from the repo root):

   ```bash
   uv sync
   ```

3. **Activate the environment** (so you don’t need `uv` in every command):

   **Windows (PowerShell / CMD):**

   ```bash
   .venv\Scripts\activate
   ```

   **macOS / Linux:**

   ```bash
   source .venv/bin/activate
   ```
4. **Run the theme miner** on your TSV of gap sentences:

   ```bash
   gap2idea theme-mine --gaps-tsv data/gaps/raw_gaps.tsv
   ```
5. **Explore clusters** in Streamlit:

   ```bash
   streamlit run src/gap2idea/app/streamlit_app.py
   ```

### Alternative (no activation)

If you prefer not to activate the virtual environment, prefix commands with `uv run`:

```bash
uv run gap2idea theme-mine --gaps-tsv data/gaps/raw_gaps.tsv
uv run streamlit run src/gap2idea/app/streamlit_app.py
```

## Run with Docker

You can ship the entire UI + pipeline stack via the provided [`Dockerfile`](Dockerfile), which now mirrors the `uv`-managed environment used locally:

1. **Build the image** (once, it installs deps via `uv sync --frozen --no-dev`):

   ```bash
   docker build -t gap2idea:latest -f Dockerfile .
   ```
2. **Start the container** (ensure `data/` and `artifacts/` exist locally so the mounts work):

   ```bash
   docker run --rm -p 8501:8501 \
     -v "$(pwd)/data:/app/data" \
     -v "$(pwd)/artifacts:/app/artifacts" \
     gap2idea:latest
   ```

   On Windows CMD/PowerShell, replace `$(pwd)` with `%cd%` (or run the command inside WSL).

   **Windows PowerShell example:**

   ```powershell
   docker run --rm -p 8501:8501 `
     -v "${PWD}\data:/app/data" `
     -v "${PWD}\artifacts:/app/artifacts" `
     gap2idea:latest
   ```

   **Windows CMD example:**

   ```cmd
   docker run --rm -p 8501:8501 ^
     -v "%cd%\data:/app/data" ^
     -v "%cd%\artifacts:/app/artifacts" ^
     gap2idea:latest
   ```

Streamlit will now be reachable at http://localhost:8501 while reading/writing the same mounted folders as the local workflow.

➡️ The Docker context is trimmed via [`.dockerignore`](.dockerignore) so local notebooks, caches, and mounted data are not copied into the image.

## Data + artifacts in short

| Path                                 | Purpose                                                    |
| ------------------------------------ | ---------------------------------------------------------- |
| `data/pdfs/`                       | PDFs named `<paper_id>.pdf`; the UI links to them.       |
| `data/gaps/*.tsv`                  | Input TSV(s) with columns `paper_id`, `gap_text`, etc. |
| `artifacts/gaps_with_clusters.tsv` | Each gap with its embedding + assigned cluster.            |
| `artifacts/cluster_pairs.tsv`      | Similar cluster pairs that seed idea generation.           |
| `artifacts/ideas_openai.tsv`       | Streamlit writes generated ideas here.                     |

## Preparing data when you only have `arxiv-metadata-oai-snapshot.json`

The repository assumes you eventually own a TSV of “gap” sentences (limitations, future work, outlooks, etc.). If the only asset you own today is the raw [`arxiv-metadata-oai-snapshot.json`](https://www.kaggle.com/datasets/Cornell-University/arxiv) JSONL dump, follow the steps below to bootstrap every intermediate file. Each step mirrors the notebooks under [`notebooks/`](notebooks/).

### 0. Place the snapshot

1. Download `arxiv-metadata-oai-snapshot.json` (≈4 GB compressed) from Kaggle.
2. Decompress if needed and place/symlink it at `data/arxiv-metadata-oai-snapshot.json` (the Streamlit UI expects this relative location when it needs metadata).

### 1. Select a manageable paper subset

Start from the JSONL metadata and pick a slice that matches your domain, recency, and size constraints. Use the CLI command (no notebooks required):

```bash
uv run gap2idea select-arxiv \
  --metadata data/arxiv-metadata-oai-snapshot.json \
  --output data/papers_subset.tsv \
  --categories cs.LG,stat.ML \
  --min-year 2021 \
  --n-papers 250
```

Or run everything in one command:

```bash
uv run gap2idea run-pipeline \
  --metadata data/arxiv-metadata-oai-snapshot.json \
  --categories cs.LG,stat.ML \
  --min-year 2021 \
  --n-papers 250 \
  --model gpt-4.1-mini \
  --max-papers 50
```

Outcome: `data/papers_subset.tsv` (or another path you choose) containing 3–4 columns per paper. The IDs here will drive every subsequent step.

### 2. Download PDFs and extract lightweight text dumps

Using the subset IDs, download PDFs to `data/pdfs/` and extract the first ~10–12 pages of text to `data/texts/` via the CLI:

```bash
uv run gap2idea fetch-pdfs --papers-tsv data/papers_subset.tsv
```

Expected directories afterwards:

* `data/pdfs/<arxiv-id>.pdf`
* `data/texts/<arxiv-id>.txt`

### 3. Mine limitation / future-work sections → `sections_extracted.jsonl`

Run the CLI to parse tail pages of each PDF, look for headings such as “Limitations”, “Future Work”, “Discussion”, and emit rows shaped as:

```json
{"id": "2503.17793", "section_type": "future_work", "heading": "Future Work", "section_text": "..."}
```

```bash
uv run gap2idea extract-sections --output data/sections_extracted.jsonl
```

This structured JSONL is now the raw material for gap extraction.

### 4. Convert sections → `data/gaps_openai.tsv`

Use the OpenAI-backed extractor to convert sections into a TSV of gaps (expects `OPENAI_API_KEY` in `.env`):

```bash
uv run gap2idea extract-gaps \
  --sections-jsonl data/sections_extracted.jsonl \
  --output data/gaps_openai.tsv \
  --model gpt-4.1-mini \
  --max-papers 50
```

The output TSV contains: `id`, `gap_type`, `gap_sentence`, `paragraph_text`, `confidence`.

### 5. Run the Gap2Idea pipeline + UI

Once `data/gaps_openai.tsv` exists, the rest is the standard workflow already outlined above:

```bash
uv run gap2idea theme-mine --gaps-tsv data/gaps_openai.tsv
uv run streamlit run src/gap2idea/app/streamlit_app.py
```

Artifacts (`artifacts/gaps_with_clusters.tsv`, `artifacts/cluster_pairs.tsv`, etc.) will populate, and the Streamlit dashboard will let you explore them immediately. If you later gather more metadata or PDFs, simply rerun steps 1–4 and re-execute the CLI.

## Commands you will actually use

| Goal                                | Command                                             |
| ----------------------------------- | --------------------------------------------------- |
| Sync deps after pulling changes     | `uv sync`                                         |
| Run full pipeline from metadata     | `gap2idea run-pipeline --metadata data/arxiv-metadata-oai-snapshot.json` |
| Run the clustering/theme mining CLI | `gap2idea theme-mine --gaps-tsv <your_file.tsv>`  |
| Launch the Streamlit dashboard      | `streamlit run src/gap2idea/app/streamlit_app.py` |

Additional internals worth knowing:

* [`src/gap2idea/cli.py`](src/gap2idea/cli.py) wires the `gap2idea` console entry point.
* [`src/gap2idea/app/streamlit_app.py`](src/gap2idea/app/streamlit_app.py) powers the dashboard experience.
* Pipelines for selection, PDF parsing, OpenAI calls, and clustering live under [`src/gap2idea/pipeline/`](src/gap2idea/pipeline/).

## License

No license is declared yet. Please obtain the repository owner's permission before redistribution.
