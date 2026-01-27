
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

   This creates a virtual environment (default: `.venv`) and installs dependencies exactly as pinned in `uv.lock`.

   > 💡 Windows / OneDrive tip (optional):
   >
   > ```bash
   > UV_VENV_PATH=.uv-venv uv sync
   > ```

3. **Activate the environment** (so you don’t need `uv` in every command):

   **Windows (PowerShell / CMD):**

   ```bash
   .venv\Scripts\activate
   ```

   (or if you used `UV_VENV_PATH=.uv-venv`):

   ```bash
   .uv-venv\Scripts\activate
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

| Path                               | Purpose                                                |
| ---------------------------------- | ------------------------------------------------------ |
| `data/pdfs/`                       | PDFs named `<paper_id>.pdf`; the UI links to them.     |
| `data/gaps/*.tsv`                  | Input TSV(s) with columns `paper_id`, `gap_text`, etc. |
| `artifacts/gaps_with_clusters.tsv` | Each gap with its embedding + assigned cluster.        |
| `artifacts/cluster_pairs.tsv`      | Similar cluster pairs that seed idea generation.       |
| `artifacts/ideas_openai.tsv`       | Streamlit writes generated ideas here.                 |

## Commands you will actually use

| Goal                                | Command                                           |
| ----------------------------------- | ------------------------------------------------- |
| Sync deps after pulling changes     | `uv sync`                                         |
| Run the clustering/theme mining CLI | `gap2idea theme-mine --gaps-tsv <your_file.tsv>`  |
| Launch the Streamlit dashboard      | `streamlit run src/gap2idea/app/streamlit_app.py` |

Additional internals worth knowing:

* [`src/gap2idea/cli.py`](src/gap2idea/cli.py) wires the `gap2idea` console entry point.
* [`src/gap2idea/app/streamlit_app.py`](src/gap2idea/app/streamlit_app.py) powers the dashboard experience.
* Pipelines for gaps, embeddings, OpenAI calls, and clustering live under [`src/gap2idea/pipeline/`](src/gap2idea/pipeline/).

## License

No license is declared yet. Please obtain the repository owner's permission before redistribution.
