# Gap2Idea

Gap2Idea is a tool that transforms research gaps identified in academic papers into novel research ideas. It clusters gaps into themes, identifies connections between themes, and uses AI to generate innovative research proposals.

## Features

- **Gap Clustering**: Embed and cluster research gaps using sentence transformers and HDBSCAN.
- **Theme Mining**: Automatically label and summarize gap clusters.
- **Idea Generation**: Combine gaps from different themes to generate new research ideas using OpenAI's GPT models.
- **Interactive Exploration**: Web app for visualizing clusters, exploring papers, and generating ideas.
- **Command Line Interface**: Process gaps and generate artifacts via CLI.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment: Copy `.env` and add your OpenAI API key: `OPENAI_API_KEY=your_key_here`

## Usage

### 1. Prepare Data

You need a TSV file with research gaps. The file should have columns like `id`, `gap_sentence`, `confidence`, etc.

Example gaps.tsv:
```
id	gap_sentence	confidence
paper1	There is a lack of studies on...	0.8
...
```

### 2. Run Theme Mining

Process the gaps to cluster them into themes:

```bash
python -m gap2idea.cli theme-mine --gaps-tsv path/to/gaps.tsv --root .
```

This will create artifacts in the `artifacts/` directory: cleaned gaps, embeddings, clusters, labels, summaries, and pairs.

### 3. Explore and Generate Ideas

Run the Streamlit app:

```bash
streamlit run src/gap2idea/app/streamlit_app.py
```

- Search for papers by title.
- Visualize gap clusters in 2D.
- Explore clusters and papers.
- Generate ideas by combining themes.

## Project Structure

- `src/gap2idea/`: Main package
  - `cli.py`: Command line interface
  - `app/streamlit_app.py`: Web app
  - `pipeline/`: Processing modules
- `notebooks/`: Jupyter notebooks for experimentation
- `data/`: Data files (PDFs, texts, etc.)
- `artifacts/`: Generated outputs

## Requirements

- Python 3.8+
- OpenAI API key
- Dependencies listed in `requirements.txt`

## Note

This is part of a thesis project on automating research idea generation from literature gaps.