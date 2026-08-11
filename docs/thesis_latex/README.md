# Gap2Idea — thesis LaTeX project (Overleaf-ready)

A self-contained LaTeX source tree for the thesis. Drafted from the repository
code, experiments, and the literature-review sheet.

## Structure
```
thesis_latex/
├── main.tex                    # preamble + \input of all sections + \bibliography
├── references.bib              # 82 sources from the review sheet + infra/method refs
├── sections/
│   ├── 01_research_questions.tex
│   ├── 02_literature_review.tex   # RQ2 / RQ1 / RQ3, cites references.bib
│   ├── 03_method.tex
│   ├── 04_results.tex
│   └── 05_progress.tex
├── figures/                    # put .png/.pdf figures here (\graphicspath is set)
└── README.md
```

## Compile
- **Overleaf:** New Project → Upload Project → zip this `thesis_latex/` folder
  (or drag the files in). Set the compiler to **pdfLaTeX** and the main file to
  `main.tex`. Overleaf runs `pdflatex → bibtex → pdflatex → pdflatex`
  automatically.
- **Locally:**
  ```bash
  pdflatex main
  bibtex   main
  pdflatex main
  pdflatex main
  ```
  (or `latexmk -pdf main.tex`).

## Bibliography
- `references.bib` keys are readable slugs, e.g. `limgen2024`, `bagels`,
  `zhangfws2022`, `futuregen2025`, `science4cast`→`predictingfuture2022`,
  `chimera2025`, `ideasynth2025`. All 82 sheet papers are present, so you can
  `\cite{key}` any of them as the draft grows.
- Uses `natbib` + BibTeX with `unsrtnat`. To switch to IEEE style, replace the
  two lines in `main.tex`:
  `\usepackage[numbers,sort&compress]{natbib}` / `\bibliographystyle{unsrtnat}`
  with `\bibliographystyle{IEEEtran}` (and, for an IEEE Access submission, swap
  `\documentclass{article}` for the `IEEEtran` class).

## TODO before submission
- Fill in author names, affiliation, and supervisor in `main.tex`.
- Complete the abstract's final results sentence after the human study /
  graph-vs-random ablation (see `05_progress.tex`, P0/P1).
- Add figures (the funnel/cost diagrams exist as `docs/figures/*` in the repo).
- Sections 3.3–3.5 (graph / ideation / evaluation) describe *built* components;
  their empirical results are pending real runs — keep the honest framing.

## Provenance
- Sources: `artifacts/lit_review_papers.md` (parsed sheet), `references.bib`.
- Numbers: `docs/experiments/` (`results_registry.md`, `experiment_log.md`,
  `stage_c_output.md`, `bagels_output.md`), `docs/related_work_analysis.md`.
- Prose baseline: `docs/paper/thesis_draft.md` (Markdown mirror of this project).
