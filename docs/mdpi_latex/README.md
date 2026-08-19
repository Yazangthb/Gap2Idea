# MDPI-styled draft (`docs/mdpi_latex/`)

Standalone, compilable working draft of the Gap2Idea paper — an **organizing scaffold**, not a
submission-ready file.

## Build
```bash
cd docs/mdpi_latex
latexmk -pdf main.tex     # -> main.pdf (4 pages)
```

## Conventions
- `\done{...}` (green) — results we have measured (real numbers from the experiments).
- `\pending{...}` (red ⟨PENDING: …⟩) — blanks still to fill. **These are the to-do list**;
  each maps to a Phase-2 step or a missing front-matter field.

## Two caveats
1. **Not the official MDPI class.** The proprietary `mdpi.cls` bundle can't be redistributed
   here, so this approximates the layout with the standard `article` class. For submission,
   open the official *MDPI LaTeX* template on Overleaf and paste the section bodies in — the
   content and `\cite` keys transfer directly (bib is shared with `../thesis_latex`).
2. **3 empty-author bib warnings** (`limgen2024`, `bagels`, `futuregen2025`) — those entries
   lack an `author` field in `references.bib`; fill them to silence the warnings.

## Source of truth
`references.bib` is copied from `../thesis_latex/references.bib`. The prose mirrors
`../paper/mdpi_draft.md` (Markdown version, easier to edit).
