# MDPI layout review - 2026-11-06

Scope: formatting in the separate `Gap2Idea-paper-review` worktree only. The
Task 5b checkout, source files and running experiments were not modified.

## Template verification

- The manuscript uses `Definitions/mdpi` with `make,article,submit,pdftex,moreauthors`.
  The bundled class identifies itself as v6.5d, 2026-10-19, and resolves `make`
  to Machine Learning and Knowledge Extraction. The stale September-version
  comment in `main.tex` was corrected; the class file itself was not changed.
- The MDPI-authored [official Overleaf template](https://www.overleaf.com/latex/templates/mdpi-article-template/fcpwsspfzsph)
  uses the same submission-class convention. Its source explicitly demonstrates
  ordinary text-width objects and optional wide objects through
  `adjustwidth{-\extralength}{0cm}` / `\fulllength`.
- The [official example PDF](https://www.overleaf.com/latex/templates/mdpi-article-template/fcpwsspfzsph.pdf)
  was rendered and compared visually. Its body is likewise offset to the right
  to reserve the left publication-information column. The local class sets left
  5.87 cm, right 1.27 cm, and an extra-width allowance of 4.61 cm. That asymmetry
  is intentional, not an accidental geometry override.
- No symmetric-margin patch, journal-option substitution or class edit was made.
  This verifies the journal-class selection and layout usage, not byte-for-byte
  identity with the newest downloadable class release or final submission readiness.

## Changes

- All result plots now stay within the normal text measure. The pipeline was
  already within that measure. The mode-results and SAL paired-comparison tables
  now also use text width, with wrapped confidence intervals and adequate row
  height rather than small scaled-down text.
- Wide layout is retained only for the information-dense related-work,
  evaluation-data and eight-column cost tables. Those tables need the additional
  width to retain their content at a readable size, and MDPI permits it.
- Local float placement uses `htbp` instead of forcing every object with `H`.
  Float-only pages are top-aligned with fixed separation, eliminating artificially
  stretched gaps between figures and tables. The header receives 20 pt of height
  for compatibility with the current fancyhdr package; horizontal geometry is
  unchanged. Abstract math uses PDF-string-safe alternatives for metadata.
- Related-work Table 1 is preserved in full. It has larger text, more space
  between entries, and adjusted column widths. The two Lan 2024 studies are
  identified as SAL and multi-label work. The column is named `Target categories`
  because some rows concern broader rhetorical labels rather than gap statements.
- Category wording/capitalisation is consistent: `Limitations`, `Future work`,
  `Research gaps`, ordered in that sequence when several apply. Sci-Sentence's
  two categories now read `Limitations; Research gaps`. Knowledge-gap and broader
  rhetorical tasks retain their distinct terminology; no task equivalence is invented.

## Validation

- Built with `latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error
  -outdir=build main.tex`; final output has 19 pages.
- Rendered the whole document for overview inspection, then inspected the first
  page, complete related-work table, plot/table pages, and the narrowed tables at
  readable resolution. Checked the final layout after the last changes.
- Final log: no overfull/underfull boxes, no oversized floats, no undefined
  references/citations, no PDF-string warnings and no header-height warning.
  Two pre-existing bold-symbol font-substitution warnings remain; their rendered
  corresponding-author symbol is present and legible.
- The bundled class, source figure PDFs and generated numeric table rows are
  unchanged. No experiment number, research claim, citation entry or study was
  removed. The new PDF replaces the local manuscript preview; nothing was pushed.

The draft still contains pending scientific/submission items already identified
by its authors. This review addresses the requested template and presentation
issues only.
