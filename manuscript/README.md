# Gap2Idea extraction manuscript

Current source: `main.tex`; reviewed PDF: `main.pdf`. The existing 11pt A4 article template, 2.5 cm margins, one-and-a-half spacing, author block, natbib and three-column TikZ pipeline are retained. Venue formatting remains a later step.

## Current evidence base

The 4 November 2026 revision uses `../Gap2Idea`, branch `minimal`, commit `28a6c2dbfad8ad332a66dd11e7b16f74d3ec4267`. The copied historical `code/` and `data/` inside this manuscript folder are not the authority for current results.

- `EXPERIMENT_COVERAGE.md`: complete experiment-to-paper inclusion/exclusion matrix and all 72 tracked source files.
- `FINAL_REVIEW.md`: review outcome, contradictions addressed and evidence issues for final human review.
- `sections/01_introduction.tex` through `06_conclusion.tex`: current manuscript.
- `sections/07_appendix.tex`: supplementary development experiments, reference construction, configurations and prompt provenance.
- `sections/prompts_sal.tex`, `prompt_gap.tex`, `prompt_wrapper.tex`: surviving instructions and the shared decision suffix.
- `tables/related_work_comparison.tex`: supervisor-required Table 1, preserving all 13 original studies plus four additions and an explicit relevance column.
- `tikz/pipeline.tex`: requested three-column workflow from collection to RQ analysis; exact SAL example.
- `revision_audit/verify_minimal.py`: current offline verifier; writes generated numerical tables, source hashes and current verification results without API inference or training.
- `revision_audit/2026-11-04/`: current verification manifest, experiment registry, build, rendered pages and backup of the preceding revision.
- `TEAM_EVIDENCE_REQUEST.md`: earlier handoff, now superseded for current status by the coverage matrix and final review. Its 64/800 recovery tasks remain distinct from the newer annotation protocol.

The principal SAL result is candidate-specific F1 **0.658** (RCT prompt), not the older union-run **0.701**. The documented 230-paper model-labelled corpus and 362-row annotation interface do not supply completed human-validation results.

## Reproduce saved-artifact metrics

From the workspace root:

```powershell
python -X utf8 "em.dcs clean gap2idea/revision_audit/verify_minimal.py"
```

This checks candidate coverage, strict scores, paper bootstrap, prevalence sensitivity, positional sensitivity, aggregate consistency, annotation-interface counts and all tracked source hashes. It regenerates the paper's machine-generated numerical tables. It does not call an API, retrain models or change the code repository.

## Build

A normal build from this directory is:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

For isolated QA, copy `main.tex`, the bibliography, and `sections/`, `tables/`, `tikz/` into a dedicated build directory and run all commands **from that directory**. Merely specifying an output directory can cause this MiKTeX installation to read stale auxiliary files from the source working directory.

Preserved earlier revisions are under `revision_audit/2026-10-13/` and `2026-10-13-expanded/`; the 4 November `original/` folder preserves the preceding manuscript. Do not overwrite these backups when updating the draft.
