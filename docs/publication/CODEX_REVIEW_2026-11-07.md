# Codex publication review — 2026-11-07

## Review scope

- Branch: `paper-writing-v2`
- Starting commit: `343c7f6`
- Reviewed material: the complete MDPI manuscript, both wrapper files, the bibliography, generated tables and figures, saved experiment summaries, data-split documentation, and the publication-review records available in the repository.
- Constraints followed: no new experiments, API calls, or metric recomputation from model inference; no changes outside `manuscript_mdpi/` and `docs/publication/`; no changes to the title, author list, affiliations, ORCID, CRediT placeholders, or other submission placeholders.
- Privacy: no private-data files or credentials were read, copied into the manuscript, or staged. No secret-bearing files are included in the change set.

## Reviewer assessment

The draft reports a useful and unusually well-instrumented comparison of funnel and whole-paper workflows, but several statements originally read as broader causal conclusions than the saved evidence supports. The main review intervention was therefore to separate observed results from general claims and to make the known comparison asymmetries visible at the point where each result is interpreted. No saved numerical result was found to be incorrectly transcribed.

The revised manuscript is suitable as a transparent preprint/submission draft provided that readers understand the remaining evidence limitations: most configurations are single runs; multiple confidence intervals are descriptive and uncorrected; the SAL comparison favours the funnel through a section gate; the FWS baseline uses a broader prompt; the arXiv whole-paper labels are the comparison reference rather than independently adjudicated gold labels; and the 4B/27B comparison changes model generation as well as size.

## Changes made

### `manuscript_mdpi/main.tex` and `manuscript_mdpi/preprint.tex`

- Rewrote the abstract in both wrappers and kept the abstract text identical.
- Preserved all reported headline values while adding the central uncertainty and design qualifications.
- Replaced causal or general wording with dataset-, prompt-, and model-scoped wording.
- Added a PDF-string-safe form of F1 in the MDPI wrapper so the final build no longer reports the related bookmark-string warning.
- Final abstract length: approximately 197 words.

### `manuscript_mdpi/sections/01_introduction.tex`

- Reorganized the stated findings under RQ1–RQ4 so each contribution is traceable to a research question.
- Scoped the claims to the evaluated datasets and prompts.
- Added the SAL non-significance and section-gate qualification, the FWS prompt mismatch, the arXiv whole-input label provenance, the concentration of the observed 4B advantage in longer papers, and the NLI precision-versus-F1 trade-off.
- Clarified that the label-checking experiment tests label plausibility rather than establishing independent ground truth.

### `manuscript_mdpi/sections/03_materials_methods.tex`

- Stated explicitly that the FWS whole-paper baseline uses a broader prompt, so that comparison cannot isolate input strategy alone.
- Stated explicitly that the SAL funnel alone receives the section gate, which structurally favours that workflow.
- Removed the unsupported server-package version. The repository's saved run records identify vLLM 0.32.0, but the client lock file does not reproduce the full server environment; the revised text now says so without inventing a PyTorch version.

### `manuscript_mdpi/sections/04_results.tex`

- Identified the secondary comparisons as analyses designed after the main results.
- Put the SAL gate, FWS prompt mismatch, and arXiv label provenance next to the affected results and captions.
- Marked the reported confidence intervals as uncorrected for multiple comparisons.
- Replaced statements of superiority with observed-difference language where uncertainty includes zero.
- Limited the small-model finding to the tested 4B and 27B checkpoints and made the model-generation confound explicit.
- Clarified that the length interaction is concentrated in the longer-paper region rather than universal.
- Reworded ambiguous “times fewer/less” expressions as direct ratios or fractions of the comparator.
- Clarified the sign of the plotted differences.

### `manuscript_mdpi/sections/05_discussion.tex`

- Scoped RQ2 to the evaluated datasets and prompt designs.
- Added the SAL, FWS, and model-generation qualifications to the RQ3 interpretation.
- Described the NLI classifier as viable but less precise, and disclosed that the reported classifier was selected using test-set results.
- Replaced an unsupported effect-size heuristic with the evidence-backed limitation that, except for one stability repeat, configurations are single runs and the intervals are uncorrected.

### `manuscript_mdpi/sections/06_conclusions.tex`

- Recast the conclusion as explicit answers to all four research questions.
- Kept numerical findings unchanged while distinguishing observation from generalization.
- Carried forward the test-set influence, SAL gate/non-significance, FWS prompt mismatch, arXiv label provenance, 4B/27B generation confound, and NLI precision/F1 limitations.

### `manuscript_mdpi/references.bib`

- Completed the LimGen proceedings metadata using the official Springer book record.
- Completed the ACM publication/location fields for the relevant conference records using their DOI records.
- Completed the PubMedBERT journal volume, issue, pages, and publisher metadata using the publication record.
- Normalized the ModernBERT model-card revision note, eliminating the bibliography line-breaking warning without changing the cited artifact.

### `manuscript_mdpi/figures/whole.pdf`

- Regenerated the figure from the repository's saved result files. Its dimensions and content remain consistent with the manuscript source data.

### `manuscript_mdpi/main.pdf` and `manuscript_mdpi/preprint.pdf`

- Rebuilt both deliverables from their revised wrappers with complete bibliography passes.
- Retained the generated PDFs in version control as requested, despite the repository's general PDF ignore rule.

### `docs/publication/CODEX_REVIEW_2026-11-07.md`

- Added this permanent audit trail of the reviewer findings, source checks, numerical checks, revisions, build validation, and remaining limitations.

## Numerical audit and corrections

No metric, confidence interval, count, timing, token total, cost, p-value, or sample size in the manuscript required a value correction. Regenerating the manuscript tables from the saved results produced no textual table-row differences.

Three numerical expressions were corrected for mathematical clarity; the underlying values did not change:

| Original expression | Revised interpretation | Source and check |
|---|---|---|
| “1.4 times fewer” | The whole-paper workflow used 1.4× the funnel quantity in the reported comparison. | Saved whole-paper/funnel cost results and the generated whole-cost table; the displayed values were divided directly. |
| “7.4/8.3 times fewer” | Funnel input was approximately one-seventh and one-eighth of whole-paper input. | `docs/experiments/llm_size/results.json` and the generated whole-cost table: approximately 554k/75k = 7.4 and 348k/42k = 8.3. |
| “1.5–7 times less” | The whole-paper prompt took approximately 1.5–7× as long per paper. | Saved timing fields in the same size-comparison results and the generated whole-cost table; ratios were expressed with the comparator and direction made explicit. |

Data cardinalities were independently checked from the saved split files and agree with the manuscript records:

- Gold validation/test: 34 papers and 11,201/11,741 sentences, with 699/664 positives.
- SAL development/test: 40 papers and 7,729/7,968 sentences, with 180/205 positives.
- FWS development/test: 418/209 papers and 3,005/1,506 sentences, with 418/220 positives.
- The documented FWS old-training-file SHA-256 prefix was reproduced as `f8d5fd58c5b9f1bb`.

## Verification performed

- Regenerated the paper table rows from saved results; all tracked textual table outputs were byte-identical to the committed versions.
- Regenerated the whole-paper figure from saved results.
- Built `main.tex` and `preprint.tex` independently with a full `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` sequence.
- Final journal build: 28 A4 pages, 2,021,409 bytes.
- Final preprint build: 27 A4 pages, 2,020,873 bytes.
- Both final logs contain no LaTeX errors, undefined citations/references, overfull boxes, underfull boxes, or BibTeX warnings.
- Extracted text from both PDFs contains no unresolved `??` markers.
- Rendered and visually inspected every page of both final PDFs. No clipping, overlap, missing figure, broken table, or unreadable reference layout was found.
- Verified that the two wrapper abstracts are exactly identical.
- Ran `git diff --check`; no whitespace errors were found.

## Remaining limitations and unresolved warnings

- The MDPI build retains two font-substitution warnings for bold `wasy` symbols at 7.6 pt and 6 pt during the class's first-page setup. These arise from the journal class/font combination, do not affect manuscript content, and produced no visible defect in the rendered PDF.
- Most configurations remain single runs; the manuscript now states this rather than implying run-to-run robustness.
- Confidence intervals are descriptive and uncorrected for multiplicity.
- The SAL and FWS comparisons retain their documented design asymmetries because no new experiments were authorized.
- The arXiv whole-paper labels remain the comparison reference and are not independently adjudicated gold labels.
- The small-versus-large model comparison still changes model generation as well as parameter count.
- The reported NLI classifier remains influenced by test-set selection.
- The full inference-server environment cannot be reconstructed from the client lock file; only the saved vLLM version is supported by the run records.
- The FWS source files match the documented shape, but a direct checksum against the historical acquisition artifact is unavailable; the manuscript does not claim such a comparison.
- Human validation and reproduced published-method comparisons remain future evaluation work, not completed evidence.
