# Codex publication review — 2026-11-07

## Review scope

- Branch: `paper-writing-v2`
- Starting commit: `343c7f6`
- Reviewed material: the complete MDPI manuscript, both wrapper files, the bibliography, generated tables and figures, saved experiment summaries, data-split documentation, and the publication-review records available in the repository.
- Constraints followed: no new experiments, API calls, or metric recomputation from model inference; no changes outside `manuscript_mdpi/` and `docs/publication/`; no changes to the title, author list, affiliations, ORCID, CRediT placeholders, or other submission placeholders.
- Privacy: no private-data files or credentials were read, copied into the manuscript, or staged. No secret-bearing files are included in the change set.

## Reviewer assessment

The draft reports a useful and unusually well-instrumented comparison of funnel and whole-paper workflows, but several statements originally read as broader causal conclusions than the saved evidence supports. The main review intervention was therefore to separate observed results from general claims and to make the known comparison asymmetries visible at the point where each result is interpreted. No saved numerical result was found to be incorrectly transcribed.

The revised manuscript is suitable as a transparent preprint/submission draft provided that readers understand the remaining evidence limitations: most configurations are single runs; multiple confidence intervals are descriptive and uncorrected; the SAL comparison favours the funnel through a section gate; the FWS baseline uses a broader prompt; the arXiv whole-paper labels are the comparison reference rather than independently adjudicated gold labels; and the 27B checkpoint differs from the 9B and 4B checkpoints in model generation as well as size.

## Changes made

### `manuscript_mdpi/main.tex` and `manuscript_mdpi/preprint.tex`

- Rewrote the abstract in both wrappers and kept the abstract text identical.
- Preserved all reported headline values while adding the central uncertainty and design qualifications.
- Replaced causal or general wording with dataset-, prompt-, and model-scoped wording.
- Added a PDF-string-safe form of F1 in the MDPI wrapper so the final build no longer reports the related bookmark-string warning.
- Integrated the completed Qwen3.5-9B comparison while retaining the SAL, FWS, label-provenance, and uncertainty qualifications.
- Final abstract length: approximately 196 words.

### `manuscript_mdpi/sections/01_introduction.tex`

- Reorganized the stated findings under RQ1–RQ4 so each contribution is traceable to a research question.
- Scoped the claims to the evaluated datasets and prompts.
- Added the SAL non-significance and section-gate qualification, the FWS prompt mismatch, the arXiv whole-input label provenance, the 9B and 4B long-input results, and the NLI precision-versus-F1 trade-off.
- Clarified that the label-checking experiment tests label plausibility rather than establishing independent ground truth.

### `manuscript_mdpi/sections/02_related_work.tex`

- Integrated a supplementary LR-AI-assisted literature search from commit `336a23e2c999150309b8419b0c312d69eea3cf63`.
- Disclosed the use of LR-AI for search, deduplication and title/abstract screening, while stating that retained citations were checked against primary records.
- Added Hu and Wan's early rule-based future-work extraction and classification study.
- Added Ma et al.'s small-model-filter/LLM-reranker architecture as the closest general information-extraction analogue.
- Positioned the funnel relative to FrugalGPT's model-cascade framing, RULER's long-context benchmark, order-preserving retrieval versus full-context input, and Si et al.'s controlled study of LLM research ideation.
- Added Hu and Wan and Ma et al. to the selected-related-work table; broader background papers remain in prose to avoid presenting unlike tasks as direct extraction benchmarks.

### `manuscript_mdpi/sections/03_materials_methods.tex`

- Stated explicitly that the FWS whole-paper baseline uses a broader prompt, so that comparison cannot isolate input strategy alone.
- Stated explicitly that the SAL funnel alone receives the section gate, which structurally favours that workflow.
- Removed the unsupported server-package version. The repository's saved run records identify vLLM 0.32.0, but the client lock file does not reproduce the full server environment; the revised text now says so without inventing a PyTorch version.
- Added the Qwen3.5-9B revision, precision, and serving setup from the saved run record, and stated that all whole-paper calls completed.
- Added the LR-AI literature-review support disclosure to the software and generative-AI subsection.

### `manuscript_mdpi/sections/04_results.tex`

- Identified the secondary comparisons as analyses designed after the main results.
- Put the SAL gate, FWS prompt mismatch, and arXiv label provenance next to the affected results and captions.
- Marked the reported confidence intervals as uncorrected for multiple comparisons.
- Replaced statements of superiority with observed-difference language where uncertainty includes zero.
- Added all 9B whole-paper, published-funnel, fine-tuned-Stage-B, and one-call-per-paper results.
- Limited the small-model finding to the tested 9B and 4B checkpoints and retained the generation confound when comparing them with the 27B checkpoint.
- Clarified that the length interaction is concentrated in the longer-paper region rather than universal.
- Reworded ambiguous “times fewer/less” expressions as direct ratios or fractions of the comparator.
- Clarified the sign of the plotted differences.

### `manuscript_mdpi/sections/05_discussion.tex`

- Scoped RQ2 to the evaluated datasets and prompt designs.
- Added the SAL, FWS, and model-generation qualifications to the RQ3 interpretation, and used the same-generation 9B/4B comparison to show that the long-input pattern is not unique to 4B.
- Described the NLI classifier as viable but less precise, and disclosed that the reported classifier was selected using test-set results.
- Replaced an unsupported effect-size heuristic with the evidence-backed limitation that, except for one stability repeat, configurations are single runs and the intervals are uncorrected.

### `manuscript_mdpi/sections/06_conclusions.tex`

- Recast the conclusion as explicit answers to all four research questions.
- Kept numerical findings unchanged while distinguishing observation from generalization.
- Carried forward the test-set influence, SAL gate/non-significance, FWS prompt mismatch, arXiv label provenance, three-model generation confound, and NLI precision/F1 limitations.

### `manuscript_mdpi/references.bib`

- Completed the LimGen proceedings metadata using the official Springer book record.
- Completed the ACM publication/location fields for the relevant conference records using their DOI records.
- Completed the PubMedBERT journal volume, issue, pages, and publisher metadata using the publication record.
- Normalized the ModernBERT model-card revision note, eliminating the bibliography line-breaking warning without changing the cited artifact.
- Added primary-record metadata for Hu and Wan (2015), Ma et al. (2023), FrugalGPT, RULER, order-preserving RAG, and Si et al. (2025).
- Added the requested LR-AI software citation: Alnakri, Orabi, Mohammed and Alnoukari (2026), Zenodo version 1, DOI `10.5281/zenodo.24018390`.

### `manuscript_mdpi/figures/whole.pdf`

- Regenerated the figure from the repository's saved result files and added the 9B bars and paired differences.

### `manuscript_mdpi/tables/whole_table.tex`, `wholecost_table.tex`, and `variants_table.tex`

- Regenerated all three tables with the existing table generator and the committed 9B verdicts.
- Added the 9B accuracy, paired confidence intervals, token counts, output counts, timings, and funnel-format variants without hand transcription.

### `manuscript_mdpi/main.pdf` and `manuscript_mdpi/preprint.pdf`

- Rebuilt both deliverables from their revised wrappers with complete bibliography passes.
- Retained the generated PDFs in version control as requested, despite the repository's general PDF ignore rule.

### `docs/publication/CODEX_REVIEW_2026-11-07.md`

- Added this permanent audit trail of the reviewer findings, source checks, numerical checks, revisions, build validation, and remaining limitations.

## Numerical audit and corrections

No previously reported metric, confidence interval, count, timing, token total, cost, p-value, or sample size required a value correction. The follow-up revision adds new 9B results rather than correcting the earlier 27B or 4B results.

The 9B source is experiment commit `657bd665258e8da25e7cf5d6c047589a0a281ef0`. The generator read the verdict files and `docs/experiments/llm_size/results.json` at that commit. The principal added $F_1$ values are:

| Set | Mode | Whole paper | Published funnel | Difference |
|---|---|---:|---:|---:|
| arXiv | High precision | 0.313 | 0.599 | +0.286, 95% CI [-0.03, +0.42] |
| arXiv | Balanced | 0.236 | 0.564 | +0.328, 95% CI [+0.04, +0.41] |
| arXiv | High recall | 0.403 | 0.556 | +0.153, 95% CI [-0.06, +0.25] |
| SAL-40 | High precision | 0.684 | 0.690 | +0.007, 95% CI [-0.07, +0.09] |
| SAL-40 | Balanced | 0.680 | 0.684 | +0.004, 95% CI [-0.07, +0.09] |
| SAL-40 | High recall | 0.665 | 0.607 | -0.059, 95% CI [-0.15, +0.03] |
| FWS | High precision | 0.654 | 0.731 | +0.077, 95% CI [+0.03, +0.12] |
| FWS | Balanced | 0.654 | 0.725 | +0.071, 95% CI [+0.03, +0.11] |
| FWS | High recall | 0.654 | 0.728 | +0.074, 95% CI [+0.04, +0.11] |

For the 9B high-precision runs, the saved cost record reports 553,834 arXiv whole-paper input tokens and 92 seconds, versus approximately 403,000 input tokens and 287 seconds for the 20-per-call published funnel. The fine-tuned one-call-per-paper funnel uses 74,997 input tokens and 32 seconds. The manuscript therefore distinguishes input reduction from latency: the published call format reduces input but produces more output and is slower, whereas the one-call format reduces both input and time.

## LR-AI literature-search follow-up

The latest related-work push, commit `336a23e2c999150309b8419b0c312d69eea3cf63`, records 208 unique OpenAlex/arXiv records, 204 passing automated filters, and 88 retained after title/abstract screening (32 core and 56 skim). The run notes describe it as a quick supplementary search, not a complete PRISMA-style systematic review. The manuscript therefore uses it to broaden coverage and does not make systematic-review completeness claims.

Every newly cited technical claim was checked against a primary record: ACL Anthology for Ma et al.; arXiv records for Hu and Wan, FrugalGPT, RULER and order-preserving RAG; and the ICLR/OpenReview paper for Si et al. The LR-AI bibliographic entry follows the citation supplied by the authors. Candidate summaries alone were not treated as evidence.

Six substantive references were selected because each fills a distinct gap in the original review. Other recommended items were not added when they duplicated an existing citation, addressed a more distant task, or lacked enough verified information to improve the manuscript proportionately.

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

- Regenerated the paper table rows from saved results. The prior 27B and 4B cells remained unchanged, and the generator inserted the new 9B cells.
- Regenerated the whole-paper figure with 27B, 9B, and 4B bars.
- Built `main.tex` and `preprint.tex` independently with a full `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` sequence.
- Final journal build after the 9B and literature-review integrations: 28 A4 pages, 2,032,736 bytes.
- Final preprint build after the 9B and literature-review integrations: 28 A4 pages, 2,033,647 bytes.
- Visually inspected all 56 rendered pages across both final PDFs; the expanded related-work table is legible and contained, and the bibliography has no clipping or overlap.
- Both final logs contain no LaTeX errors, undefined citations/references, overfull boxes, underfull boxes, or BibTeX warnings.
- Extracted text from both PDFs contains no unresolved `??` markers.
- Verified that the two wrapper abstracts are exactly identical.
- Ran `git diff --check`; no whitespace errors were found.

## Remaining limitations and unresolved warnings

- The MDPI build retains two font-substitution warnings for bold `wasy` symbols at 7.6 pt and 6 pt during the class's first-page setup. These arise from the journal class/font combination, do not affect manuscript content, and produced no visible defect in the rendered PDF.
- Most configurations remain single runs; the manuscript now states this rather than implying run-to-run robustness.
- Confidence intervals are descriptive and uncorrected for multiplicity.
- The SAL and FWS comparisons retain their documented design asymmetries because no new experiments were authorized.
- The arXiv whole-paper labels remain the comparison reference and are not independently adjudicated gold labels.
- The 9B and 4B checkpoints share the Qwen3.5 generation, but the 27B checkpoint is Qwen3.8; the three-model comparison still cannot isolate parameter count from all model-design differences.
- The reported NLI classifier remains influenced by test-set selection.
- The full inference-server environment cannot be reconstructed from the client lock file; only the saved vLLM version is supported by the run records.
- The FWS source files match the documented shape, but a direct checksum against the historical acquisition artifact is unavailable; the manuscript does not claim such a comparison.
- Human validation and reproduced published-method comparisons remain future evaluation work, not completed evidence.
