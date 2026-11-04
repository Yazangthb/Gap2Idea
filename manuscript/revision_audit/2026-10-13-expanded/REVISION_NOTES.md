# Expanded manuscript revision

13 October 2026. This revision responds to requests for richer sections, the original pipeline presentation, preservation of the supervisor-required comparison table, and a concrete team evidence handoff.

## Changes

- Expanded introduction: motivation, attribution/target definitions, structural challenges, prior extraction and generation work, rationale, RQs and contributions (approximately 1,080 words versus 570 in the shorter revision).
- Expanded related work: rhetorical annotation and context, clinical limitation recognition, future-work recognition/characterization, LLM extraction versus inferred/generated critique and explicit comparison dimensions (approximately 1,410 prose words versus 750).
- Restored all 13 original study rows in Table 1; added four relevant studies; retained comparison dimensions and added an explicit relevance column. Chronological ordering follows the cited version. The table spans three landscape pages with repeated headers.
- Restored the pipeline's three-column style, added collection/preparation and RQ analysis, and used exact cached SAL counts/configuration instead of obsolete venue-analysis RQ3 or approximate output counts.
- Expanded methodology with collection/preparation/label alignment, reasons for tools and stages, detector choices, verification context/batching, and metric/resampling rationale. Table 3 maps each RQ to its comparison, measures and justification.
- Expanded discussion with explicit RQ interpretation and implications for evidence-based research support. Existing numerical results remain tied to the saved artifacts.
- Added `TEAM_EVIDENCE_REQUEST.md`: nine claimable evidence-recovery cards, five conditional experiment cards, integration/review tasks, a ready-to-send team message and supervisor coverage map. Completed human/venue studies are recovery tasks, not presumed undone experiments.

## Validation

The isolated PDF has 30 pages, 28 cited works and ten numbered tables. All pages were rendered with Poppler and visually inspected, with detailed table/figure checks. The final LaTeX log has no warnings, unresolved citations/references, missing glyphs, overfull or underfull boxes. All fonts are Type1. `main.pdf` is byte-identical to the checked build; hash and source commit/status are in `VALIDATION.json`.

The clean code repository remains unchanged. No new model training or paid API inference was performed. The source-grounded metric checks from the preceding revision remain applicable because result artifacts and their generated numerical tables were not changed.

## Remaining work

The writing requirements are covered for the reported experiments. Submission readiness still depends on the evidence/provenance cards, whether the recovered human evaluation supports an independent frozen test, any baseline/front-end experiments necessary for the final claims, and eventual venue-specific formatting. Use the new team queue as the current detailed handoff; the earlier readiness note describes the preceding 20-page revision.

## Backups

`original/` here preserves the shorter revision before this expansion. The preceding `../2026-10-13/original/` preserves the pre-rewrite rich manuscript. Neither backup is overwritten.
