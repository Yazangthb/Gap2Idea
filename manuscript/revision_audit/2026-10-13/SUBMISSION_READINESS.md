# Manuscript revision and submission readiness

Revision date: 13 October 2026. Authority: `../Gap2Idea`, branch `minimal`, commit `66c237a` (full hash in `verified_results.json`). The earlier project audit and copied `code/`/`data/` inside the paper folder are historical material, not the authority for this revision.

## Completed in this revision

- Rewrote the abstract and all six manuscript sections around actual clean-branch experiments.
- Preserved the article template, authors, affiliations, margins, section files, natbib and TikZ workflow. Latin Modern replaces bitmap-only Computer Modern font instances while preserving the same typography family.
- Replaced unsupported proposal-style AI/DS sampling/annotation/venue claims with the evaluated extraction scope. Those completed-but-unavailable studies are preserved in `DEFERRED_STUDIES.md` and the original backup.
- Corrected Stage A to the section-aware union and distinguished it from the SAL Discussion/Conclusion gate.
- Distinguished the runner's bge/self-distilled head, the notebook's MiniLM/mixed-public-data head and SAL's in-domain cached binary detector vectors.
- Recomputed SAL A/B/C scores from every saved prediction/verdict ID, generated two LaTeX tables and recorded source hashes.
- Added a descriptive 10,000-resample paired paper bootstrap. It does not remedy prompt/test selection.
- Used the checkable candidate-specific SAL result (F1 0.658), not the earlier union-candidate 0.701 headline.
- Summed the input accounting CSV: 282846/75622 = 3.7403; reported estimated input volume, not a measured dollar saving or speedup.
- Corrected the full-paper baseline F1 to approximately 0.427 from saved rounded precision/recall; older prose's 0.429 does not follow those values.
- Retained FWS/Sci-Sentence aggregate evidence with target/split/provenance limits; excluded mixed-ontology overall recall on the 49-item internal reference.
- Added six missing bibliography entries and checked the cited source definitions against original papers/publisher or author resources. Kept FWS's verified 2022 arXiv version rather than guessing journal metadata.
- Added the exact surviving SAL keep-biased prompt and few-shot examples.
- Preserved originals and compiled/rendered the updated PDF.

## What still prevents calling this submission-ready

| Item | Required concrete evidence/action |
|---|---|
| Human/venue results | Supply the already-completed studies if their claims are to be included. They are intentionally excluded now. An independent frozen evaluation would strengthen the present exploratory verifier study. |
| Training provenance | Recover checkpoint/tokenizer hashes, splits, thresholds, seeds and training configs for each cached SAL detector. A clean regenerator of `sal_test_allmodels.json` is absent. |
| Exact API-run prompts/version | Archive the missing historical `SYSTEM_RCT_LOOSE` helper and the model/version actually resolved by Yandex's `latest` alias. The current RCT instruction is not byte-identical to the older helper found in cached history; do not silently substitute it. |
| Missing prediction-level results | Recover FWS/Sci-Sentence row decisions and full-paper responses/failures. Current aggregate files do not support independent paired tests or error adjudication. |
| Baseline robustness | Repeat or clearly retain the ten-paper baseline as a limited pilot; chunking/retry differences and two parse failures confound the mechanism claim. |
| Total efficiency | Input estimates are supported. Actual total runtime/billing, output/retry tokens and detector work are not supported by the accounting CSV alone. |
| Release and venue | Make experiment scripts portable, document data/checkpoint access, then choose venue and apply its template, author/declaration and anonymity requirements. No venue-policy claim was made in this revision. |

The LaTeX draft is complete for the available evidence, but evidence recovery/independent testing and venue-specific requirements remain. A successful build does not resolve these scientific limitations. No new paid inference or training was run.

## Offline table reproduction

Run from the workspace root:

```powershell
python -X utf8 "em.dcs clean gap2idea/revision_audit/verify_results.py"
```

The script writes `verified_results.json` and the two generated SAL tables. The source files in the main code repo are read-only. For a normal paper build, run pdflatex, bibtex, then pdflatex twice from the paper directory. The final QA build is isolated under `build/` to avoid concurrent editor builds sharing aux/log files.

## Final build validation

20 pages. All pages rendered and visually checked; detailed figure/table inspection passed. No LaTeX warnings, missing citations/references, missing glyphs or overfull boxes in the isolated final log. All PDF fonts are Type1. Final `main.pdf` matches the checked build byte-for-byte. The source repo Git status is unchanged.
