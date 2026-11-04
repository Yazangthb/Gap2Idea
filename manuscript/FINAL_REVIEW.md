# Gap2Idea: final manuscript review

Review date: **4 November 2026**. Evidence authority: `Gap2Idea`, branch `minimal`, commit `28a6c2dbfad8ad332a66dd11e7b16f74d3ec4267`.

**Status: the branch-grounded manuscript is complete for final human review.** All tracked files and identified experiment/development/protocol families are accounted for. This does not certify missing experiments, model identities or annotations as completed, and does not guarantee acceptance or eliminate the need for the authors' scientific review.

## Delivered manuscript

- `main.tex` and the included section/table/figure files are current.
- `main.pdf`: 41 pages in the preserved article template, including the comparison table, supplementary experiments, prompts and references.
- Table 1 preserves all 13 original studies, four additional studies and the explicit relevance column. The requested three-column pipeline is retained.
- `EXPERIMENT_COVERAGE.md` maps 54 experiment/development/protocol entries to inclusion status, actual section/table numbers, source files and exclusion reasons. These are families, not 54 independent completed experiments.
- Main results distinguish full saved decisions, aggregate JSON, notebook/log development values and protocols without completed results. Supporting development tables are in the appendix so the main argument remains focused.

## Review passes and fixes

| Pass | Checks / finding | Resolution |
|---|---|---|
| Branch inventory | Read all experiment notebooks, result JSON/CSV, result summaries and relevant implementation; mapped all 72 tracked files | Coverage registry and source hashes recorded; old audit/README assertions do not override current artifacts |
| Numerical review | Matched SAL rows, sections, eight detector vectors and both complete 391-candidate verdict maps | Recomputed strict P/R/F1, stage counts and paired paper bootstrap; principal RCT F1 remains 0.658 |
| Supplementary numerical review | Replayed the eight-seed prevalence analysis and actual neighbor scorer; checked pilot counts, token sums and annotation-interface rows | Generated tables for these results; row checks and source hashes stored in verified_results.json |
| Configuration review | Multiple heads, entry points, thresholds, rule protection and batch defaults | Separate configuration table; final notebook MiniLM, runner bge and SAL binary PubMedBERT are not conflated |
| Localizer review | Terminal-window cap was previously described too broadly | Corrected: 160 applies to terminal-window selection; mid-paper and whole-section additions have no global cap |
| Prompt review | Intended gap-matched base conflicts with the shared authors'-own decision suffix | Both layers documented; affected runs labelled exploratory; source code and saved results not silently changed |
| Reference review | Datasheet uses model agreement and pending human validation | 230-paper statistics explicitly documented-only; 362 targets/40 papers verified; no completed human accuracy or agreement claimed |
| Claim consistency | Older union F1 0.701, projected corrected precision, macro/positive metric mixing, FWS pool mismatch, model-reference sensitivity and cost extrapolations | Separated or excluded with reasons; full-article rounded F1 corrected to approximately 0.427 |
| Literature review | Rechecked principal benchmark definitions and selected precise source claims against primary papers | No unsupported SOTA, universal transfer or systematic benchmark-noise claim; related-work comparison preserved |
| Build review | Stale source-directory auxiliary files affected cross-references under MiKTeX | Built from a dedicated source snapshot; all references resolve, no warnings or overfull/underfull boxes |
| Visual review | Rendered every final PDF page with Poppler | Inspected all 41 pages and detailed tables/figure/prompts; appendix tables remain with their explanatory sections |
| Final integrity | Compared reviewed PDF, compiled source snapshot and live paper files; rehashed all 72 research-repo files | Delivered PDF matches build exactly; source repo remains clean and unchanged |

## Results supporting the current conclusions

- SAL: 40 articles, 7,968 sentences, 205 labelled limitations. Stage A retains 203 positives among 1,885 sentences.
- PubMedBERT A+B: precision 0.460, recall 0.878, F1 0.604. Candidate-specific RCT verification: 269 kept, 156 TP, precision 0.580, recall 0.761, F1 0.658. The SAL keep-biased alternative has F1 0.652.
- At the reconstructed 20.7% section-pool prevalence, fixed-decision F1 is approximately 0.749 / 0.757 / 0.741 for detector / RCT / SAL. These are scoring sensitivities, not published-checkpoint reproductions.
- Estimated forty-paper input volume: 282,846 versus 75,622 tokens, ratio 3.7403. Actual total cost, throughput and retries are not established.
- FWS and Sci-Sentence breadth, classical baselines and development studies are included under their own label/input conditions. Their evidence status is visible in captions and the coverage matrix.

## Remaining matters for the authors' final review

These are external-evidence or research-design limits, not unresolved formatting errors. The draft already bounds its claims accordingly.

1. **Expanded reference provenance.** The raw 230-paper manifest, sentence labels, votes, adjudication records, generating scripts and API logs are absent. The documented model identifiers need verification against the actual requests. Confirm the datasheet values before presenting them as independently reproduced corpus results or claiming a released validated dataset. At present the appendix labels them documented-only.
2. **Completed human validation.** The 362-row interface and merge script exist, but no completed answers/private key/scored report are supplied. The sample covers biomedical disagreements, not independent validation of all arXiv papers. Its consensus-only scores would require a separate disagreement/adjudication and sampling discussion. The earlier 64-paper study is distinct and also unavailable here.
3. **Prompt composition.** If the team wants a clean claim about gap-matched verification, repair the shared wrapper in a new configuration, freeze the full request and evaluate it appropriately. The present paper accurately describes the existing conflicting configuration; no new API run was performed.
4. **Exact regeneration.** The frozen-head recipe is supplied, but raw training data and fine-tuned checkpoints are not. The historical RCT helper/prompt alias remains absent; current SAL/GAP prompt centralization does not recover it. Provider `latest` aliases are not pinned versions.
5. **Strength of comparison.** Current verifier choices used benchmark feedback. A genuinely independent frozen evaluation and a stronger chunked/retry-aware full-article baseline would support stronger generalization/efficiency claims. The manuscript does not claim those results already exist.
6. **Unavailable studies.** No results for the earlier 800-paper venue corpus, downstream idea quality or the earlier 64-paper study were inferred from the branch. Restore their actual artifacts before adding their conclusions.
7. **Submission-specific decisions.** Confirm author order/affiliations, funding/conflict declarations, artifact accessibility and redistribution permissions, then select the venue and apply its format and disclosure requirements. No venue or author declarations were invented.

## Reproduction and backups

From the workspace root:

```powershell
python -X utf8 "em.dcs clean gap2idea/revision_audit/verify_minimal.py"
```

This performs offline saved-artifact verification and generates the numerical tables. It does not retrain heads, modify the research repository or spend API credits.

- `revision_audit/2026-11-04/verified_results.json`: metrics and every tracked source hash.
- `revision_audit/2026-11-04/experiment_registry.json`: machine-readable coverage.
- `revision_audit/2026-11-04/VALIDATION.json`: final build/integrity checks and PDF hash.
- `revision_audit/2026-11-04/build/`: exact compiled source snapshot, logs and checked PDF.
- `revision_audit/2026-11-04/render/`: final rendered pages and contact sheets.
- `revision_audit/2026-11-04/original/`: preceding manuscript preserved before this rewrite.

A successful build and complete inventory establish document integrity and coverage. They do not turn exploratory or model-labelled evidence into an independently validated scientific result.
