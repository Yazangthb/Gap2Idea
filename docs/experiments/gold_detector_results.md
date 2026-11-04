# Stage-B detector trained on the gap gold

Paper-level split (seed 42), stratified by domain: train 162 / val 34 / test 34 papers. Per-sentence 4-class (none/limitation/future_work/research_gap); frozen encoder + multinomial logreg; binary-gap threshold tuned on val. **Labels are LLM-ensemble gold (not yet human-certified).**

## Encoder sweep (binary gap = any type)

| encoder | tau | VAL F1 | TEST P | TEST R | TEST F1 | type-acc |
|---|---|---|---|---|---|---|
| bge-small **(best)** | 0.85 | 0.466 | 0.328 | 0.733 | 0.453 | 0.758 |
| pubmedbert | 0.85 | 0.457 | 0.327 | 0.681 | 0.442 | 0.723 |
| specter | 0.85 | 0.461 | 0.342 | 0.608 | 0.438 | 0.715 |
| _old pipeline_head_ | 0.50 | — | 0.174 | 0.658 | 0.275 | — |

## Best: bge-small (test)

Per-domain binary-gap P/R/F1 (test):
- arxiv_cs: P=0.314 R=0.734 F1=0.440
- biomed_sal: P=0.394 R=0.730 F1=0.512

Per-type detection recall (test): future_work 66.0% (n=97), limitation 68.2% (n=220), research_gap 78.7% (n=347)

Type accuracy among correctly-detected gaps: 75.8%

### Notes
- This is a **standalone per-sentence detector** at realistic prevalence (all test sentences), not the Stage-A-gated funnel; it isolates Stage B's ceiling.
- 'type-acc' = of gaps correctly flagged, fraction with the right type.
- Best head saved to `gap_head_gold.joblib` (bundle: clf + encoder); splits in `splits.json`.
- Test labels are LLM-gold; the human pass will re-certify the test set.
