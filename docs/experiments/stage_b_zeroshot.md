# Stage B — zero-shot NLI cross-encoder baseline (no training)

Model: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`. Same split/test set as the trained head (test 11741 sents, 664 gaps). Gap = any type; τ tuned on VAL for binary F1; realistic prevalence.

| detector | τ | TEST P | TEST R | TEST F1 |
|---|---|---|---|---|
| **zero-shot NLI** | 0.4 | 0.419 | 0.625 | **0.502** |
| trained frozen bge-small + logreg (ref) | 0.85 | 0.328 | 0.733 | 0.453 |

VAL F1 at τ=0.4: 0.490

Per-domain (test): arxiv_cs P=0.388 R=0.643 F1=0.484, biomed_sal P=0.644 R=0.555 F1=0.596

Per-type recall (test): future_work 49.5% (n=97), limitation 49.5% (n=220), research_gap 74.4% (n=347)

Type accuracy among detected gaps: 44.3%

### Notes
- Zero-shot = no gold training; gold-independent. A reference point for whether a cross-encoder (joint attention over hypothesis+sentence) beats the frozen bi-encoder.
- τ tuned on val (a hyperparameter, not training) to match the trained head's protocol.
- Single sentence, no document context — a known handicap for this task.
