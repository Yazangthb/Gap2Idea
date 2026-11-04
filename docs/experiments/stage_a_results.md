# Stage A — trained structural localizer vs rules

**What I did:** looked at the corpus (gap distribution by section + position), then trained a cheap,
recall-first Stage-A localizer on CHEAP structural features only (normalized position, section,
length — no embeddings) with a paper-level domain-stratified split. Model: HistGradientBoosting
(captures section x position interactions). Compared to the current rule-based `slice_section_aware`
on the held-out TEST papers. Lower retention = more sentences dropped. Labels = LLM-gold, all gap types.

## Corpus findings (train split)
- **Section labels exist for biomed (40/40) but NOT arXiv (0/190)** — the root cause of weak arXiv localization.
- biomed gaps by section: DISCUSS 63% | INTRO 14.5% | METHODS 8% | RESULTS 5.5% | CONCL 4% | ABSTRACT 4%.
  Zero-gap (droppable): TITLE, TABLE, FIG, COMP_INT, AUTH_CONT, ACK_FUND, SUPPL.
- Position: biomed gaps cluster at END (65% last 3 deciles) + front (19%); **arXiv gaps are spread** (30% front, 25% end, rest mid).

## Recall at fixed retention (test) — trained GBM
| set | keep 15% | 20% | 30% | 40% | 55% |
|---|---|---|---|---|---|
| ALL | 46 | 55 | 67 | 78 | 87 |
| arXiv | 43 | 50 | 65 | 76 | 86 |
| biomed | 48 | 57 | 76 | 85 | 90 |

Rule-slice single operating point: **ALL R=47% @ keep16%** · arXiv R=41% @ keep14% · biomed R=69% @ keep29%.

## Conclusion
- **A trained structural Stage A ~ ties the hand-tuned rules** at the aggressive operating point (ALL ~46-47% recall @ ~15% retention); it gives a small **biomed edge** (76% vs 69% at ~30% keep) and, crucially, a **tunable recall knob** the rules lack.
- **It does NOT transform arXiv** — structural features can't localize spread-out arXiv gaps without section labels. To reach high recall you must keep most of the paper.
- **The real Stage-A lever is section segmentation for arXiv** (e.g., GROBID section trees; the funnel already supports `slice_grobid_regions`), not a learned model over weak features.
- **Practical setting:** since Stage B is cheap, run Stage A **recall-first** (keep ~40% → ~78% recall ALL / 76% arXiv / 85% biomed), lifting the funnel's recall ceiling from ~62%, and let the improved Stage B + Stage C filter.

Model saved: `data/gold/stage_a_model.joblib` (HistGradientBoosting; dv + clf). Split: `splits.json`.
