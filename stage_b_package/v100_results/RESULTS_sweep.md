# Stage B experiment 3 — frozen encoder sweep (train = Sci-Sentence + FWS + SAL)

**Date:** 2026-10-06 · V100 · same recipe as exp 2 (frozen encoder + class-weighted logreg,
train 7,706 rows, per-dataset paper-disjoint val). Only the sentence encoder changes.

## Gold (10 arXiv papers, 49 gaps) — the decision metric

| encoder | dim | recall_all (rules+head) | limitation | future_work | kept | val macro-F1 |
|---|---|---|---|---|---|---|
| bge-small (baseline) | 384 | 0.429 | 0.545 | 0.786 | 97 | 0.718 |
| bge-base | 768 | 0.408 | 0.455 | 0.786 | 76 | 0.742 |
| e5-base | 768 | 0.408 | 0.455 | 0.786 | 71 | 0.753 |
| mpnet | 768 | 0.571 | 0.455 | 0.786 | 186 | 0.668 |
| **MiniLM-L6** | 384 | **0.571** | **0.636** | **0.857** | 170 | 0.656 |
| SPECTER2 | 768 | — | — | — | — | — (failed: legacy .bin, needs safetensors convert) |

## Findings
1. **MiniLM-L6 is the new best head.** Highest gold recall_all (0.571, tied with mpnet) AND the best
   per-class recall — limitation **0.636** and future_work **0.857** — beating bge-small's 0.429/0.545/0.786.
2. **Recall/precision tradeoff is real.** MiniLM and mpnet reach 0.571 by keeping more candidate sentences
   (170–186 vs ~70–97). So they trade precision for recall. For a recall-first Stage B that feeds a Stage-C
   LLM precision filter, this is the right tradeoff; if Stage B must be precise on its own, bge-small/e5 keep
   fewer. (True precision isn't measurable on the positive-only gold — `kept` is the load proxy.)
3. **Validation macro-F1 is anti-correlated with gold here.** The highest-val encoders (e5 0.753, bge-base
   0.742) are the *worst* on gold (0.408), while the lowest-val MiniLM (0.656) is the best on gold. Judge on
   gold, never on in-distribution validation.
4. **Bigger ≠ better.** The two 384-dim models (bge-small, MiniLM) match or beat the 768-dim ones.

## Leaderboard (gold rules+head recall_all)
1. **MiniLM-L6 + logreg + SAL — 0.571** (limitation 0.636, future_work 0.857, kept 170) ← best
2. mpnet — 0.571 (limitation 0.455, kept 186)
3. bge-small + SAL — 0.429 (limitation 0.545, kept 97)  [most precise]
4. SciBERT / PubMedBERT FT — 0.327 · DistilBERT FT — 0.306

## Next
- Convert **SPECTER2** to safetensors and add it (scientific encoder — the one untested).
- Decide the operating point: **MiniLM** (max recall, feed Stage C) vs **bge-small** (fewer kept, more precise).
- Optional: a precision check — annotate a sample of MiniLM's extra kept sentences to quantify the tradeoff.

Files: `frozen_minilm.json`, `frozen_mpnet.json`, `frozen_bgebase.json`, `frozen_e5base.json` (+ exp 1/2 files).
