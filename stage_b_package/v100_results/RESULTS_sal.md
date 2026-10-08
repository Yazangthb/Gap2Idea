# Stage B experiment 2 — Sci-Sentence + FWS + **SAL**, per-dataset validation + PubMedBERT

**Date:** 2026-10-06 · V100 · train 7,706 rows · paper-disjoint val **per dataset**.

Added SAL (biomedical limitation data, capped: 567 limitation + 1,828 none) to fix the limitation
bottleneck. Each dataset contributes its own paper-disjoint validation split; we report overall +
per-dataset macro-F1, plus end-to-end recall on the 10 arXiv gold papers.

## Validation (in-distribution) — macro-F1

| | sci | fws | sal | **overall** |
|---|---|---|---|---|
| **Frozen bge-small + logreg** | 0.638 | 0.532 | 0.487 | **0.718** |
| **PubMedBERT (full FT)** | 0.922 | 0.594 | 0.921 | **0.890** |

(fws has no limitation class, sal has no future_work class → those per-dataset macros are over the labels present.)

## End-to-end on 10 gold arXiv papers (49 gaps)

| | rules+head recall_all | limitation | future_work | head-only recall_all | kept |
|---|---|---|---|---|---|
| Frozen (SciSent+FWS, no SAL) | 0.429 | 0.455 | 0.786 | 0.347 | 104 |
| **Frozen + SAL** | **0.429** | **0.545** | 0.786 | 0.388 | 97 |
| **PubMedBERT + SAL (FT)** | 0.327 | 0.364 | 0.714 | 0.245 | 65 |

## Findings
1. **SAL fixed the limitation class (frozen):** gold limitation recall **0.455 → 0.545** (rules+head) and
   head-only limitation **0.273 → 0.364**, with a *tighter* kept set (104 → 97). Overall held at 0.429
   (SAL trades a few scope/assumption hits for cleaner limitation ones). Goal achieved.
2. **The fine-tune-overfits pattern repeats a 3rd time.** PubMedBERT scores a huge **0.890 overall
   validation macro-F1** (sci 0.922, sal 0.921!) yet transfers **worse** to gold (0.327 vs frozen 0.429).
   Across DistilBERT, SciBERT, and now PubMedBERT, full fine-tuning wins in-distribution and loses on our
   real arXiv task. **Frozen bge + logreg remains the best Stage-B head.**
3. **Validation ≠ task.** The per-dataset validation numbers are almost useless as a proxy for gold recall —
   the biomedical SAL/FWS validation is trivially high for a fine-tuned model but doesn't reflect arXiv
   gap extraction. Always judge on the gold.

## Leaderboard so far (gold rules+head recall_all)
1. **Frozen bge + logreg + SAL — 0.429** (limitation 0.545) ← best
2. Frozen bge + logreg (SciSent+FWS) — 0.429 (limitation 0.455)
3. SciBERT FT — 0.327 · PubMedBERT FT — 0.327 · DistilBERT FT — 0.306

Files: `frozen_sal.json`, `pubmedbert_sal.json` (+ earlier `frozen_result.json`, `scibert_result.json`, `bert_result.json`).
