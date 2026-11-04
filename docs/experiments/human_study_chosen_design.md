# Gap2Idea human study — CHOSEN design (decided 2026-10-27)

Decision record for the human-labelling study. **Chosen: 2 raters + 1 external tie-breaker, ~94
pooled items (biomed/SAL first).** Full rationale and the E5 lineage are in
[human_study_design.md](human_study_design.md); this file freezes the operational choice so we don't
re-litigate it. **Status: design frozen, NOT implemented, no labelling started.** Freeze the sample
key + SHA-256 and the hypotheses before any rater sees an item; log dated deviations.

## What we validate (settled)
We human-validate **our gold's labels**, not SAL (already human) and not a moving pipeline. One
**pooled, blind** round, biomed/SAL papers first (that is where the SAL comparison lives; arXiv gold
is a later round). The same labels answer three questions at once:
1. is the gold a valid benchmark (precision + recall vs humans)?
2. what is the funnel's **true** precision (measured against humans, gold-independent)?
3. does SAL **under-label** (our extras are real)?

## The design in one table
| | |
|---|---|
| **Primary hypothesis** | Share of our **SAL-extras** that are real gaps: H₀ p ≤ 50% vs H₁ > 50%, one-sided, α = 0.05 |
| **Power** | 80% at a true share of 70% (effect 0.20); n ≈ 37 suffices for the exact binomial test |
| **Raters** | **2 independent raters per item** + **1 external blind tie-breaker** → majority-of-three |
| **Blindness** | No source / tier / SAL-membership / control-role shown; private key never embedded in the page |
| **Validity gate** | Claim only if ≤ 2 of 12 decoys are labelled "gap" (leaning "gap" fails the gate) |
| **Adjudication** | Tie-breaker rates every disagreement, blind to the two votes; leftovers decided by the two raters + tie-breaker; unresolved → *not a gap*. No author decides. |
| **Calibration** | None (raters use the guideline + type definitions + shown context) |
| **Analysis** | weighted share + one-sided 95% lower bound; Rogan–Gladen corrected share (robustness only); Krippendorff's α + % agreement; gate |

## Pooled strata — minimized item counts
| stratum | n | role |
|---|---|---|
| **A. SAL-extras** (gold `limitation`, not in SAL) | **40** | primary test + "SAL under-labels" claim |
| **B. pipeline positives** (funnel+Stage C, gold `none`) | **30** | **true precision** of the funnel, gold-independent (CI ±~15%) |
| **C. positive controls** (high-confidence gold gaps) | **12** | gold precision (strong tier) + rater sanity |
| **D. decoys** (negatives) | **12** | specificity + validity gate |
| (E. SAL-missed — optional) | 0–10 | gold recall hole; omit to minimise |
| **total distinct** | **~94** | ~210 ratings (2× + ~20 tie-breaks) |

Sources: `data/gold/disagreement_pools.json` (extras, missed), the both-agree∩SAL gold (controls),
neither-judge∩not-SAL negatives (decoys), `data/gold/funnel_v2_candidates*.json` (pipeline positives).

## Workload (chosen: 2 raters, full overlap)
≈ 30 s per item (sentence + ±1 context, one-key Yes/No → type, autosave):
- **each rater: ~94 items ≈ 45–55 min**
- **tie-breaker: ~20 disputed items ≈ 10–15 min**
- **total ≈ 2 person-hours** across 3 people.

Scaling knob (recorded, not chosen): splitting items across **4 raters → ~25 min each**, or **6 →
~17 min each** — same validity (the pooled test is unaffected by who rates which item; rater/stratum
confounding is an accepted limitation). We chose **2+1** to minimise *recruiting*, accepting ~50
min/rater.

## Why these numbers (so we don't second-guess later)
- **A = 40:** the primary one-sided binomial test reaches 80% power at p=0.70 vs H₀=0.50 by n≈37; 40 gives margin. (E5 needed 60 only because its effect was smaller, 0.20→0.15 band.)
- **B = 30:** an estimate, not a test → ±~15% 95% CI, enough to show true precision ≫ gold-measured.
- **C = D = 12:** controls/decoys are shared validity overhead; 12 decoys with a ≤2 gate keep the
  accidental false-pass probability <5%.
- **2 raters + tie-breaker (not 1):** most items' correctness is contested; a single rater's errors
  bias the share. Two independent + a blind external tie-breaker ≈ majority-of-three (the E5 argument).
- **No practice round:** matches E5 design B; raters rely on the written guide.

## What's intentionally deferred
- arXiv/CS gold validation (no SAL reference there) → a later direct-labelling round.
- Full-paper (vs sentence+context) judgement.
- A pre-registered external registration (this is a frozen, time-stamped protocol, not a registration).

## To build (mirrors lr-ai E5 scripts)
1. `scripts/dataset/human_study_sample.py` — balanced draw of A–D(+E) + private key + sample SHA-256.
2. blind per-rater pages (extend `annotate.html`: rater ID + hash, hard-stop, decoy role hidden).
3. `scripts/dataset/human_study_analyze.py` — gate + one-sided test + Rogan–Gladen + Krippendorff α;
   rejects incomplete/foreign files.

_Decided 2026-10-27. Adapted from `lr-ai-paper/results/E5_chosen_design.md` (design B)._
