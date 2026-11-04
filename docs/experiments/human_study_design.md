# Gap2Idea human study — design (mirrors LR-AI E5 "design B")

A pre-specified, probability-sampled human study that validates the **extra gaps** our gap gold
flags beyond the **SAL human benchmark** (Lan et al. 2024). It answers whether those extras are
genuine research gaps (i.e. SAL under-labels and our method's extra labels are real), using two
blind raters per item plus an external blind tie-breaker, hidden positive controls and hidden
decoys with a validity gate, a one-sided hypothesis test fixed before labelling, and a
Rogan–Gladen robustness correction. Modelled on `lr-ai-paper/results/E5_chosen_design.md`.

**Status: design only — not yet implemented, no labelling started.** Freeze this file and the
sample key before any rater sees an item; record dated deviations.

> **Operational choice (2026-10-27): 2 raters + 1 external tie-breaker, ~94 pooled items, biomed-first.**
> The frozen decision record (minimized item counts, workload, rationale) is in
> [human_study_chosen_design.md](human_study_chosen_design.md). Read that first.

## The design in one table

| | |
|---|---|
| **Goal (GQM)** | Validate that our gold's *extra* limitations (flagged by our LLM-ensemble method, absent from SAL) are real gaps — so "precision vs SAL" understates true precision because SAL under-labels. |
| **Question** | What share of the extras are judged real research gaps by independent humans, from the sentence + its context? |
| **Hypotheses** | H₀: p ≤ 50% vs H₁: p > 50%, one-sided, α = 0.05 |
| **Planning target** | Detect a true share of **70%** with 80% power (effect size 0.20, as in E5). At 70% the claim is "most of what we add over SAL is real." |
| **Population** | The **227 extras** = sentences our final gold labels `limitation` on the 40 SAL papers that are NOT in SAL's gold (`data/gold/disagreement_pools.json → extras`). |
| **Sample** | **~60 extras**, a systematic balanced subset of the 227, stratified by gap type and paper; first subset (seed order) with all standardised covariate differences < 0.1. Freeze the key + SHA-256. |
| **Positive controls** | 15 high-confidence gold gaps (both judges agree **and** in SAL gold), hidden among items. Expected: **gap**. |
| **Decoys** | 15 negatives (neither judge flagged, not in SAL), hidden among items. Expected: **not a gap**. |
| **Validity gate** | Claim made only if ≤ **3 of 15** decoys are labelled "gap". Labelling everything "gap" therefore cannot produce the claim. |
| **Raters** | 2 independent raters per item, biomed-literate where possible (domain familiarity helps on dense RCT lines but is not required — the task is rhetorical, not clinical). Each sees items in their own random order. |
| **Tie-breaker** | One **external** person (not an author), blind to the two votes, rates every item where the two raters disagree → exact majority-of-three label. |
| **Leftovers** | Items still without two matching votes: the two raters + tie-breaker decide together (recorded in `consensus.csv`); no authors. Unresolved → counts as *not a gap*. |
| **Calibration** | None (no practice items). Raters rely on the guideline, the type definitions, and the context shown with each item. |
| **Decision rule** | Reject H₀ if the weighted share of confirmed gaps exceeds the power-simulated threshold **and** the validity gate passes. |
| **Report** | gate result; the test; estimate + one-sided 95% lower bound; Rogan–Gladen corrected share (robustness, from control sensitivity + decoy specificity); Krippendorff's α + % agreement. |

## Blind labelling task
Per item the rater sees: the **sentence** (marked in its ±2-sentence context), the paper's one-line
topic, and the three type definitions. They answer **gap: yes/no** and, if yes, **type**
(limitation / future_work / research_gap). No source, tier, SAL membership, or control/decoy role is
shown; the private key is never embedded in the page. (We reuse and extend `data/gold/annotate.html`.)

## Totals (indicative)
| | distinct items | ratings |
|---|---|---|
| extras (enter the test) | 60 | 120 |
| positive controls | 15 | 30 |
| decoys | 15 | 30 |
| tie-breaks (expected ~20–25%) | ~20 | ~20 |
| **all** | **90** | **~200** (≈ 2 raters × 90 + tie-breaks) |

Each rater ≈ 45 items (20–30 min); tie-breaker ≈ 20 items once.

## Analysis (pre-specified, before labels exist)
- **Primary:** weighted proportion of extras labelled a gap by majority-of-three, with its one-sided
  95% lower bound; reject H₀ if it clears the simulated threshold and the gate passes.
- **Gate:** decoys labelled "gap" ≤ 3/15.
- **Robustness:** Rogan–Gladen correction using positive-control sensitivity and decoy specificity —
  reported, never used to rescale the primary estimate.
- **Agreement:** Krippendorff's α (gap yes/no and type) + per-stratum % agreement, pre-adjudication.
- **Secondary (descriptive only, underpowered):** per-type and per-paper shares.

## Why this is valid (same guarantees as E5)
- Test, n, gate, and rater setup fixed **before** any label exists; file frozen + hashed.
- Probability sample (systematic subset of a systematic sample); the balance rule uses only
  covariates, never the gold label.
- Label quality = majority of three (two independent + blind external tie-breaker).
- Decoys + gate make "yes" and "no" count equally — leaning "gap" fails the gate.
- No author decides a contested item.

## Risks / limits to state
- Powered for 70%; near 60% power drops — if we don't reject, report "data do not show > 50%",
  never "H₀ accepted".
- One tie-breaker's leaning affects every tie; decoys that reach a tie are judged by them too and
  the gate counts them.
- Judgement uses sentence + local context, not the full paper.
- Control/decoy checks are small (±CI wide) → Rogan–Gladen is robustness, not the test.
- **Circularity note:** the gold was labelled by Opus+Sonnet; this study replaces that machine
  judgement with independent humans, which is the point — but recruit raters who are not the paper's
  authors.

## Optional second stratum — pipeline precision (pooled, same labelling round)
To also measure the **funnel's** true precision in one effort (TREC-style pooling), add a stratum of
**funnel-v2 predicted gaps that the gold calls `none`** (`data/gold/funnel_v2_candidates*.json`),
blind and shuffled with the rest. The same labels then yield pipeline precision alongside the
SAL-extras test. Keep it as a separate, clearly-labelled stratum so the primary test stays clean.

## What to build (mirrors the E5 scripts)
1. `scripts/dataset/human_study_sample.py` — draw the balanced 60-extras subset + 15 controls + 15
   decoys from the frozen pools; write the blind item set, the private key, and the sample SHA-256.
2. Blind per-rater pages (extend `annotate.html`): metadata sheet (rater ID + sample hash), autosave,
   XLSX export, hard-stop on incomplete, decoy role hidden.
3. `scripts/dataset/human_study_analyze.py` — validate rater/sample, build majority-of-three, run the
   gate + one-sided test + Rogan–Gladen + Krippendorff's α; stop on any incomplete/foreign file.
4. A frozen, time-stamped protocol with a dated deviations log.

_Design drafted 2026-10-26, adapted from LR-AI E5 design B._
