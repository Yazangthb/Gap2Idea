# Gap2Idea human study: chosen design

**Revised 2026-11-05 for the final funnel. Status: built and sample frozen ([human_study_freeze.md](human_study_freeze.md)), no labelling started.** Freeze the sample key, its SHA-256 and the hypotheses before any rater sees an item, and log dated deviations. Rationale and lineage: [human_study_design.md](human_study_design.md). The original 2026-10-27 strata are at the end of this file.

## What changed and why

| Before | Now | Why |
|---|---|---|
| Biomedical first, arXiv in a later round | One pooled round covering SAL-40 and arXiv | 428 of 455 gold false positives were arXiv, and arXiv has no human labels |
| Main question: are our SAL extras real? | Main question: **how many extracted gaps are true?** | Precision is now the project's priority |
| Pipeline positives from `funnel_v2` candidates | Output of the final funnel, high precision mode | That is the system the paper reports |
| About 94 items | **136 items** | arXiv needs its own estimate |

## What it answers

1. **True precision of the final funnel** on arXiv and SAL-40, measured against humans.
2. **Does SAL under-label?** Are sentences SAL rejects but our funnel and LLM labels accept real limitations?
3. **Are the LLM gold labels trustworthy?** Do humans agree with them on kept sentences, including those it calls false positives?

It doesn't measure recall, which needs people to read whole papers. Recall stays reported against SAL and LLM labels.

## Design

| | |
|---|---|
| **Primary test 1, arXiv** | Weighted true precision of high precision mode, one-sided 95% lower bound. Claim "most extracted gaps are true" only if the bound is above **70%** |
| **Primary test 2, SAL** | Share of stratum S2 that are real limitations. H₀ p ≤ 50%, H₁ p > 50%, exact binomial, one-sided, α = 0.05 |
| **Raters** | 2 independent raters per item + 1 external blind tie-breaker, majority of three |
| **Who can rate** | Nobody who has seen gold labels, SAL labels or funnel outputs, including whoever works on tasks 2, 5, 6 or 7. No author adjudicates |
| **Blindness** | No source, stratum, label or dataset shown. Items shuffled together. The private key is never embedded in the page |
| **Question asked** | Is this sentence a gap stated by the authors? If yes: limitation, future work or research gap |
| **Validity gate** | Report claims only if at most 2 of 12 decoys are labelled a gap |
| **Disagreements** | The tie-breaker rates each one, blind to both votes. Unresolved items count as not a gap |
| **Analysis** | Weighted precision with finite-population variance per track; Krippendorff's α and % agreement; gate result; each stratum's share reported separately |

## Strata

All items come from the test splits. "Kept" means high precision mode of NLI + Qwen kept the sentence.

| Stratum | Population | n | Role |
|---|---|---|---|
| **B1. arXiv, kept, gold `none`** | 56 | **30** | Are the funnel's false positives really false? |
| **B2. arXiv, kept, gold gap** | 281 | **30** | Are its true positives really true? Also checks LLM gold |
| **S1. SAL-40, kept, SAL yes** | 174 | **12** | Positive control and rater check |
| **S2. SAL-40, kept, SAL no, LLM gap** | 77 | **40** | Primary test 2: does SAL under-label? |
| **S3. SAL-40, kept, SAL no, LLM none** | 12 | **12** (all) | Sentences both label sets reject |
| **D. Decoys, not kept, all labels none** | Large | **12** (6 arXiv, 6 SAL) | Validity gate |
| **Total** | | **136** (135 after exclusions, see freeze file) | About 300 ratings |

Counts were checked on 2026-11-05 against the saved verdicts. Exact sampling draws at random within each stratum with a recorded seed.

**Estimates**
- arXiv precision = (281 × B2 share + 56 × B1 share) / 337.
- SAL-40 precision = (174 × S1 share + 77 × S2 share + 12 × S3 share) / 263. Count only the limitation type as correct, since that is what this track extracts; report any gap type as a second reading.

**Why these sizes**
- B1 and B2: if B2 is about 90% and B1 about 50%, arXiv precision is about 83% with a standard error near 4.5 points. The chance of clearing 70% is about 90% at a true 83%, and about 70% at a true 80%.
- S2: 40 items keep the original test's 80% power at a true share of 70%.
- D: 12 decoys with a limit of 2 keep the chance of an accidental pass below 5%, as in the original design.

## Workload

At about 30 s per item:
- Each rater: 136 items, **65–75 minutes**.
- Tie-breaker: about 30 items, **15 minutes**.
- Total: **about 2.5 person-hours** across 3 people.

Splitting items across 4 raters brings this to about 35 minutes each with the same validity.

## Before sampling

- Exclude sentences that appear in the public `data/gold/human_validation_worksheet.csv`, which shows labels. At least 23 kept sentences are affected. Report how many each stratum loses; S3 may fall below 12.
- Freeze the funnel. Any later change to its setup means redrawing the sample.
- Check for the same sentence in two strata, and keep one copy.

## Limits to state in the paper

- Simple random sampling ignores clustering within papers, and 5 long papers hold most arXiv false positives. Report how many B1 items come from them.
- 30 s per sentence with ±1 context is not a full-paper reading.
- Biomedical items are clinical trial reports only.
- The 70% bar was set on 2026-11-05, after the LLM-based precision estimate of 83.4% was known but before any human label.

## Built

1. `scripts/dataset/human_study_sample.py`: draws the groups and writes the private key, manifest and blind rater pages. It refuses to redraw once a key exists.
2. `data/gold/human_study/rater_R1.html` and `rater_R2.html`: self-contained, with autosave and a hard stop until every item is answered. Each rater sees a different order.
3. `scripts/dataset/human_study_analyze.py`: tie-breaker page, gate, both primary tests, weighted precision and Krippendorff's α. It rejects incomplete, foreign or repeated files.

## Original strata (2026-10-27, superseded)

A. SAL extras, 40. B. Pipeline positives with gold `none`, 30. C. Positive controls, 12. D. Decoys, 12. E. SAL-missed, optional 0–10. About 94 items in total, biomedical papers only, with arXiv deferred.

_Adapted from `lr-ai-paper/results/E5_chosen_design.md` (design B)._
