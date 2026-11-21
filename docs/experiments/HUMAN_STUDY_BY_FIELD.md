# Gap2Idea human study by field

Organiser guide for the field split of the frozen 135-item sample, prepared on
20 November 2026. The original [freeze record](human_study_freeze.md) gives the
sample provenance, previous deviations and supervision protocol.

## Raters and time

| Field | First-pass raters | Items for each person | Time at 30 seconds per item | Suggested booking |
| --- | ---: | ---: | ---: | ---: |
| Clinical medicine (SAL-40 RCT reports) | 2 | 69 | 34.5 minutes | 40–45 minutes |
| Computer science and machine learning (arXiv) | 2 | 66 | 33 minutes | 40–45 minutes |

Both raters in a field see every item for that field. Recruit independent
reviewers from outside the project team. A third person adjudicates only items
where the two raters differed on the gap decision or type. With one adjudicator
per field, recruit six people in total. Five people suffice if the same
independent adjudicator can cover both fields. Actual adjudication time depends
on the number of disagreements. Allow about 30 seconds per disputed item, plus
time for instructions. The original design estimated about 30 disputes overall,
or about 15 minutes of rating across the two fields.

## First-pass files

Open each page in Chrome or Edge. Give each rater only their assigned HTML
file from `data/gold/human_study/by_field/first_pass/`.

| Rater ID | File | Expected download |
| --- | --- | --- |
| MED_R1 | `rater_MED_R1.html` | `answers_MED_R1.json` |
| MED_R2 | `rater_MED_R2.html` | `answers_MED_R2.json` |
| CS_R1 | `rater_CS_R1.html` | `answers_CS_R1.json` |
| CS_R2 | `rater_CS_R2.html` | `answers_CS_R2.json` |

Before live rating, record the university's ethics decision, obtain consent
under the agreed process and record rater backgrounds and eligibility. The
original key appeared on a public branch. Record the decision to use the
original sample and confirm that participants have never seen that branch, the
key, existing labels or funnel outputs. Splitting by field does not address
prior exposure to the key.

Review the pages in a separate browser profile before the sessions. For each
live session, copy the assigned page alone to an empty folder on the study
device, use a fresh browser profile, and check that it starts at zero answers.
Read the page instructions first, then have the rater work independently.
When the page shows 69 or 66 answers, have the rater select **Save my answers**.
Collect the latest download from that session under the exact name above in
`data/gold/human_study/by_field/answers/`. Keep this directory private.
Follow the freeze record's browser and download cleanup before the next session.

## Adjudication and analysis

From the manuscript worktree root, after collecting the four JSON files:

```powershell
python scripts/dataset/human_study_by_field.py tiebreak data/gold/human_study/by_field/answers
```

For a field with disagreements, the command creates `rater_MED_TB.html` or
`rater_CS_TB.html` in `data/gold/human_study/by_field/tie_breakers/`. Each contains
only disputed items, with the earlier votes hidden. Run the appropriate
adjudicator session and collect `answers_MED_TB.json` or `answers_CS_TB.json` in
the same answers directory. A field with zero disputes needs no adjudicator file.
Then run:

```powershell
python scripts/dataset/human_study_by_field.py analyze data/gold/human_study/by_field/answers --out data/gold/human_study/by_field/results.md
```

## Protocol amendment to record before rating

The field split changes the rater assignment from one pair across both fields
to one pair per field. Each rater knows the broad field of their packet. The
pages contain the same frozen items, neighbouring text and instructions, with
separate shuffled orders and browser storage keys. Each field retains its six
decoys. The original weighting, two primary tests and pooled decoy gate
(at most two of 12) remain as specified. Report agreement by field, record the
actual adjudicator arrangement, and disclose that field and rater pair vary
together when interpreting differences between the fields. Record the date
and the team's approval of this assignment amendment before live rating.

## Frozen file checks

The original key is publicly accessible in this repository. The organiser must
assess and document the impact on blinding before any rating. Its SHA-256 agrees
with the freeze record. First-pass page hashes are listed below.

| File | SHA-256 |
| --- | --- |
| `rater_MED_R1.html` | `a0b90fe1d51d3312ebe7e71ba5a441bde3ec567841d188dc2f68137d8522d4e7` |
| `rater_MED_R2.html` | `143d326503e648a0cff636314f0b73a6b7a2aef496441e45e00af14de7cc6dfc` |
| `rater_CS_R1.html` | `b977b4b082f5a31506d8ff1eaccf6c34f681b1662b410b74b6b4082f550a2801` |
| `rater_CS_R2.html` | `aa0f42d9a1198e3cfb39876e5a2435acf9f13df4ef221dcda998d5f59489fe8e` |
