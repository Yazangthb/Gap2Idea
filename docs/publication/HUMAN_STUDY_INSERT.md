# Adding the human-study results to the paper

The manuscript on branch `submission-make` is written so that the human study can be added in one pass.
Do the steps in order. Every place that must change is marked in the `.tex` files with a comment that starts
with `% HUMAN STUDY`.

## 1. Run the study

Follow `docs/experiments/human_study_freeze.md` ("Running the study"). In short:

1. Two raters who are not authors and have not seen the gold labels, the SAL labels or the funnel output
   each fill in one page (`data/gold/human_study/rater_R1.html`, `rater_R2.html`). Never send `key_PRIVATE.json`.
2. `python scripts/dataset/human_study_analyze.py tiebreak answers_R1.json answers_R2.json`
   makes the page for the tie-breaker.
3. `python scripts/dataset/human_study_analyze.py analyze answers_R1.json answers_R2.json answers_TB.json --out docs/experiments/human_study_results.md`
   writes the report.

The sample must be the frozen one: SHA-256 of `key_PRIVATE.json` starts with `f8f55f1b`.

## 2. Fill in the numbers

Open `manuscript_mdpi/sections/04b_human_study.tex` and copy the numbers from
`docs/experiments/human_study_results.md` into the macros at the top of the file:

| Macro | Where it is in the report |
|---|---|
| `\hsDecoyYes` | "Validity gate": decoys labelled a gap |
| `\hsDisputed` | first line: items sent to the tie-breaker |
| `\hsArxivPrec`, `\hsArxivLower` | "Primary tests", arXiv row |
| `\hsSTwoYes`, `\hsSTwoShare`, `\hsSTwoP` | "Primary tests", SAL row |
| `\hsSalLim`, `\hsSalAny` | "True precision", the two SAL-40 rows |
| `\hsBTwoShare`, `\hsBOneShare` | "Each group", rows B2 and B1 ("Any gap" divided by "Rated") |
| `\hsAgreeGap`, `\hsAlphaGap` | "Rater agreement", first line |

Two paragraphs in that file contain a pair of alternative sentences. Keep the one that matches the outcome
and delete the other. If the validity gate fails (more than 2 of 12 decoys called a gap), no claim can be
made: report the gate result and the raw shares only, and say so.

## 3. Change the manuscript

| File | Marker | What to do |
|---|---|---|
| `sections/04_results.tex` | Section 4.7.1 | Add `\input{sections/04b_human_study}` at the marker. Replace "The planned human study is designed to settle this question." by a pointer to the new subsection. |
| `sections/03_materials_methods.tex` | "Planned human study" | Rename the paragraph "Human study" and put it in the past tense. Add the sample groups (30 + 30 arXiv, 12 + 40 + 11 SAL-40, 12 decoys), the two tests fixed in advance (arXiv lower bound above 70%; more than half of the disputed SAL sentences are limitations) and the decoy check. |
| `main.tex` | abstract | Replace "not yet checked by people" by the main result. Keep the abstract at 200 words or fewer. |
| `sections/01_introduction.tex` | RQ3 and contribution 4 | RQ3 says "a human study ... is the next step": change to what the study covered. Add the result to contribution 4. |
| `sections/05_discussion.tex` | RQ3 paragraph; Section 5.3 | Replace "the planned human study will test" and the last sentence of Section 5.3 by the result and what it means for the 66.2% and 93.2% readings. |
| `sections/05_discussion.tex` | Limitations, "LLM labels" | Say what was checked by people (135 sentences, precision only) and what was not (recall, the rest of the corpus). |
| `sections/06_conclusions.tex` | third question; next steps | Add the result; change "human validation of the gold corpus" in the next steps to what is still open (recall, more sentences). |
| `main.tex` | back matter | Acknowledgments: thank the raters (with their consent). Institutional Review Board Statement: "Not applicable" is usual for volunteers who label published text, but check the university's rule. |
| `figures/overview.tex` | "(not yet checked by people)" | Change the note under "LLM labels". |

## 4. Check

- No `TODO` and no `% HUMAN STUDY` marker is left: `grep -rn "TODO\|HUMAN STUDY" manuscript_mdpi/main.tex manuscript_mdpi/sections`.
- The words "planned human study" no longer appear.
- The paper builds without warnings, and the abstract has 200 words or fewer.
- Report every deviation from the frozen design (for example a redrawn sample) in the Methods paragraph.
