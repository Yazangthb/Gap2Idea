# MAKE submission: status of branch `submission-make`

Written 2026-11-11. Branch `submission-make` was cut from `paper-writing-v2` at `eee7008` (tag `preprint-v1`).
It is local only: nothing has been pushed.

## Done on this branch

| Item | Where |
|---|---|
| Published-method baselines added (Task 5b), scored on identical test sentences with paired tests | `docs/experiments/task5b_baselines/`, `scripts/bench/import_task5b_baselines.py`, `scripts/bench/paper_baselines_table.py`, Table 7, Methods 3.5, Results 4.4 |
| Figure 1 redrawn as a vector diagram (no AI-generated image, no resolution problem) | `manuscript_mdpi/figures/overview.tex` |
| Table 8 (funnel versus whole paper): intervals on their own line at 8 pt; Tables 4 and 5 widened so intervals no longer wrap | `tables/whole_table.tex`, `scripts/bench/paper_whole_tables.py`, `sections/04_results.tex` |
| Abstract: 197 words; names the variant behind the headline numbers, the 27B arXiv result and the unchecked labels | `main.tex`, `preprint.tex` |
| Text revised for clarity and flow in every section; numbers unchanged | `sections/01`–`06` |
| Human-study results can be added in one pass | `sections/04b_human_study.tex`, `docs/publication/HUMAN_STUDY_INSERT.md` |
| Cover letter drafted with the two required statements | `docs/publication/COVER_LETTER_MAKE.md` |
| Script that builds the PDF, the source zip and the 600 dpi figures zip | `scripts/publication/make_submission_package.py` |

Checks that were open and are now settled:

- FWS dev really has 418 papers and 418 positive sentences (counted from the rebuilt split).
- "Batch 32" for the SAL-trained encoders is correct (`scripts/training/h100_allmodels.py`).
- "Without research gaps" means that sentences labelled as research gaps are removed before scoring
  (`scripts/bench/stage_c_modes_report.py`); the paper now says so.
- Kilicoglu et al.: rules had the best accuracy (91.5% against 89.6% supervised); the text now says "slightly more accurate".

## Content changes to confirm (they change a claim, not only the wording)

1. **Introduction and abstract.** "Criticism of prior work" is no longer given as an example of a non-gap,
   because the research-gap definition includes problems "in the field or in prior work". The examples are
   now "a reported result or a piece of background that mentions a difficulty".
2. **Discussion 5.3.** The sentence "recall on long papers is understated for every system" was replaced.
   If long papers are under-labelled, precision is understated; the effect on recall is unclear.
3. **Discussion 5.2.** A direct answer to "When is whole-paper prompting enough?" was added.
4. **New finding from the baselines.** On FWS, TF-IDF with logistic regression (F1 0.748) is not
   significantly behind the funnel (0.760). The paper states this in Results 4.4, Discussion and Conclusions.
5. **Results 4.3.** The result on the 7 untouched gold test papers now has its own sentences
   (high precision keeps its precision, 93.9%, but recall is 41.4%).

## Left to do

### Needs the team
- [ ] **Author list**: is the supervisor added? Who is the corresponding author? (`main.tex` L53–62)
- [ ] **Author Contributions** in CRediT form (`main.tex`, still a placeholder).
- [ ] **Funding and Rubetek**: the Funding statement says "no external funding" (marked TODO) while the
      Acknowledgments thank Rubetek "for supporting this work". Decide which is right.
- [ ] **Figure 1**: approve the redrawn diagram. To go back to the old image instead, restore the
      `\includegraphics` line in `sections/01_introduction.tex`, supply a file at 600 dpi and add the image tool
      to the GenAI sentence in the Acknowledgments.
- [ ] **Human study**: find `data/gold/human_study/` (probably on Yazan's machine), recruit two raters and a
      tie-breaker who are not authors, run it, then follow `HUMAN_STUDY_INSERT.md`.
- [ ] All authors read the final PDF and approve it.
- [ ] Suggested reviewers (3–5, entered in the submission system), APC check.

### Small technical items
- [ ] `references.bib`: `sileo2024tasksource` has no DOI or URL.
- [ ] Methods 3.5: the 9B model has no repository id (the 4B has `Qwen/Qwen3.5-4B`).
- [ ] Methods 3.2: the paper does not say why sentences from the earlier classifier's training file were
      removed from FWS. Add the reason if there is one.
- [ ] Data Availability: after the final commit, tag it (for example `submission-v1`) and replace
      `preprint-v1` in `main.tex`.
- [ ] `preprint.tex` shares the sections, so it now builds the revised text. `preprint.pdf` in the
      repository is still v1. If a second preprint version is posted, change the version label and rebuild.
- [ ] Task 5b server jobs (FWS SciBERT, cached SAL encoders) are not needed for the paper. If they finish,
      they can be added with `import_task5b_baselines.py`.
- [ ] On another machine, `paper_whole_tables.py` should be run once to confirm that it regenerates
      `whole_table.tex` exactly (it needs `docs/experiments/nli_variants.json`, which is not in the repository).

### At submission
- [ ] `python scripts/publication/make_submission_package.py` → `manuscript.pdf`, `source.zip`, `figures.zip`.
- [ ] Fill in and attach the cover letter.
- [ ] `grep -rn "TODO\|HUMAN STUDY" manuscript_mdpi/main.tex manuscript_mdpi/sections` returns nothing.
