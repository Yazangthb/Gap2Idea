# Human study: frozen sample

Drawn 2026-11-05, before any rater saw an item. Design: [human_study_chosen_design.md](human_study_chosen_design.md).

**Sample hash (SHA-256 of `key_PRIVATE.json`):** `f8f55f1b22b3ffcebff0a402b2cb4533fdee8489a00f6ed807807d2d0093e286`
**Seed:** 20261105. **Funnel:** NLI + Qwen3.8-27B, high precision mode, test splits.

| Group | Population | Eligible after exclusions | Sampled |
|---|---|---|---|
| B1. arXiv, kept, gold none | 56 | 54 | 30 |
| B2. arXiv, kept, gold gap | 281 | 267 | 30 |
| S1. SAL-40, kept, SAL yes | 174 | 161 | 12 |
| S2. SAL-40, kept, SAL no, LLM gap | 77 | 69 | 40 |
| S3. SAL-40, kept, both none | 12 | 11 | 11 |
| D. Decoys (6 arXiv, 6 SAL), not kept in any mode | 8,517 | 8,517 | 12 |
| **Total** | | | **135** |

Eligibility excludes duplicates across groups, and any sentence whose first 60 characters appear in a public file that shows labels (`data/gold/human_validation_worksheet.csv`, `scripts/bench/bench_stagec_opt.py`). Sampled sentences may also appear in the public `data/gold/annotate.html`, which shows no labels. Tests fixed before rating: arXiv one-sided 95% lower bound above 70%; S2 share above 50%; at most 2 of 12 decoys labelled a gap.

## Deviations

- 2026-11-05: the first draw (hash `305b95dc…`, 136 items) matched exclusions only on exact text, and 4 sampled sentences appeared with labels in public files under slightly different text. It was replaced by this draw before any rater saw either. S3 now has 11 of its 12 sentences.
- 2026-11-16: the key, manifest and rater pages were pushed to the public branch `human-study-materials` (commit `528e319`). No rater had seen any item. The sample was kept, because a new draw would reuse most of the same sentences (S3 has no spare items, S2 has 29). Instead, raters rate in person on a device the team provides, with only their page open. The public commit also shows the sample existed before any rating.
- 2026-11-17, before any rater saw a page: the instructions on the rater pages were aligned with the LLM labelling guide (`scripts/dataset/gold_label_full.py`), and reviewed by an independent agent before rating. Changes:
  - Limitation: the full example list of the guide, "their own work (their study, method, model or data)", and the guide's cause-or-consequence rule.
  - Research gap: "not necessarily in their own study", as in the guide.
  - Not a gap: "recommendations for practice or policy" instead of "recommendations", so that research recommendations are not read as Not a gap.
  - New rule for highlighted text that holds two sentences or a heading (about 8 of the 135 items): judge the whole text, Yes if any part states a gap.
  - New rule for earlier work, spelling out the guide's "in the field or in prior work": a shortcoming or unsolved problem of existing methods or earlier studies is a Research gap, even if the paper then addresses it; a sentence that only reports what earlier work did, or a problem it already solved, is Not a gap. About 25 of the 60 arXiv items are of this kind, and without the rule people could read them either way.
  - The type order of the guide: Limitation, then Research gap, then Future work.
  - The number of items, a note that breaks are fine, and "tell the study organiser" instead of "send the file".
  - Page code: the number keys 1–3 now only set a type after Yes (before, pressing 1 on an unanswered item silently answered Yes); the saved file holds only the page's own items; the browser storage key includes the page version `2026-11-17`, so answers left by an older page never pre-fill this one.
  - The pages were rewritten from the unchanged key (`human_study_sample.py pages`): same items, same context, same order per rater. SHA-256 of the final pages: `rater_R1.html` `7be6d3b3e530a9fcb48d7edcf361ac96bc10cae87fcdc623e0a0ce0529587393`, `rater_R2.html` `6e412b6d09b26ce391df3266b6495b40269bf9a200779d769b499a163a37fefb`.
- 2026-11-17, before any rating: a supplementary section was added to `human_study_analyze.py`. The primary tests are unchanged. It adds exact 95% intervals per group; a conservative lower bound (exact bound per group, Bonferroni over groups) that stays valid when every answer in a group agrees, since that group then adds no uncertainty to the primary bound; the range of the estimates if the items excluded before sampling were all gaps or all not gaps; the smallest significant S2 count (26 of 40); the decoys labelled a gap; and the number of items whose type stayed tied.
- 2026-11-17, before any rating: two decoys can fairly be read as caveats: q084 (a comparison caveat; it is the sentence just before S3 item q130 and S2 item q028 in the same paper) and q023 (a protocol difference, followed by "could warrant further evaluation"). The gate is unchanged (at most 2 of 12); the report also gives the decoy count without these two.

**How labels are combined (as implemented since the freeze).** Each item has two votes, plus the tie-breaker's vote when the two raters differ on gap or on type. An item is a gap if most votes say so; with three votes this is never tied. Its type is the most common type among the Yes votes; if that is tied (for example 1–1 after the tie-breaker says No), the type is "unresolved": the item counts as a gap for the any-type reading and not as a limitation.

## Running the study

Files are in `data/gold/human_study/` (gitignored). Raters come in person and rate on a device the team provides.

1. **Raters.** Choose 2 raters and 1 tie-breaker from outside the project team. No authors, and nobody who has seen the gold labels, the SAL labels, the funnel output or the `human-study-materials` branch; ask each to confirm this. Record each person's field and background and their consent (the paper's Methods need both).
2. **Device, before each session.** Copy only that rater's page (`rater_R1.html` or `rater_R2.html`, check its SHA-256 above) into an otherwise empty folder. The device holds no key, no `human_study` folder and no checkout of the repository. Use a new Chrome or Edge profile for each rater (not Guest or InPrivate, which lose answers when closed). Open the page file in that profile and check that the header reads "0 answered". Close all other tabs and programs, and go offline if possible. The organiser stays in the room.
3. **Rating.** The rater reads the "How to rate" box first. The organiser answers questions about using the page only, never about an item. About 60–75 minutes; breaks are fine, the answers stay in that profile.
4. **Collecting.** When the header reads "135 answered", the rater selects "Save my answers" as the last action. Take the newest `answers_R1*.json` (or `answers_R2*.json`) from the Downloads folder, rename it to `answers_R1.json`, then delete it from the device, clear the download list and remove that browser profile before the next rater.
5. **Tie-breaker.** Make the tie-breaker page and run the session in the same way with `rater_TB.html` (expect "0 answered" and the number of disputed items):

   `python scripts/dataset/human_study_analyze.py tiebreak answers_R1.json answers_R2.json`

6. **Score:**

   `python scripts/dataset/human_study_analyze.py analyze answers_R1.json answers_R2.json answers_TB.json --out docs/experiments/human_study_results.md`

Don't open the key or any answers file before step 6, other than through these commands.
