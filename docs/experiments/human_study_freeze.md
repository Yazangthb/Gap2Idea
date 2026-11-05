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

## Running the study

Files are in `data/gold/human_study/` (gitignored; share only what's listed).

1. Choose 2 raters and 1 outside tie-breaker who haven't seen gold labels, SAL labels or funnel outputs. Ask them not to browse the project repository.
2. Send `rater_R1.html` and `rater_R2.html`, one per rater. **Never send `key_PRIVATE.json`.** Each page opens in a browser and works offline.
3. Collect `answers_R1.json` and `answers_R2.json`.
4. Make the tie-breaker page and send the resulting `rater_TB.html`:

   `python scripts/dataset/human_study_analyze.py tiebreak answers_R1.json answers_R2.json`

5. Score:

   `python scripts/dataset/human_study_analyze.py analyze answers_R1.json answers_R2.json answers_TB.json --out docs/experiments/human_study_results.md`

Don't open the key or any answers file before step 5, other than through these commands.
