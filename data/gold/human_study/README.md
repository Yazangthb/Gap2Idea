# Human-study materials

## Field-specific study packet for team review

The current packet has four first-pass pages in [by_field/first_pass/](by_field/first_pass/).
Two clinical raters each review 69 items and two CS/ML raters each review 66.
The [organiser guide](../../../docs/experiments/HUMAN_STUDY_BY_FIELD.md) covers
assignments, timing, eligibility, answer collection, adjudication and scoring.
Give raters only their assigned HTML page. The repository contains the answer key.

## Original mixed-field materials

This branch contains the frozen sample, rater pages, manifest, and key for the study described in docs/experiments/human_study_freeze.md. The key and labels are publicly accessible on this branch. Raters should receive only their assigned blind page and should not inspect the repository. The original design assumed the key was not public; document this change and consider a fresh sample if blinding may have been compromised.

The sample generation also needs docs/experiments/nli_variants.json and nli_sal_fws.json, provided on this branch. The frozen key has SHA-256 f8f55f1b22b3ffcebff0a402b2cb4533fdee8489a00f6ed807807d2d0093e286. Do not overwrite the frozen sample when preparing a new study.
