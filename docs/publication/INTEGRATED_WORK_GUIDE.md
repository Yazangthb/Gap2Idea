# Gap2Idea publication work: start here

This branch brings together the submission draft, Hamza's baseline evidence, Yazan's scoring and provenance work, and Osama's field-specific human-study preparation. Historical plans and run records remain in their original locations and are not overwritten. The preprint is [on Zenodo](https://doi.org/10.5281/zenodo.24047677).

## Manuscript and revision

- Journal manuscript: [`manuscript_mdpi/main.tex`](../../manuscript_mdpi/main.tex) and its `sections/`, `tables/`, and `figures/` folders. The committed PDF is the existing draft; it has not been updated with the completed H100 results or a human study.
- Team's [revision plan](REVISION_PLAN_2026-11-16.md): read as a plan, not as a list of completed tasks. The author subsequently assigned Qwen full-paper and 20-sentence baseline experiments to Hamza and paired scoring to Yazan.
- Related-work [candidates](RELATED_WORK_CANDIDATES_2026-11-07.md) and the [saved search run](../../lr-runs/gap-funnel-related-2026-11-07/).

## Experiments and scoring

- **H100:** [final evidence and report](../experiments/baseline_revision_h100_2026-11-18/README.md), including all 60 configurations and the verified final snapshot. Four calls failed after retries; no sentences were unattempted. The report calls itself "V100 baseline evidence" because of its original generator; its manifests and directory identify H100/vLLM. These are baseline scores, not identity-matched paired funnel comparisons.
- **V100:** [evidence and historical status](../experiments/baseline_revision_v100_2026-11-18/README.md); its final status was not verified during integration. Do not mix H100 and V100 results without stating the model and runtime differences.
- **Paired scoring:** [instructions and limitations](../experiments/PAIRED_SCORING_2026-11-18.md). Nine historical 27B comparisons were reproduced; new paired comparisons require complete funnel predictions with verified matching model identity. The scorer and preparation changes are on this branch, not in the frozen campaign checkout.
- **Third-party SAL/FWS data:** follow [`data/splits/README.md`](../../data/splits/README.md) to fetch pinned sources and rebuild splits. Check licences and the manuscript's data-availability statement before redistribution.
- **Triple-check:** [provenance audit](../experiments/TRIPLECHECK_PROVENANCE_2026-11-18.md); the per-item requested model is inferred from a captured run and is not per-request response metadata.

## Human study: deferred

The owner has deferred the study; no ethics approval or ratings had been reported when that decision was made. The [field-specific guide](../experiments/HUMAN_STUDY_BY_FIELD.md), [original freeze](../experiments/human_study_freeze.md), [preparation notes](../experiments/HUMAN_STUDY_PREPARATION_2026-11-18.md), and rater pages are retained for later review. The original answer key is public; splitting the pages by field does not restore secrecy. Resolve ethics, consent, and the sample/blinding decision before recruitment. Combining files here is not approval to begin the study.

## Remaining handover

1. Confirm final V100 status with its owner; recover its final evidence separately if needed.
2. Produce and verify matched paired comparisons and integrate the accepted results into manuscript tables and text, including failures and comparison limitations.
3. Complete the text-licence and data-availability audit, and rebuild the publication tables from a fresh checkout on a second machine.
4. Resolve authorship, funding, disclosures, cover letter, and final author approval. If the team proceeds with the study, finish its ethics and design decisions before recruiting.
