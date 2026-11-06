# Related-work review - 2026-11-06

Scope: Section 2, its comparison table, and two added bibliography entries in the
`Gap2Idea-paper-review` worktree. Existing manuscript formatting edits were
preserved. No Task 5b source, results, processes or Git state were changed.

## Editorial decisions

- Organise the argument as discourse classification -> targeted detection ->
  LLM extraction/inference/critique -> methodological position.
- Explain why each group is relevant, instead of listing studies and scores.
- Preserve every existing comparison-table study; add Cheng et al. and
  Thaepprasit et al. Keep target ordering and capitalisation consistent.
- Distinguish author-stated limitations, criticism of previous work, inferred
  gaps and generated critiques. LimGen and LimitGen are separate resources.
- Describe zero-shot NLI as no gap-specific classifier fitting, while explicitly
  acknowledging prior NLI training and labelled threshold selection.
- Avoid a first-system claim, an untested claim of superior domain transfer,
  or a claim that extracted gaps establish idea quality or continuing novelty.
- Keep human-labelled SAL/FWS evidence distinct from the provisional joint
  three-type evaluation. No pending server results were inserted.

## Claim checks and sources

| Item | Decision and primary source |
| --- | --- |
| CoreSC | Corrected the old description of gap/failure/future-work categories: CoreSC represents investigation components. [Liakata et al.](https://aclanthology.org/L10-1440/) |
| Dr. Inventor | Future Work belongs to argumentative structure; Limitation is a crosswise feature. The earlier separate-layers wording was underspecified, not grounds for removing either target. [Fisas et al., introduction and annotation schema](https://aclanthology.org/L16-1492.pdf) |
| Abstract resources | Retained PubMed RCT/CSAbstruct; described MAZEA as a multi-label framework and Emerald's combined Research limitations/implications heading explicitly. [MAZEA](https://aclanthology.org/L12-1428/), [Emerald](https://aclanthology.org/U19-1016.pdf) |
| Sci-Sentence | Corrected the claim that its inputs are only review sections. Section 3.2 samples introductions, related work and limitations. Its Limitation definition concerns a previous study. [Schema and benchmark](https://arxiv.org/html/2508.04337v1) |
| Contextual classification | Retained the contextual/sequential relationship without claiming these systems solve our exact task. [Lan et al.](https://aclanthology.org/2024.findings-emnlp.944.pdf), [Brack et al.](https://doi.org/10.1007/s00799-023-00392-z), [Zhang and Zhang](https://doi.org/10.1007/s11192-024-05048-6) |
| Clinical limitations | Section/zone rules precede Stage A. Lan's 0.821 is a published detector score, not a matched full-paper baseline score. [Kilicoglu et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6016608/), [Lan et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC11807350/) |
| Zhu et al. | Publisher abstract supports BERT-based recognition and clustering; retained both. [Publisher](https://doi.org/10.1002/pra2.206) |
| FWS recognition | Attributed 0.9073 macro-F1 specifically to Bernoulli naive Bayes; distinguished binary recognition from multiclass direction typing and positive-class F1. Year 2022 denotes the cited preprint. [Zhang et al.](https://arxiv.org/abs/2212.13860) |
| Downstream analysis | Suray examines subsequent engagement with future-work statements. Added Thaepprasit as a distinct analysis of reported limitations, not an extraction benchmark. [Suray](https://arxiv.org/abs/2405.20785), [Thaepprasit](https://arxiv.org/abs/2609.15191) |
| BAGELS/FutureGen | Both extraction and generation are relevant precedents; removed the unsupported relative label closest extraction precedent. [BAGELS](https://aclanthology.org/2025.findings-emnlp.1050/), [FutureGen](https://arxiv.org/html/2503.16561v3) |
| Cheng et al. | Added a direct LLM future-work extraction precedent with section selection. Publisher supplies English metadata/abstract; the full-text technical-route description specifies Conclusion/Future Work section selection. No extraction-performance comparison was inferred from its cosine-similarity scores. [Publisher](https://kxgc.magtechjournal.com/CN/10.15978/j.cnki.1673-5668.20250721) |
| GAPMAP | Explicit/implicit distinction and paragraph/full-paper settings supported. [Authors' preprint](https://arxiv.org/abs/2510.25055) |
| Generated critique | LimGen uses author limitations for suggestive generation; LimitGen evaluates feedback/limitation generation; Tree-of-Concerns specifically targets unstated limitations. Avoided claiming that all LimitGen examples are unstated. [LimGen](https://arxiv.org/abs/2403.15529), [LimitGen](https://aclanthology.org/2025.acl-long.1009/), [Tree-of-Concerns](https://arxiv.org/abs/2608.20777) |
| NLI lineage | Added the missing methodological connection to entailment-based zero-shot classification using an existing bibliography entry. [Yin et al.](https://aclanthology.org/D19-1404/) |

## Review limits and handoff

This is a targeted related-work review, not a systematic review or proof of
exhaustive coverage. No additional experiment is required to support the
revised literature descriptions. Experimental claims elsewhere in the paper
still need their separate final review after Task 5b finishes. Some entries
intentionally cite author preprints; final venue versions and bibliography
format should be refreshed before submission. Access restrictions on some
publisher full texts were handled through public author manuscripts, publisher
abstracts and the cited primary sources; no unverified fine-grained result was
added.

## Validation

- All 27 citation keys in Section 2 resolve in the bibliography.
- Table 1 retains 17 previous studies and adds 2 studies (19 total), plus
  the current-study row.
- Full LaTeX build and rendered-page review performed after the revision;
  no undefined citations/references, overfull boxes or oversized floats.
- The MDPI template and intentional asymmetric text layout are preserved.
