# Gap2Idea: evidence requests and shared completion queue

> Superseded status note (4 November 2026): use `EXPERIMENT_COVERAGE.md` and `FINAL_REVIEW.md` for the current branch audit. Frozen-head regeneration and centralized experiment prompts now exist; fine-tuned checkpoints, the historical RCT helper, completed human annotations and raw expanded-corpus labels still require separate evidence. The tasks below preserve the earlier handoff.

Updated 13 October 2026. Current authority: `../Gap2Idea`, branch `minimal`, commit `66c237a985ebf3351d684b1096ad4d91bf051786`.

This is a claimable queue for the three-person team, rather than fixed assignments. Pick one card, record your name and status, and return the listed artifacts. Time estimates below are estimates of packaging/review effort **if the files already exist**, not estimates of retraining or annotation. The 64-paper human study and 800-paper venue study are completed according to the team; their results are unavailable to this checkout. Recover them before proposing repetition.

## Message ready to send to the team

> We have expanded the paper using the clean minimal branch, restored the related-work comparison table, and written the collection ? preparation ? localization ? detection ? verification ? RQ analysis methodology. Please recover the artifacts for the studies you worked on, especially the completed 64-paper human evaluation and 800-paper venue corpus. We need article IDs, annotation guidelines/raw labels, predictions, exact run configurations, and results, rather than only a headline score. The saved SAL predictions and 391-candidate verifier decisions are already available; please supply their missing training/data/prompt provenance rather than regenerating them unnecessarily. Claim any card below that fits your time. Return its folder path, run ID/commit, a README describing how outputs were produced, and any unavailable items. Do not send API keys or credentials.

## What is already available

- **Principal SAL analysis:** 40 papers, 7,968 sentences, 205 gold positives, eight binary detector vectors in `stage_c_package/sal_criteria/sal_test_allmodels.json`. Both prompt verdict vectors for all 391 PubMedBERT A+B candidates survive in `faithful_pubmed_verdicts.json`. Strict stage scores and a descriptive paper bootstrap have been recomputed offline.
- **Efficiency estimates:** 40-paper `cost_tokens.csv`, with 282,846 estimated full-article versus 75,622 selective-input tokens (3.7403 ratio). This is an accounting profile, not exact billing.
- **Auxiliary results:** FWS and Sci-Sentence aggregate metrics in `breadth_results.json` and `baseline_grid_results.json`; a selected ten-paper full-article baseline and stage summaries in `cost_fullpaper_results.json` and `cost_perstage_results.json`.
- **Development records:** Stage-B notebooks/markdown summaries and Stage-C sweeps survive. Their existence does not supply every checkpoint, split, response, or row-level prediction.
- The paper folder's historical `code/` and `data/` contain older artifacts, including a classifier and a 19-item overlap reference. These are not established matches for the final heads or the 49-item internal reference. The clean repo has no root `data/` directory in this checkout.

## P0: recover existing work before deciding on new experiments

Each card is independently claimable except where a dependency is stated. Use status `unclaimed / claimed / delivered / verified` and add an owner when claimed.

### E01 ? Completed 64-paper human evaluation

Status: unclaimed. Estimated packaging: 2?4 hours. Depends on: none. Paper use: independent extraction validation; potentially both target types.

Return:
- Exact 64-paper manifest with IDs, domains/venues, years, access/source format, inclusion/exclusion criteria, and sampling procedure/seed.
- Annotation guideline/version, target definitions, annotation unit, whether all sentences or only candidates were reviewed, and how own-study versus prior-work statements were handled.
- Each annotator's original labels, annotator counts/background, adjudication rules and final reference. Return existing agreement calculations and the inputs needed to recompute them; do not invent an agreement statistic if only adjudicated labels survive.
- System predictions with paper/sentence IDs and types; exact head, threshold, prompt, context, batching and model version; run date/commit.
- Train/dev/test overlap check. State whether prompts or thresholds were changed after inspecting this set and whether evaluation covered the raw-PDF front end.
- Existing per-class and pooled results, stage counts, errors and any baseline results.

Acceptance: another team member can connect each reported count to an article and label, recompute metrics, and decide whether this set is an independent frozen evaluation. If it contains only candidate annotations, say so: it cannot establish full-article recall without an exhaustive reference.

### E02 ? Completed 800-paper venue corpus

Status: unclaimed. Estimated packaging: 2?4 hours. Depends on: none for recovery; E03 for interpreting extraction counts. Paper use: deferred corpus characterization.

Return:
- Exact article manifest and counts by venue/year/domain; collection queries, dates, eligibility, deduplication, inaccessible/excluded articles and final analyzed denominator.
- Extraction output for every article, including zero-output articles; sentence text, source IDs/sections, output type and detector/verifier provenance.
- Run configuration and parsing failures. Existing normalization, duplicate merging, clustering/taxonomy, counts, figures and analysis code if used.
- Define the denominator of each frequency: statements, papers, papers with any extraction, or venues. State whether a paper may contribute to several categories.
- Any human validation/audit sample and its selection rule. Explain how sampling imbalance and extraction errors affect comparisons.

Acceptance: all totals reconcile with the manifest and proposed tables/plots. Recover the **actual** protocol; do not fill in an assumed AI-versus-DS comparison merely because an old draft proposed it.

### E03 ? Resolve the final detector and training recipe

Status: unclaimed. Estimated packaging: 1?3 hours. Depends on: none. Paper use: final method definition and reproducibility.

Three configurations currently coexist: (a) runner bge-small plus self-distilled/ACL training; (b) Stage-B notebook MiniLM plus mixed public training; (c) cached SAL binary detectors, including fine-tuned PubMedBERT. Please identify which was actually used for each reported study and which is intended for release.

Return:
- A run-to-head matrix for SAL, FWS, Sci-Sentence, internal articles, the 64-paper set and the 800-paper corpus.
- Checkpoint/head/tokenizer files or stable download references and hashes for all eight SAL detector vectors, with training commands/configs, encoder identifiers, versions, seeds, optimization details, class mapping, class weights, thresholds and threshold-selection split.
- Training/dev/test paper manifests, dataset versions, label transformations, negative sampling and duplicate/overlap controls. Recover the code that generated `sal_test_allmodels.json`.
- Explain historical head metadata (8,395 training rows) versus older notes about 25,427 self-distilled rows and 7,706 mixed-public rows. These may refer to different sampling/training stages; supply the actual lineage rather than choosing one count.

Acceptance: each prediction vector has a documented producing configuration. A fallback-to-rules execution must be distinguishable from a trained-head execution.

### E04 ? SAL source reconstruction and section metadata

Status: unclaimed. Estimated packaging: 1?2 hours. Depends on: none. Paper use: collection/preparation and localization validity.

Return original corpus version/license/source, source annotations and splits, PMC BioC cache, article/row alignment code, section-mapping rules and the final 40-paper manifest. Explain how the 7,968-record stream was obtained and which source rows/papers were excluded, merged or resegmented. Include any existing heading/section audit. The current prediction file already contains the evaluated rows; we need their provenance and regeneration path.

Acceptance: reconstruction preserves or explicitly maps sentence IDs/order, gold labels and section tags; raw-PDF localization claims remain separate from this BioC-conditioned test.

### E05 ? Exact Stage-C run provenance

Status: unclaimed. Estimated packaging: 1?2 hours. Depends on: none. Paper use: verification method and prompt reproducibility.

Return the exact resolved historical `SYSTEM_RCT_LOOSE` instruction and examples used for the faithful run, its missing helper `scripts/bench/bench_stagec_opt.py`, actual provider model/version/date behind `latest`, request parameters, candidate order, batches/context, original responses and failure/retry logs if saved. The current RCT prompt is not byte-identical to the historical helper found in cached history; do not substitute it silently. The inline SAL-criteria instruction and complete cached verdicts are already available.

For reported batch/context/model/protection sweeps, identify selection/dev data, keep each run's candidate pool and decisions, and distinguish the older multi-detector union run from the canonical 391-candidate run. The union-run F1 of 0.701 is not the canonical full-funnel result (0.658).

Acceptance: prompt hashes/configurations and verdict IDs resolve to the actual run, with unavailable response/version information explicitly stated.

### E06 ? FWS and Sci-Sentence row-level results

Status: unclaimed. Estimated packaging: 1?2 hours. Depends on: E03 for complete provenance. Paper use: cross-target results and error analysis.

Return dataset version/supplied splits, evaluated row IDs/text/gold labels, before/after predictions and verifier decisions for FWS (1,506 rows / 220 positives) and Sci-Sentence (140 / 40). Preserve the exact label map, author-gap versus target-matched prompt policy, train-overlap exclusions and paper IDs. Exact sentence exclusion in FWS alone does not establish paper-disjointness.

Also recover row decisions/configurations supporting Sci-Sentence's target-matched F1 around 0.730 and in-domain F1 around 0.880 from notes. State whether the 140-row test set influenced fitting, threshold/prompt selection, or the reported ranking. Until these are supplied, the paper treats them as exploratory logged results rather than independently verified primary results.

Acceptance: reproduce published aggregate counts with an explicit target definition and no silent pooling across datasets.

### E07 ? Ten-paper full-article baseline outputs

Status: unclaimed. Estimated packaging: 1?2 hours. Depends on: none. Paper use: same-model pilot, baseline limitations.

Return the selected ten paper IDs (2,053 sentences / 55 positives), seed-1 sampling script/filter, exact full-article prompt, numbered input text, raw replies, parsed sentence indices and per-paper metrics. Include both empty/unparseable reply cases and any request limits/truncation/errors. Record actual retries, input/output tokens and timestamps if available.

Acceptance: baseline counts and failures can be recomputed, including all sampled papers. Its rounded P=0.559/R=0.345 imply F1 approximately 0.427; explain any different headline using unrounded data and the actual averaging rule.

### E08 ? Internal reference and PDF-boundary audit

Status: unclaimed. Estimated packaging: 1?2 hours. Depends on: none. Paper use: limited internal transfer evidence.

Return the ten-paper 49-statement reference with types (11 limitation, 14 future work; other types separate), PDF/source text, extraction IDs, label-generation prompt/model, any manual corrections, overlap matching code and per-head decisions. Distinguish this reference from older 19/33-item overlap subsets and any 37-case review mentioned in notes; explain which benchmark each count belongs to.

Acceptance: target-specific recall and candidate load are checkable; do not report precision or human agreement from positive-only LLM-built labels. If a fully human-annotated replacement already exists, return it instead of commissioning duplicate annotation.

### E09 ? Resource measurements and portable reproduction

Status: unclaimed. Estimated packaging: 2?4 hours. Depends on: E03/E05 for a complete runnable recipe. Paper use: efficiency and artifact release.

Return existing provider usage/billing logs, actual input/output/retry tokens, CPU/GPU environment, parsing/detection/verifier timings and measurement scope. Separate estimates from measurements and one-time training from per-paper inference. The 40-paper accounting and ten-paper actual pilot are different samples/protocols; do not combine them into one billed saving ratio.

Make one documented offline command reproduce SAL tables from the surviving artifacts. Replace machine-specific paths in the selected regeneration scripts, document required data/checkpoint locations and preserve fixed run/config manifests. API regeneration should be a separate explicit command. The current paper's offline verification script already regenerates the principal scores; extend rather than duplicate it.

Acceptance: a fresh environment can reproduce saved-artifact results; missing components and optional API costs are documented.

## P1: experiments to consider only after recovery

These are **conditional gaps**, not claims that the team has not done them. Check E01?E09 first. Agree on the final frozen configuration before any inference. Each card can be split into preparation, execution and analysis so people with less time can contribute.

| Card | Trigger | Minimum executable study | Required outputs / completion criterion |
|---|---|---|---|
| X01: independent validation | E01 lacks a genuinely held-out evaluation of the final configuration | Freeze labels/head/threshold/prompt/context/batching; evaluate the existing 64-paper reference if eligible. If it was used for tuning, reserve an untouched set before making confirmatory claims. | Full per-sentence predictions, per-type P/R/F1, stage reachability, article-level uncertainty, manifest/config/date; identify tuning exposure. |
| X02: stronger full-article baseline | Only the current ten-paper unchunked, weak-retry pilot survives | Compare final funnel and a full-article baseline on the same held-out papers, with the same target policy/model, context-window-safe chunking and a documented retry/output-parsing policy. Save failed attempts rather than dropping them. | Same-paper stage/baseline predictions, failures, actual input/output/retry tokens and timings. Report paired effectiveness and resource measures. Do not select protocol on final-test scores. |
| X03: raw-PDF front-end evaluation | Paper will claim end-to-end PDF performance but E01/E08 lack a front-end audit | On a fixed human-labelled article sample compare gold source sentences, parsed sentences and localized regions. Separate parsing/segmentation losses from localization losses. | Per-gold alignment and error categories, stage-A reachability/compression, heading errors, same-paper structured-versus-PDF comparison where available. |
| X04: verifier mechanism ablations | Claiming why a context/batch/protection choice is best | On development data, hold detector candidates/model/prompt target fixed; compare no versus local context, selected batch sizes, and protection on/off where implemented. Freeze the selected setup for held-out evaluation. | Row decisions and TP retention/FP removal per setting, complete configs, resource use, separate dev/test report. Existing sweeps may satisfy part of this card. |
| X05: annotation-disagreement adjudication | Claiming that benchmark false positives are actually valid limitations | Select a defined representative disagreement sample plus controls. Have independent readers apply a written policy, retain individual labels and adjudication, and report sampling uncertainty. | Original and adjudicated labels, agreement, disagreement types and sampling weights if needed. LLM self-judgment alone does not establish corrected precision. |

X01 is the most consequential scientific gap **if** the completed human study cannot supply independent evaluation. X02 supports a stronger efficiency/comparison claim. X03 is necessary for a strong raw-PDF claim. X04/X05 are optional when the corresponding claims are omitted or bounded. Do not run all five merely to expand the paper.

## P2: integration and final review cards

- **W01 ? Integrate human results:** after E01, add the actual protocol/results and update all linked counts, abstract and claims. Reconsider the current RQs only if the restored study changes the paper's central contribution.
- **W02 ? Integrate venue analysis:** after E02/E03, add its actual research question, collection/analysis method, uncertainty/error discussion and figures. Current RQ3 means input efficiency; do not reuse its number silently for venue analysis.
- **W03 ? Scientific review:** a teammate checks every table against an artifact and every method against the producing code/config. Review the 17-study comparison against cited sources and verify any precise dataset sizes in the final venue version.
- **W04 ? Release/venue pass:** after evidence integration and venue selection, adapt the existing template, check venue-specific submission requirements, release access instructions and rebuild/render all pages. No venue is selected yet.

## Supervisor writing requirements: current coverage

| Requirement | Where implemented | Status / remaining dependency |
|---|---|---|
| Full pipeline, collection to analysis and each RQ | Methodology: task, data, collection/preparation, Stages A/B/C, design, measures; Figure 1 | Writing complete for experiments currently reported. Restore actual 64/800 study protocols after E01/E02. |
| Justify each method, metric, tool | Methodology explains local PDF tools, structural reachability, frozen heads/cues, fine-tuning, selective context/batching, P/R/F1, paper bootstrap and token accounting; RQ-design table | Justifications written; some training/provider provenance still needs E03?E05. No claim that every chosen parameter is optimal. |
| Pipeline diagram / flowchart | Figure 1: original three-column style, expanded to collection/preparation and RQ analysis, exact SAL counts | Restored and updated; this illustration distinguishes SAL from raw-PDF runner configurations. |
| First Methodology draft | `sections/03_methodology.tex` | Complete, now expanded. |
| Review/update other sections | Introduction, related work, chronological Table 1 with explicit relevance, results/discussion/conclusion and appendix | Revised; evidence claims bounded by available artifacts. |
| Supervisor's related-study table | Table 1: all 13 original study rows preserved, four relevant additions, explicit relevance column | Preserved; final article rows describe current reported scope. |

The writing requests are covered. Submission readiness still depends on recovering provenance/results, deciding which conditional experiments are necessary for the final claims, and selecting a venue. No new training, paid API inference, or messages to teammates were performed during this manuscript revision.
