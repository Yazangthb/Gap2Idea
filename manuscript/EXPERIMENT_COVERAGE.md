# Gap2Idea: minimal-branch experiment coverage

Authority: `minimal` at `28a6c2dbfad8ad332a66dd11e7b16f74d3ec4267`. Revision date: **4 November 2026**.

**54 experiment, development and protocol entries; all 72 tracked files accounted for.** This is a family-level inventory, not a claim that 54 independent experiments were completed.

## How to read the coverage

- **Main / appendix:** numerical observations are included, with their evidence status. Complete saved decisions are recomputed offline; aggregate files are transcribed; notebook and markdown references remain labelled development observations.
- **Protocol / mentioned:** the implementation or study design is described without inventing an outcome.
- **Performance excluded:** the entry is accounted for, but its number is not used to support the paper conclusions. The reason is given.
- Exact article membership, task definitions and metric denominators remain separate. The 362-row interface is not the earlier 64-paper study; the documented 230-paper reference is not the 800-paper venue corpus.

## Experiment-to-paper matrix

| ID | Experiment / family | Available evidence | Included in paper? Where? | Result / scope | Qualification or exclusion reason |
|---|---|---|---|---|---|
| A01 | Terminal versus mid-paper anchors | Executed notebook outputs | Main + appendix; Section 4 (p. 21), Table 13 (p. 31) | 433 -> 559 retained; loc@0.70 0.633 -> 0.857; mixed 49-statement silver reference. | No raw ten-paper corpus is included; these are preserved outputs, not a rerun. |
| A02 | Whole-section union (v3) | Implementation + notebook narrative | Appendix; no new measured row; Section A.2 (p. 31), Section 3.4 (p. 15) | Narrative says v3 matches v2 on fragmented internal blocks. | Executed side-by-side output contains v1/v2 only; no invented v3 metric. |
| A03 | PDF front end and structured-section alternative | Implemented/demo | Method + appendix; no accuracy result; Section 3.4 (p. 15), Section A.2 (p. 31) | PyMuPDF/pdfplumber front end and GROBID-section slicing implemented. | No exhaustive PDF heading or parser-comparison accuracy study. |
| A04 | SAL localization split and five-fold partitions | Notebook reference values | Main + appendix; Table 14 (p. 31), Section 4 (p. 21) | Anchors 0.861 +/- 0.054; whole sections 0.988 +/- 0.020; distinct SAL-training development protocol. | Token containment and supplied section names; not strict 40-paper test scoring or nested training CV. |
| A05 | Train cue mining and localization cue augmentation | Code + notebook reference | Appendix; Table 14 (p. 31), Section A.2 (p. 31) | Cue-augmented reachability 0.880 +/- 0.047; whole-section+cue 0.990 +/- 0.020; split before/after values retained. | Augmented drop accounting adds overlapping windows; fold cues are not re-mined per fold. |
| A06 | Localizer error localization | Notebook narrative + analysis code | Appendix; Section A.2 (p. 31) | Discussion/cue-less misses motivate section retention. | No independent error annotations supplied; no population-level missed-label claim. |
| B01 | Sci-Sentence/FWS fitting: frozen versus full FT | Markdown/notebook logs | Appendix; Table 15 (p. 32) | SF 6,700 training rows; frozen/DistilBERT/SciBERT val macro 0.633/0.745/0.846; per-target transfer retained. | Do not call transfer drop proof of overfitting alone; no checkpoints/predictions. |
| B02 | Adding SAL and PubMedBERT to mixed training | Markdown/notebook logs | Main discussion + appendix; Table 15 (p. 32), Table 9 (p. 25) | SFS 7,706 training rows; bge limitation recovery 0.455 -> 0.545; PubMedBERT pooled val macro 0.890. | Early per-dataset macro values conflict with consolidated conventions; those columns are not mixed into the paper. |
| B03 | Five frozen encoder sweep | Markdown/notebook logs | Main + appendix; Table 9 (p. 25), Table 15 (p. 32) | MiniLM limitation/future recovery 0.636/0.857, kept 170; full five-encoder target-specific comparisons retained. | Mixed-denominator recall_all is not the headline target metric. |
| B04 | Rules plus head versus head alone | Markdown logs | Appendix; Table 15 (p. 32) | Nine fits: head-only and hybrid per-target recall. | Candidate counts are review load, not precision. |
| B05 | MiniLM balancing/resampling/paraphrases | Notebook/log reference | Appendix with denominator warning; Section A.2 (p. 31) | All-49 recovery weighted 0.571; no weights 0.265; resample 0.551; augmented 0.571. | Includes 24 non-target types; no per-type decisions; not a clean two-target efficacy estimate. |
| B06 | Future-work and limitation cue optimization | Notebook reference + shipped rules | Appendix; Section A.2 (p. 31) | Curated FWS positive F1 0.585 -> 0.650; SAL cue F1 0.267 -> 0.275. | Rejected permissive FWS0.694 variant was affected by development selection; excluded as held-out evidence. |
| B07 | FWS within-corpus benchmark development | Notebook reference tables | Appendix; Table 16 (p. 33) | Balanced grouped holdout: SciBERT macro0.876/positive0.880; six methods. | Not directly comparable to published-paper scores or the native-prevalence grid; predictions unavailable. |
| B08 | SAL within-corpus benchmark development | Notebook reference tables | Appendix; Table 16 (p. 33) | New sampled training-article holdout: SciBERT macro0.900/positive0.843; six methods. | Not directly comparable to published-paper scores or the native-prevalence grid; predictions unavailable. |
| B09 | Sci-Sentence within-corpus benchmark development | Notebook reference tables | Appendix; Table 16 (p. 33) | Three proxy classes: SciBERT macro0.856; five methods; own grouped split. | Not directly comparable to published-paper scores or the native-prevalence grid; predictions unavailable. |
| B10 | Native-prevalence classical baseline grid | Saved aggregate JSON | Main; Table 7 (p. 24) | All-negative, TF-IDF/LR, binary-count NB, cues and bge/LR on SAL/FWS/Sci-Sentence; all rows retained. | FWS split differs from breadth pool. SAL separately fitted bge0.273 differs from cached bge0.282. |
| B11 | Legacy self-distillation/ACL-cap recipe | Package README description only | Mentioned; performance excluded; Section A.2 (p. 31), Section 3.5 (p. 16) | README describes 0 / 600 / 1,500 ACL augmentation; current notebooks instead use public mixtures. | No completed cap-ablation outputs in this branch; no legacy cache treated as current trained model. |
| B12 | SPECTER2 and regularized fine-tuning | Failed/unrun/proposed | Mentioned; no performance claim; Section A.2 (p. 31) | SPECTER2 legacy-loading failure; regularized FT proposed. | Not completed experiments with usable performance results. |
| C01 | Five-judge by three-batch easy SAL probe | Implemented + qualitative reference | Appendix; no numerical ranking; Section A.3 (p. 34) | Five configured models; 30 + 30 sample; batch 10 / 20 / 40. | No executed sweep table survives; lite@20 is a documented selection, not a verified optimum. |
| C02 | Stage B -> C SAL prototype and prevalence matching | Implemented notebook; output absent | Method history; no extra numerical result; Section A.3 (p. 34) | Frozen bge prototype; TEST_CAP1500 by default; can run full test. | References absent bench helpers; cannot assume default smoke run is full-stream evaluation. |
| C03 | SAL section reconstruction and eight-detector comparison | Full row predictions | Main; recomputed; Table 6 (p. 23), Table 4 (p. 22) | 7,968 rows/40 papers/205 positive; section gate203 TP / 1,885; all eight detector vectors. | BioC-conditioned stream, not raw-PDF validation. |
| C04 | Earlier broad-prompt union judging | Notebook reference only | Appendix; not headline; Section A.3 (p. 34) | Eight logged F1 values, PubMedBERT0.701. | Different batch membership/configuration; no saved raw union verdicts. Candidate-specific0.658 is principal. |
| C05 | Context/batch/prompt diagnosis | Narrative/log only | Appendix; bounded observation; Section A.3 (p. 34) | Hard-sample best approximately 0.827; recorded 56/60 versus 37/60 kept under different batch settings. | Missing complete configuration/decisions; not a controlled proof of batch/context causality. |
| C06 | Strict SAL-criteria union run across eight detectors | Saved aggregate JSON | Main interpretation + appendix; Table 18 (p. 34) | All eight rows; PubMedBERT F1 0.568. | Different judge/run from keep-biased lite; no prompt-only causal attribution. |
| C07 | Three-judge keep-biased hard-sample sweep | Saved aggregate JSON | Appendix; Table 17 (p. 34) | 120 candidates, 60+60; lite/pro/5.1 F1 0.789/0.779/0.784; six calls each. | Selection sample, not full-stream extraction performance; no billed cost/latency ranking. |
| C08 | Two-prompt union judging across eight detectors | Saved aggregate JSON | Appendix; Table 18 (p. 34) | All 16 prompt/detector aggregate combinations retained. | Union pool1,036; batch effects; missing ab_prev_verdicts means only aggregates can be inspected. |
| C09 | Candidate-specific PubMedBERT: two prompts | Full row predictions/verdicts | Main; recomputed; Table 4 (p. 22) | 391 candidates; RCT: 269 kept / 156 TP F1 .658; SAL: 249 kept / 148 TP F1 .652. | Bootstrap conditional on saved predictions and test-informed choices; historical RCT helper missing. |
| C10 | Section-pool prevalence sensitivity | Offline replay from saved decisions | Main; recomputed; Table 5 (p. 23) | 203 positives+778 sampled negatives; eight seeds; AB/RCT/SAL meanF1 .749/.757/.741. | Scoring sensitivity, not a fresh API run or faithful published checkpoint reproduction. |
| C11 | Neighbor-position sensitivity W0/1/2 | Implemented scorer + saved decisions | Main + appendix; recomputed; Table 20 (p. 35) | RCT F1 .658/.821/.853 for W0/1/2. | Many-to-many positional credit, not semantic span truth or improved strict model accuracy. |
| C12 | SAL disagreement/error categorization | Single-reader narrative | Discussion + appendix; projected precision excluded; Section 5 (p. 26), Section A.3 (p. 34) | 22/101 FP sample; roughly 45/36/18% categories; independent human adjudication absent. | The stated projected precision about 0.87 is unsupported. Even reclassifying 45% of 101 FP gives (148+.45*101)/249 = approximately 0.777, not .87; sampling/labels unverified. |
| C13 | Kilicoglu sentence cross-check | Notebook reference only | Appendix; Section A.3 (p. 34) | n=1,505, approximately 31.8% positive; detectorP/R/F1 .719/.781/.749; +C .876/.503/.639. | No row decisions or reproduction of original rule baseline; not raw-PDF front-end validation. |
| D01 | FWS breadth funnel | Saved aggregate JSON | Main; Table 8 (p. 24) | 1,506/220 positives; AB F1 .667 -> ABC .707. | Exact training-sentence exclusions do not prove paper-disjointness; different pool from baseline grid. |
| D02 | Sci-Sentence author-gap breadth run | Saved aggregate JSON | Main; Table 8 (p. 24) | 140/40 positive; .612 -> .600; prior-work target mismatched. | Not evidence of universally beneficial verification. |
| D03 | Intended gap-matched Sci-Sentence prompt | Markdown logged aggregate + code | Main + appendix; Table 19 (p. 35) | P/R/F1 1.000/.575/.730. | Base allows prior work but shared wrapper rejects it; no raw decisions; exploratory test feedback. |
| D04 | Sci-Sentence in-domain/recall-first adaptation | Markdown logged aggregate + code | Main + appendix; Table 19 (p. 35) | bge/LR threshold 0.5 F1 .814; threshold 0.3 .772; +intended-gap judge .880. | 140-row binary task, not seven-class leaderboard; wrapper conflict; no final untouched test. |
| D05 | FWS in-domain threshold sweep | Implemented, no result output | Appendix; no measured row; Section A.4 (p. 35) | Threshold 0.5/.4/.3/.2; optional StageC@.3. | Different pool from breadth; cannot invent results from script intent. |
| D06 | FWS kept-negative inspection | Implementation + narrative examples | Appendix; no corrected scores; Section A.4 (p. 35) | Candidate disagreement inspection. | Examples are not independently adjudicated gold corrections. |
| E01 | Forty-paper input-token accounting | Saved per-paper CSV | Main; recomputed; Table 10 (p. 26) | 282846/75622=3.7403; input reduction73.3%. | Estimated cl100k profile with wrappers/context; not billed total cost or faithful request log. |
| E02 | Ten-paper full-article LLM pilot | Saved aggregate JSON + run description | Main; Table 11 (p. 26) | n=10; P .559/R .345; derived F1 approximately 0.427; reported input 6,540/paper. | Raw responses unavailable; two parse failures described; no robust chunk/retry baseline. Older .429 F1 corrected. |
| E03 | Ten-paper per-stage funnel | Saved aggregate counts | Main; arithmetic verified; Table 11 (p. 26) | 2053 sentences/55 gold; A498/55 TP; AB109/51; ABC77/45, F1 .682. | Same selected ten-paper sample; no combining its actual input with40-paper estimates into a billed ratio. |
| E04 | Batching/caching/long-paper dollar projections | Scenario calculations in notes | Discussion only; numerical savings projections excluded; Section 5 (p. 26) | Cross-paper/caching totals64248/55508 and long-paper13.2x are hypothetical profiles. | No saved scenario request traces, runtime or billing; no measured efficiency claim from projections. |
| G01 | Expanded three-type reference collection | Datasheet-only statistics | Method + appendix; explicitly documented; Section 3.9 (p. 20), Table 21 (p. 36) | 230 papers / 68,719 sentences / 4,462 positives; 190 arXiv + 40 SAL; type/domain totals reconcile. | Full manifest/labels, generating scripts and provider logs absent; not a validated released benchmark. |
| G02 | Two-model agreement and adjudication tiers | Datasheet-only statistics | Appendix; Table 21 (p. 36) | 2,960 agreement + 1,502 adjudicated;1,737 candidates = 1,502 accepted + 235 rejected. | Model-agreement tier is not human confidence calibration; identifiers unverified request metadata. |
| G03 | Inter-model kappa and SAL overlap checks | Datasheet-only statistics | Appendix; Table 21 (p. 36) | Any-gap kappa 0.778; typed kappas; SAL binaryrecall 0.956/limitationrecall 0.927. | Not human agreement; not an independent holdout; .927=190/205, typed and binary recall differ. |
| G04 | Third-model check and annotation expenditure | Datasheet-only statistics | Appendix; Table 21 (p. 36) | 116/227 extras accepted; documented spend approximately USD 37.5. | Another model is not a human gold adjudicator; no request/billing records 0.227 is not saved-funnel FP count. |
| G05 | Blind human-validation interface | Generated targets + builder | Method + appendix; UI counts verified; Section 3.9 (p. 20) | 362 numbered targets/40 papers, no embedded system-label fields; four documented strata; seed13. | Only biomedical disagreements/control sample; does not validate all190 arXiv papers. Builder imports absent gold_label_full. |
| G06 | Human annotation merge/scoring | Implemented; no completed answers | Protocol only; human results excluded; Section 3.9 (p. 20) | Coverage, kappa, consensus-only and either-positive summaries implemented. | No completed exports/key/report supplied. Excluding disagreements is not independent adjudication or population precision. |
| H01 | Alternative19/33-item target-only references | Markdown development logs | Main sensitivity + appendix; Section A.2 (p. 31) | 33:reach23/33/recovery 0.455;19:reach14/19/recovery 0.632; different model references on same articles. | Incomplete long-paper reading/40k cap; not a system improvement or human benchmark. |
| H02 | 49-item silver-reference validation | Markdown development log | Method + appendix; Section 3.2 (p. 12), Section A.2 (p. 31) | 49 source-overlap checks,25 target labels,42/49 reachable; non-target classes separated. | Token containment is not a literal quotation proof; no raw 49-item file in branch; historic paper-folder19item file not assumed same set. |
| H03 | Historical LimGen binary detection | Comparison-note log only | Performance excluded; Section A.7 (p. 38) | Logged DistilBERT 0.629/SciBERT 0.743; original generation corpus repurposed. | No producing predictions/checkpoints/harness in minimal; not an official detection leaderboard result. |
| H04 | Historical BAGELS positive coverage | Comparison-note log only | Performance excluded; Section A.7 (p. 38) | Logged coverage .738 -> .434 after C. | No producing artifact/harness; coverage cannot establish precision.Study remains in related-workTable1. |
| H05 | Older reconstructed FWS/SAL leaderboard runs | Comparison-note logs | Excluded as primary results; Section A.7 (p. 38), Section A.2 (p. 31) | FWS 0.879/.877/.864 andSAL 0.762 -> 0.810--0.812 logged. | Not saved-prediction current protocol; published-paper leakage/SOTA allegations not established by supplied outputs. |
| H06 | Copied corrected legacy audit / N100 / idea tooling | Historical audit prose | Excluded from current research evidence; Section A.7 (p. 38) | Old refs and local-data claims refer to another checkout. | No N100 corpus or idea-generation evaluation in minimal; do not resurrect original40-family audit as current evidence. |
| P01 | Additional comparisons, annotation, idea evaluation | Plans or unavailable studies | Not completed-result claims; Section A.7 (p. 38) | Regularized/cross-provider baselines and64/800studies not evidenced in this snapshot. | Citation-reuse counts are a title/abstract scan, not a systematic proof that no shared benchmarks exist. |

## Exact source paths

- **A01:** [stage_a_package/stage_a_all_experiments.ipynb](../Gap2Idea/stage_a_package/stage_a_all_experiments.ipynb), [stage_a_package/stage_a_final.ipynb](../Gap2Idea/stage_a_package/stage_a_final.ipynb)
- **A02:** [stage_a_package/stage_a_all_experiments.ipynb](../Gap2Idea/stage_a_package/stage_a_all_experiments.ipynb), [stage_a_package/stage_a_final.ipynb](../Gap2Idea/stage_a_package/stage_a_final.ipynb), [src/gap2idea/pipeline/gap_funnel.py](../Gap2Idea/src/gap2idea/pipeline/gap_funnel.py)
- **A03:** [stage_a_package/stage_a_final.ipynb](../Gap2Idea/stage_a_package/stage_a_final.ipynb), [src/gap2idea/pipeline/sections.py](../Gap2Idea/src/gap2idea/pipeline/sections.py), [src/gap2idea/pipeline/gap_funnel.py](../Gap2Idea/src/gap2idea/pipeline/gap_funnel.py)
- **A04:** [stage_a_package/stage_a_optimization.ipynb](../Gap2Idea/stage_a_package/stage_a_optimization.ipynb)
- **A05:** [stage_a_package/stage_a_optimization.ipynb](../Gap2Idea/stage_a_package/stage_a_optimization.ipynb)
- **A06:** [stage_a_package/stage_a_optimization.ipynb](../Gap2Idea/stage_a_package/stage_a_optimization.ipynb)
- **B01:** [stage_b_package/v100_results/RESULTS.md](../Gap2Idea/stage_b_package/v100_results/RESULTS.md), [stage_b_package/v100_results/ALL_METRICS.md](../Gap2Idea/stage_b_package/v100_results/ALL_METRICS.md), [stage_b_package/stage_b_all_experiments.ipynb](../Gap2Idea/stage_b_package/stage_b_all_experiments.ipynb)
- **B02:** [stage_b_package/v100_results/RESULTS_sal.md](../Gap2Idea/stage_b_package/v100_results/RESULTS_sal.md), [stage_b_package/v100_results/ALL_METRICS.md](../Gap2Idea/stage_b_package/v100_results/ALL_METRICS.md), [stage_b_package/stage_b_final.ipynb](../Gap2Idea/stage_b_package/stage_b_final.ipynb)
- **B03:** [stage_b_package/v100_results/RESULTS_sweep.md](../Gap2Idea/stage_b_package/v100_results/RESULTS_sweep.md), [stage_b_package/v100_results/ALL_METRICS.md](../Gap2Idea/stage_b_package/v100_results/ALL_METRICS.md), [stage_b_package/stage_b_all_experiments.ipynb](../Gap2Idea/stage_b_package/stage_b_all_experiments.ipynb)
- **B04:** [stage_b_package/v100_results/ALL_METRICS.md](../Gap2Idea/stage_b_package/v100_results/ALL_METRICS.md), [stage_b_package/stage_b_all_experiments.ipynb](../Gap2Idea/stage_b_package/stage_b_all_experiments.ipynb)
- **B05:** [stage_b_package/stage_b_all_experiments.ipynb](../Gap2Idea/stage_b_package/stage_b_all_experiments.ipynb), [stage_b_package/v100_results/MODEL_COMPARISON.md](../Gap2Idea/stage_b_package/v100_results/MODEL_COMPARISON.md)
- **B06:** [stage_b_package/stage_b_all_experiments.ipynb](../Gap2Idea/stage_b_package/stage_b_all_experiments.ipynb), [src/gap2idea/pipeline/gap_funnel.py](../Gap2Idea/src/gap2idea/pipeline/gap_funnel.py)
- **B07:** [stage_b_package/stage_b_model_comparison.ipynb](../Gap2Idea/stage_b_package/stage_b_model_comparison.ipynb)
- **B08:** [stage_b_package/stage_b_model_comparison.ipynb](../Gap2Idea/stage_b_package/stage_b_model_comparison.ipynb)
- **B09:** [stage_b_package/stage_b_model_comparison.ipynb](../Gap2Idea/stage_b_package/stage_b_model_comparison.ipynb)
- **B10:** [stage_c_package/baseline_grid.py](../Gap2Idea/stage_c_package/baseline_grid.py), [stage_c_package/baseline_grid_results.json](../Gap2Idea/stage_c_package/baseline_grid_results.json), [docs/experiments/comparison_plan.md](../Gap2Idea/docs/experiments/comparison_plan.md)
- **B11:** [stage_b_package/README.md](../Gap2Idea/stage_b_package/README.md), [scripts/run_extraction_pipeline.py](../Gap2Idea/scripts/run_extraction_pipeline.py)
- **B12:** [stage_b_package/v100_results/RESULTS_sweep.md](../Gap2Idea/stage_b_package/v100_results/RESULTS_sweep.md), [stage_b_package/v100_results/MODEL_COMPARISON.md](../Gap2Idea/stage_b_package/v100_results/MODEL_COMPARISON.md)
- **C01:** [stage_c_package/stage_c_all_experiments.ipynb](../Gap2Idea/stage_c_package/stage_c_all_experiments.ipynb), [stage_c_package/stage_c_final.ipynb](../Gap2Idea/stage_c_package/stage_c_final.ipynb)
- **C02:** [stage_c_package/stage_c_all_experiments.ipynb](../Gap2Idea/stage_c_package/stage_c_all_experiments.ipynb)
- **C03:** [stage_c_package/sal_criteria/sal_test_allmodels.json](../Gap2Idea/stage_c_package/sal_criteria/sal_test_allmodels.json), [stage_c_package/stage_c_end_to_end_funnel.ipynb](../Gap2Idea/stage_c_package/stage_c_end_to_end_funnel.ipynb)
- **C04:** [stage_c_package/stage_c_end_to_end_funnel.ipynb](../Gap2Idea/stage_c_package/stage_c_end_to_end_funnel.ipynb)
- **C05:** [stage_c_package/stage_c_end_to_end_funnel.ipynb](../Gap2Idea/stage_c_package/stage_c_end_to_end_funnel.ipynb), [docs/experiments/sal_stage_c_summary.md](../Gap2Idea/docs/experiments/sal_stage_c_summary.md)
- **C06:** [stage_c_package/sal_criteria/funnel_salcriteria_results.json](../Gap2Idea/stage_c_package/sal_criteria/funnel_salcriteria_results.json), [stage_c_package/stage_c_end_to_end_funnel.ipynb](../Gap2Idea/stage_c_package/stage_c_end_to_end_funnel.ipynb)
- **C07:** [stage_c_package/sal_criteria/newcriteria_sweep_results.json](../Gap2Idea/stage_c_package/sal_criteria/newcriteria_sweep_results.json), [stage_c_package/sal_criteria/stagec_newcriteria_sweep.py](../Gap2Idea/stage_c_package/sal_criteria/stagec_newcriteria_sweep.py)
- **C08:** [stage_c_package/sal_criteria/ab_prev_funnel_results.json](../Gap2Idea/stage_c_package/sal_criteria/ab_prev_funnel_results.json), [stage_c_package/sal_criteria/stagec_newcriteria_ab_prev.py](../Gap2Idea/stage_c_package/sal_criteria/stagec_newcriteria_ab_prev.py)
- **C09:** [stage_c_package/sal_criteria/faithful_pubmed_verdicts.json](../Gap2Idea/stage_c_package/sal_criteria/faithful_pubmed_verdicts.json), [stage_c_package/sal_criteria/stagec_faithful_pubmed.py](../Gap2Idea/stage_c_package/sal_criteria/stagec_faithful_pubmed.py)
- **C10:** [stage_c_package/sal_criteria/stagec_faithful_pubmed.py](../Gap2Idea/stage_c_package/sal_criteria/stagec_faithful_pubmed.py), [stage_c_package/sal_criteria/faithful_pubmed_verdicts.json](../Gap2Idea/stage_c_package/sal_criteria/faithful_pubmed_verdicts.json)
- **C11:** [scripts/span_eval.py](../Gap2Idea/scripts/span_eval.py), [stage_c_package/sal_criteria/sal_test_allmodels.json](../Gap2Idea/stage_c_package/sal_criteria/sal_test_allmodels.json), [stage_c_package/sal_criteria/faithful_pubmed_verdicts.json](../Gap2Idea/stage_c_package/sal_criteria/faithful_pubmed_verdicts.json)
- **C12:** [docs/experiments/sal_stage_c_summary.md](../Gap2Idea/docs/experiments/sal_stage_c_summary.md), [stage_c_package/stage_c_end_to_end_funnel.ipynb](../Gap2Idea/stage_c_package/stage_c_end_to_end_funnel.ipynb)
- **C13:** [stage_c_package/stage_c_end_to_end_funnel.ipynb](../Gap2Idea/stage_c_package/stage_c_end_to_end_funnel.ipynb)
- **D01:** [stage_c_package/breadth_results.json](../Gap2Idea/stage_c_package/breadth_results.json), [stage_c_package/breadth_funnel.py](../Gap2Idea/stage_c_package/breadth_funnel.py)
- **D02:** [stage_c_package/breadth_results.json](../Gap2Idea/stage_c_package/breadth_results.json), [stage_c_package/breadth_funnel.py](../Gap2Idea/stage_c_package/breadth_funnel.py)
- **D03:** [docs/experiments/cross_domain_breadth.md](../Gap2Idea/docs/experiments/cross_domain_breadth.md), [stage_c_package/breadth_diag.py](../Gap2Idea/stage_c_package/breadth_diag.py)
- **D04:** [docs/experiments/cross_domain_breadth.md](../Gap2Idea/docs/experiments/cross_domain_breadth.md), [stage_c_package/scisent_recall_first.py](../Gap2Idea/stage_c_package/scisent_recall_first.py)
- **D05:** [stage_c_package/fws_recall_first.py](../Gap2Idea/stage_c_package/fws_recall_first.py)
- **D06:** [stage_c_package/breadth_diag.py](../Gap2Idea/stage_c_package/breadth_diag.py), [docs/experiments/cross_domain_breadth.md](../Gap2Idea/docs/experiments/cross_domain_breadth.md)
- **E01:** [stage_c_package/cost_tokens.csv](../Gap2Idea/stage_c_package/cost_tokens.csv), [stage_c_package/cost_tokens.py](../Gap2Idea/stage_c_package/cost_tokens.py), [docs/experiments/cost_analysis.md](../Gap2Idea/docs/experiments/cost_analysis.md)
- **E02:** [stage_c_package/cost_fullpaper_results.json](../Gap2Idea/stage_c_package/cost_fullpaper_results.json), [stage_c_package/cost_fullpaper.py](../Gap2Idea/stage_c_package/cost_fullpaper.py), [docs/experiments/cost_analysis.md](../Gap2Idea/docs/experiments/cost_analysis.md)
- **E03:** [stage_c_package/cost_perstage_results.json](../Gap2Idea/stage_c_package/cost_perstage_results.json), [stage_c_package/cost_perstage.py](../Gap2Idea/stage_c_package/cost_perstage.py)
- **E04:** [docs/experiments/cost_analysis.md](../Gap2Idea/docs/experiments/cost_analysis.md)
- **G01:** [annotations/README.md](../Gap2Idea/annotations/README.md)
- **G02:** [annotations/README.md](../Gap2Idea/annotations/README.md)
- **G03:** [annotations/README.md](../Gap2Idea/annotations/README.md)
- **G04:** [annotations/README.md](../Gap2Idea/annotations/README.md)
- **G05:** [annotations/annotate.html](../Gap2Idea/annotations/annotate.html), [scripts/dataset/build_annotate_html.py](../Gap2Idea/scripts/dataset/build_annotate_html.py), [annotations/README.md](../Gap2Idea/annotations/README.md)
- **G06:** [scripts/dataset/gold_human_merge.py](../Gap2Idea/scripts/dataset/gold_human_merge.py), [annotations/README.md](../Gap2Idea/annotations/README.md)
- **H01:** [stage_b_package/v100_results/GOLD_CLEAN_EVAL.md](../Gap2Idea/stage_b_package/v100_results/GOLD_CLEAN_EVAL.md)
- **H02:** [stage_b_package/v100_results/GOLD_VALIDATION.md](../Gap2Idea/stage_b_package/v100_results/GOLD_VALIDATION.md)
- **H03:** [stage_b_package/v100_results/MODEL_COMPARISON.md](../Gap2Idea/stage_b_package/v100_results/MODEL_COMPARISON.md), [docs/experiments/comparison_plan.md](../Gap2Idea/docs/experiments/comparison_plan.md)
- **H04:** [stage_b_package/v100_results/MODEL_COMPARISON.md](../Gap2Idea/stage_b_package/v100_results/MODEL_COMPARISON.md), [docs/experiments/comparison_plan.md](../Gap2Idea/docs/experiments/comparison_plan.md)
- **H05:** [stage_b_package/v100_results/MODEL_COMPARISON.md](../Gap2Idea/stage_b_package/v100_results/MODEL_COMPARISON.md), [docs/experiments/comparison_plan.md](../Gap2Idea/docs/experiments/comparison_plan.md)
- **H06:** [stage_a_package/AUDIT_CORRECTED.md](../Gap2Idea/stage_a_package/AUDIT_CORRECTED.md)
- **P01:** [PUBLICATION_PLAN.md](../Gap2Idea/PUBLICATION_PLAN.md), [docs/experiments/comparison_plan.md](../Gap2Idea/docs/experiments/comparison_plan.md), [docs/experiments/dataset_citation_audit.md](../Gap2Idea/docs/experiments/dataset_citation_audit.md)

## Complete tracked-file coverage

| Branch file | Experiment IDs or supporting role |
|---|---|
| `.env.example` | Package, configuration or provider utility; no independent experiment outcome. |
| `.gitignore` | Package, configuration or provider utility; no independent experiment outcome. |
| `PUBLICATION_PLAN.md` | P01 |
| `README.md` | Overview; several historical headlines are superseded by candidate-specific evidence. |
| `annotations/README.md` | G01, G02, G03, G04, G05, G06 |
| `annotations/annotate.html` | G05 |
| `docs/experiments/REPRODUCIBILITY.md` | Regeneration and cache constraints; checked against actual scripts. |
| `docs/experiments/comparison_plan.md` | B10, H03, H04, H05, P01 |
| `docs/experiments/cost_analysis.md` | E01, E02, E04 |
| `docs/experiments/cross_domain_breadth.md` | D03, D04, D06 |
| `docs/experiments/dataset_citation_audit.md` | P01 |
| `docs/experiments/sal_stage_c_summary.md` | C05, C12 |
| `pipeline_package/run_all_stages.ipynb` | Runnable demonstration wrapper; no executed research-result output. |
| `pyproject.toml` | Package, configuration or provider utility; no independent experiment outcome. |
| `requirements.txt` | Package, configuration or provider utility; no independent experiment outcome. |
| `scripts/bench/dump_sal_preds.py` | Frozen-head regeneration recipe; preservesFT columns; no inference rerun here. |
| `scripts/dataset/build_annotate_html.py` | G05 |
| `scripts/dataset/gold_human_merge.py` | G06 |
| `scripts/run_extraction_pipeline.py` | B11 |
| `scripts/span_eval.py` | C11 |
| `src/gap2idea/__init__.py` | Package, configuration or provider utility; no independent experiment outcome. |
| `src/gap2idea/pipeline/__init__.py` | Package, configuration or provider utility; no independent experiment outcome. |
| `src/gap2idea/pipeline/gap_funnel.py` | A02, A03, B06 |
| `src/gap2idea/pipeline/gap_llm_filter.py` | Verifier request composition, wrapper conflict, schema, fallback and protection behavior. |
| `src/gap2idea/pipeline/gap_prefilter.py` | Package, configuration or provider utility; no independent experiment outcome. |
| `src/gap2idea/pipeline/llm.py` | Package, configuration or provider utility; no independent experiment outcome. |
| `src/gap2idea/pipeline/sections.py` | A03 |
| `src/gap2idea/pipeline/stage_c_prompts.py` | Exact current SAL/GAP base instructions and examples. |
| `src/gap2idea/utils.py` | Package, configuration or provider utility; no independent experiment outcome. |
| `stage_a_package/AUDIT_CORRECTED.md` | H06 |
| `stage_a_package/README.md` | Package overview; source notebooks take precedence over stale v2-only label. |
| `stage_a_package/stage_a_all_experiments.ipynb` | A01, A02 |
| `stage_a_package/stage_a_final.ipynb` | A01, A02, A03 |
| `stage_a_package/stage_a_optimization.ipynb` | A04, A05, A06 |
| `stage_b_package/DATASETS.md` | Training label maps and resource inventory; checked against notebook transforms. |
| `stage_b_package/README.md` | B11 |
| `stage_b_package/stage_b_all_experiments.ipynb` | B01, B03, B04, B05, B06 |
| `stage_b_package/stage_b_final.ipynb` | B02 |
| `stage_b_package/stage_b_model_comparison.ipynb` | B07, B08, B09 |
| `stage_b_package/v100_results/ALL_METRICS.md` | B01, B02, B03, B04 |
| `stage_b_package/v100_results/GOLD_CLEAN_EVAL.md` | H01 |
| `stage_b_package/v100_results/GOLD_VALIDATION.md` | H02 |
| `stage_b_package/v100_results/MODEL_COMPARISON.md` | B05, B12, H03, H04, H05 |
| `stage_b_package/v100_results/RESULTS.md` | B01 |
| `stage_b_package/v100_results/RESULTS_sal.md` | B02 |
| `stage_b_package/v100_results/RESULTS_sweep.md` | B03, B12 |
| `stage_c_package/README.md` | Package overview; broad0.701 superseded for principal result. |
| `stage_c_package/baseline_grid.py` | B10 |
| `stage_c_package/baseline_grid_results.json` | B10 |
| `stage_c_package/breadth_diag.py` | D03, D06 |
| `stage_c_package/breadth_funnel.py` | D01, D02 |
| `stage_c_package/breadth_results.json` | D01, D02 |
| `stage_c_package/cost_fullpaper.py` | E02 |
| `stage_c_package/cost_fullpaper_results.json` | E02 |
| `stage_c_package/cost_perstage.py` | E03 |
| `stage_c_package/cost_perstage_results.json` | E03 |
| `stage_c_package/cost_tokens.csv` | E01 |
| `stage_c_package/cost_tokens.py` | E01 |
| `stage_c_package/fws_recall_first.py` | D05 |
| `stage_c_package/sal_criteria/README.md` | Experiment guide; no independent support for corrected precision. |
| `stage_c_package/sal_criteria/ab_prev_funnel_results.json` | C08 |
| `stage_c_package/sal_criteria/faithful_pubmed_verdicts.json` | C09, C10, C11 |
| `stage_c_package/sal_criteria/funnel_salcriteria_results.json` | C06 |
| `stage_c_package/sal_criteria/newcriteria_sweep_results.json` | C07 |
| `stage_c_package/sal_criteria/sal_test_allmodels.json` | C03, C11 |
| `stage_c_package/sal_criteria/stagec_faithful_pubmed.py` | C09, C10 |
| `stage_c_package/sal_criteria/stagec_newcriteria_ab_prev.py` | C08 |
| `stage_c_package/sal_criteria/stagec_newcriteria_sweep.py` | C07 |
| `stage_c_package/scisent_recall_first.py` | D04 |
| `stage_c_package/stage_c_all_experiments.ipynb` | C01, C02 |
| `stage_c_package/stage_c_end_to_end_funnel.ipynb` | C03, C04, C05, C06, C12, C13 |
| `stage_c_package/stage_c_final.ipynb` | C01 |

## Contradictions resolved in the manuscript

1. **SAL 0.701 versus 0.658:** these are different union-candidate and candidate-specific runs. The principal recomputed result is 0.658; the older run is identified separately.
2. **MiniLM, bge and PubMedBERT:** the transfer notebook, portable runner and SAL binary experiment use different heads/training. The paper defines their configurations instead of asserting one universal final classifier.
3. **Macro versus positive-class F1:** these measures are labelled separately; the SAL development macro-F1 of 0.900 is not compared with published positive-class F1 of 0.821.
4. **Different FWS test pools:** the baseline grid and breadth experiment are not a paired comparison.
5. **Different label policies:** author-stated extraction, binary Sci-Sentence Limitation + Research Gap, and three-class proxy training are distinct tasks.
6. **Prompt composition conflict:** the intended gap-matched base allows prior-work statements, but the shared batch suffix rejects them. Saved results are exploratory; no silent source repair or fictitious rerun was made.
7. **Model reference versus human reference:** the 230-paper totals are datasheet-only; the 362-row interface is verified, but completed answers are absent. Inter-model kappa is not human agreement or a human performance ceiling.
8. **Neighbor credit versus corrected gold:** the W=1 RCT score of 0.821 does not reproduce the published SAL score of 0.821 under the same protocol.
9. **Estimated input versus actual billing:** the 3.7403 ratio excludes output/retries/local processing; the ten-paper pilot is not combined with forty-paper estimates into a billed saving.
10. **Full-article baseline F1:** rounded P=0.559 and R=0.345 imply approximately 0.427; the archived 0.429 headline is corrected.
11. **Projected true precision:** the unsupported 0.87 claim is excluded. Even assuming that 45% of the 101 false positives were valid gives (148 + 0.45 * 101) / 249 = approximately 0.777; neither that assumption nor that projection is validated.
12. **Legacy local-data claims:** the copied audit concerns another checkout. It cannot establish that missing data, runs or checkpoints exist in this snapshot.
13. **Localizer cap:** 160 applies to the terminal-window selection; mid-paper additions and the whole-section union have no global 160-sentence cap.

## Reproduction and final review

- [Offline verification script](revision_audit/verify_minimal.py): no API calls or model training.
- [Verified metrics and all source hashes](revision_audit/2026-11-04/verified_results.json).
- [Machine-readable experiment registry](revision_audit/2026-11-04/experiment_registry.json).
- [Final review and remaining human decisions](FINAL_REVIEW.md).

The page numbers above refer to the checked 4 November build. Regenerate this mapping if later edits change pagination.
