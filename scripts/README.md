# scripts/ — experiment & ops scripts

Grouped by purpose. Run from the **repo root**, e.g. `python scripts/bench/bench_gap_recall.py`.
Each script adds `src/` to its path via `parents[2]`; sibling imports resolve within the
same folder (a few also add the specific hub folder they depend on, e.g. `training/`
scripts that import `bench/bench_gap_recall.py`).

| Folder | Purpose | Key scripts |
|---|---|---|
| **training/** | Stage-B classifier heads | `train_gap_head.py` (ships `data/gap_head.joblib`), `finetune_rct.py` (BiomedBERT detector, dual-threshold + prevalence-matched eval), `sweep_acl_cap.py`, `sweep_encoders.py`, `test_bert_stageb.py`, `test_setfit_stageb.py`, `test_scibert_gold.py`, `prep_scibert_data.py`, `_distilbert_worker.py` |
| **bench/** | Benchmarks | `bench_gap_recall.py` (Stage-A + end-to-end vs gold), `bench_gap_funnel.py`, `bench_limgen.py` (LimGen head-to-head), `bench_research_single.py`, `bench_stage_c.py`, `ab_test_stage_a.py`, `compare_extraction_bench.py`, `plot_stage_a.py`, `demo_funnel.py`, `dump_funnel_preds.py`, `enhance_stageb.py`, `finetune_and_chain.py`, `analyze_stage_c_misses.py` |
| **dataset/** | Gold + corpus building | `build_gap_gold.py` (ships `data/bench_gap/gold_sentences.tsv`), `build_gold_dataset.py`, `build_gold_v2.py`, `add_provenance.py`, `harvest_acl_limitations.py` (the ACL data fix), `extract_v{4..8}_papers.py`, `extract_test25.py`, `scale_batches{,2}.py`, `reextract_v2_and_redataset.py`, `gpu_dataset_v3.py`, `build_dataset_10papers.py`, `test_strict_precision.py`, `expand_corpus_demo.py`, `audit_gold_pipeline.py` |
| **stage_c/** | LLM precision-filter prompt iteration | `iterate_stage_c.py`, `stage_c_limgen.py`, `_stageb_limgen_worker.py`, `test_openrouter_stage_c.py`, `test_gap_junk_context.py`, `test_3b_prompts.py`, `test_safe_limgen_prompt.py`, `iter_openrouter_prompt.py`, `iter_rules_openrouter.py`, `prompt_iter.py`, `validate_iterated_prompt.py` |
| **gen/** | Idea generation + paper drafts | `gen_paper_drafts.py`, `gen_ai_drafts_only.py`, `gen_v3_drafts_and_export.py`, `gen_analysis_cv_ideas.py`, `gen_math_ideas.py`, `pick_math_ai_ideas.py`, `redraft_analysis_math.py`, `retry_top6_with_full_critic.py` |
| **eval/** | Pipeline evaluation + analysis | `eval_all_stages_gold.py`, `eval_full_pipeline_goldv2.py`, `gpu_eval_v2.py`, `analyze_eval_responses.py`, `investigate_losses.py`, `semantic_precision_audit.py`, `test_pipeline_v3.py`, `test_extraction_v2.py` |
| **deploy/** | Human evaluator form | `build_evaluator_form_gs.py`, `build_evaluator_xlsx.py`, `create_evaluator_form.py` |
| **archive/** | Superseded (lexical Tier-0 exploration) | `mine_tier0_dictionary.py`, `eval_tier0.py`, `verify_tier0.py`, `show_gt_gaps.py`, `build_label_sheet.py`, `eval_gap_extraction.py`, `check_resolved.py` |

## Benchmark suites — vs published SOTA (extraction phases)

These compare the funnel (Stage B detector + Stage C LLM precision filter, `mode="validate_rct"`
/ `validate_fws`) against dataset-specific published SOTA. All live flat in `bench/`;
**`bench_limgen.py` is the shared metrics hub** — `prf`, `build_xy`, `fetch`, `m_bge` — imported
by the others, and `bench_rct.py` / `bench_fws.py` export the per-dataset `load` + `stage_c`.

| Suite | Gap type | Scripts | SOTA compared | Write-up |
|---|---|---|---|---|
| **RCT / SAL** | limitations | `bench_rct.py`, `build_rct_pool.py` (BioC section pool), `training/finetune_rct.py`, `bench_stagec_opt.py` | Lan 2024 PubMedBERT P .751 / R .907 / **F1 .821**; we match precision, ~.01 F1 gap (recall) | [rct_stagec.md](../docs/experiments/rct_stagec.md) |
| **FWS** | future work | `bench_fws.py` (zero-shot / `--train` encoders), `bench_fws_cheap.py` (lexical), `bench_fws_balanced.py` (their protocol) | Zhang 2022 BernoulliNB **macro-F1 .9073** — reproduces (.9072) but is chi²-leakage; honest ceiling ~.85 | [fws_recognition.md](../docs/experiments/fws_recognition.md) |
| **LimGen** | limitations | `bench_limgen.py`, `bench_limgen_stagec.py` | LimGen Stage-B head-to-head + Stage-C lift | [limgen_stagec.md](../docs/experiments/limgen_stagec.md) |
| **BAGELS** | limitations | `bench_bagels.py` | whole-section-positive labelling (noisy for sentence F1) | [bagels_output.md](../docs/experiments/bagels_output.md) |
| **Stage C opt** | — | `bench_stagec_opt.py`, `analyze_stage_c_misses.py` | prompt/model tuning; residual FPs are label noise (both datasets) | rct_stagec / fws_recognition |

Key cross-dataset findings (see write-ups): (1) *data not model* — small fine-tuned bge-small
matches/beats 110M PubMedBERT/SciBERT everywhere; (2) Stage C is a **low-prevalence precision
tool** — clear lift on the 2–3% full stream, neutral on curated high-prevalence pools; (3) both
published SOTAs come back to ~honest baselines once leakage/prevalence/labels are controlled.
