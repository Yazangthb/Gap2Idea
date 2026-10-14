# Reproducibility — do experiments reflect method changes on rerun?

If you tweak the method (a prompt, the model head, the detector, the section rules) and rerun the
experiment scripts, **some changes propagate automatically and some are pinned**. Read this before
assuming a rerun used your change.

## Auto-propagates (scripts import the live method code)
- **Section-gate / cue-rule logic** in `src/gap2idea/pipeline/gap_funnel.py`
  (`slice_section_aware`, `_section_regions`, `cue_label`, `pdf_to_blocks`).
- **`LLMGapFilter` behavior** in `gap_llm_filter.py` (batching, context wrapping, parsing).
- **Stage C model/prompt** when a script passes it as an argument or uses the repo *default* prompt
  (e.g. `cost_perstage.py`, `cost_fullpaper.py`, `run_extraction_pipeline.py`). Reruns re-call the API.

## PINNED — a tweak will NOT appear on rerun unless you invalidate it

| you change… | stale artifact | why | fix |
|---|---|---|---|
| **model head** (retrain bge+logreg) | `data/pipeline_head.joblib` | loaded if present (`if cache.exists(): load`) | `rm data/pipeline_head.joblib` (or pass a new `--head` path) → retrains from `data/scibert_prep/train.jsonl` |
| **frozen detector** for **SAL** experiments | `stage_c_package/sal_criteria/sal_test_allmodels.json` | holds precomputed per-sentence predictions | **FIXED:** rerun `python scripts/bench/dump_sal_preds.py --inplace` → regenerates the frozen-encoder columns (reproduces them exactly). *Fine-tuned* columns (PubMedBERT-FT / SciBERT-FT / DistilBERT-FT) still need their GPU checkpoints and are preserved |
| **a Stage C prompt** | — | — | **FIXED:** all 5 experiment scripts now import from `src/gap2idea/pipeline/stage_c_prompts.py` (`SAL_SYS`/`SAL_SHOTS`, `GAP_SYS`/`GAP_SHOTS`). Edit once there → propagates to every experiment. (Shipped funnel prompts remain in `gap_llm_filter.py`.) |
| **a Stage C verdict** where a run cached it | any `*_verdicts.json` a script explicitly loads | avoids re-calling the API | delete the cache file before rerun |

## Practical checklist before a "did my change take effect?" rerun
1. Changed the **head**? → `rm data/pipeline_head.joblib`.
2. Changed the **SAL detector** and want SAL numbers to move? → regenerate `sal_test_allmodels.json`
   (TODO: add a `scripts/bench/dump_sal_preds.py`); otherwise expect no change on SAL.
3. Changed a **prompt**? → confirm whether the target script inlines it (list above) or uses the repo
   default; edit the right place.
4. Changed a **Stage C model**? → it's a live API arg; reruns recompute (delete any verdict cache).
5. Result files (`*_results.json`, `*.csv`, `gaps.tsv`) are **overwritten** on rerun — safe to re-run.

## Hardening — DONE (Oct 2026)
- ✅ `scripts/bench/dump_sal_preds.py` regenerates the frozen-encoder SAL predictions (reproduces the
  saved columns exactly; FT columns preserved — they need GPU checkpoints).
- ✅ Stage C experiment prompts centralized in `src/gap2idea/pipeline/stage_c_prompts.py`; all 5 scripts
  import them, so a prompt tweak propagates on the next rerun.

Remaining pin: the *fine-tuned* SAL detectors (PubMedBERT-FT etc.) still need their checkpoints + GPU to
regenerate — the headline SAL detector numbers stay fixed to the saved predictions until then.
