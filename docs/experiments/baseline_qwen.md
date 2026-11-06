# Task 5a: Qwen without the funnel

**Status: frozen** with the commit that adds this file, before the first test call. `scripts/bench/baseline_qwen.py run --split test` refuses to start while this file has uncommitted
changes. Any change made after a test result is seen goes under "Changes after test", with the reason.

## Goal

The funnel narrows each paper to candidate sentences (Stage A section slice, Stage B classifier) and then
asks Qwen to filter them (Stage C). This baseline gives the same Qwen every sentence, with no funnel, on the
same test papers, so the paper can compare accuracy and cost fairly.

## Model and server

- **Model:** `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, served as
  `qwen3.8-27b`. Same weights, revision and vLLM settings as the server that produced the funnel's verdicts.
- **Server:** our own vLLM on GPU 1 of `a100-learner` (10.100.30.86), port 8002, so runs don't share a GPU with
  other jobs. Flags copied from the funnel's server on GPU 0, port 8001, except the tool-calling flags, which
  these calls don't use:
  ```
  CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0 \
  vllm serve Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
    --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8002 --max-model-len 262144 --max-num-seqs 64 \
    --reasoning-parser qwen3 --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --api-key "$QWEN_API_KEY"
  ```
- **Every call:** `temperature=0`, `seed=0`, `chat_template_kwargs={"enable_thinking": false}`, JSON-schema
  output. The script reads `QWEN_BASE_URL` and `QWEN_API_KEY` from the environment or `.env`. Never commit the key.
- **One call at a time.** Run time is a reported metric, and the funnel's 2.7 s per batch was measured
  one call at a time.

## Data

All counts checked by `baseline_qwen.py check` against the task sheet.

| Dataset | Labels | Test set | Positives | Source | Tuning / smoke tests |
|---|---|---|---|---|---|
| SAL-40 | human, limitation | 40 papers, 7,968 sentences | 205 | `data/splits/sal_test.jsonl` | `sal_dev.jsonl`: 40 papers, 7,729 sentences |
| arXiv gold | LLM; limitation, future work, research gap | 28 papers, 10,388 sentences | 527 | `load_val("test")` in `nli_variants.py`, `dom == "arxiv_cs"` | `splits.json["val"]`, arXiv only: 28 papers, 10,087 sentences |
| FWS | human, future work | 209 papers, 1,506 sentences | 220 | `data/splits/fws_test.jsonl` | `fws_dev.jsonl`: 418 papers, 3,005 sentences |

**FWS text is lowercased** (`data/splits/README.md`). In the released corpus every positive is all
lowercase and almost no negative is, so raw text leaks the label. The split files are already lowercased,
the script checks this, and the funnel's FWS run used the same lowercased text.

Every sentence is sent, including references, captions and headings. Removing them is part of the funnel
(Stage A), so the baseline doesn't.

## Setups

Ids are numbered from 0 within each call and mapped back to (paper, sentence index).

| Setup | Input per call | Output limit | SAL-40 calls | arXiv calls | FWS calls |
|---|---|---|---|---|---|
| **A** whole paper | every sentence of one paper (FWS: one conclusion excerpt) | 4,096 tokens | 40 | 28 | 209 |
| **B** chunks | 100 consecutive sentences of one paper, no overlap | 2,048 tokens | 101 | 117 | not run |
| **C** batches | the next 20 sentences in reading order, across paper boundaries, as the funnel's Stage C batches | 1,024 tokens | 399 | 520 | 76 |

The chunk size for B is fixed at 100 before any run, not tuned. The longest single call is 63,359 input tokens
(arXiv, setup A), well inside the 262,144-token context.

## Prompts

The prompts are the funnel's Stage C prompts with **only the input and output wording changed**. The
decision rules and the strictness clause are reused verbatim, so the only difference from the funnel is the
funnel itself. The script reads them straight from Yazan's scripts, so any later edit to those scripts makes
the build fail (`swap()` exits when a phrase it replaces is missing).

**Prompt versions.** `RUBRICS` in `scripts/bench/stage_c_rubric_val.py`:

| Name | Funnel mode | Final instruction |
|---|---|---|
| `keep_biased` | high recall | when uncertain, answer YES |
| `neutral` | balanced | favour neither YES nor NO |
| `strict` | high precision | answer YES only for an explicit statement; when uncertain, NO |

**Per dataset:**

- **SAL-40:** each rubric narrowed by `limitation_scope()` (`stage_c_sal40_modes.py`), because SAL labels only limitations.
- **arXiv gold:** each rubric unchanged (limitation, future work, research gap).
- **FWS:** `validate_fws` (`SYSTEM_VALIDATE_FWS` plus its 14 calibration examples, as `LLMGapFilter` builds
  them), as the task instructions say. See note 1 below.

**Wording changed** (`HEAD`, `ITEMS`, `OUTPUT` and `FWS_IO` in the script):

| Part | Funnel | Baseline |
|---|---|---|
| Opening | "You are a precision filter for a research-gap extraction pipeline. Judge each candidate sentence:" | "You are reading sentences from scientific papers. Judge each sentence:" |
| Input | items with `i`, `type`, `sentence`, `context` | "consecutive sentences from scientific papers, in reading order, one per line as `[id] sentence`", neighbours used only to disambiguate |
| Output | `{"results":[{"id","keep"}]}`, one per item | `{"positives":[ids]}`, only the YES ids; `[]` when none |

The full text of every prompt is in `docs/experiments/baseline_qwen/prompts/`, written by `check`.

## Runs

| Dataset | Setups | Prompts | Runs |
|---|---|---|---|
| SAL-40 | A, B, C | keep_biased, neutral, strict | 9 |
| arXiv gold | A, B, C | keep_biased, neutral, strict | 9 |
| FWS | A, C | validate_fws | 2 |
| **Total** | | | **20 runs, 3,900 calls, about 8.7M input tokens** |

Estimated time one call at a time: about 3 to 3.5 hours.

**Order:**
1. Smoke test on 2 dev papers per dataset and setup (`--split dev --limit-papers 2`). Checks only that calls,
   JSON and ids work. Dev results don't change any setting above. If they show a bug, the fix goes in this
   file before it's committed. Done 2026-11-06 on the GPU 1 server: 167 calls, 0 failed, 0 invalid ids,
   0.4 to 2.4 s per call. Verdicts in `docs/experiments/baseline_qwen/*_dev_*.jsonl`.
2. Commit this file.
3. Test runs, one job at a time, each run once.

**Failures:** a call is retried twice. If all three attempts fail (error, unparseable JSON, or output hitting
the token limit), it's saved as `failed` and never counted as an empty answer. Ids outside the call's range
are dropped and counted as `invalid_ids`. Runs resume from their JSONL file.

## Scoring

- **Per sentence**, against each dataset's labels: SAL limitation, FWS future work, arXiv any of the three gap
  types. Every test sentence is in the denominator.
- **P, R and F1 with 95% CIs** from resampling papers, 2,000 draws, `random.Random(0)`.
- **F1 difference from the funnel** with a paired paper bootstrap (same method as `paired()` in
  `stage_c_nli_qwen.py`).
- Also reported: input and output tokens, run time, failed calls (and their sentences), empty answers,
  invalid ids. For arXiv, recall per gap type.
- Sentences in failed calls count as not kept, and the failed count is shown next to every number.

## Comparison with the funnel

| Dataset | Funnel | P / R / F1 | Saved verdicts |
|---|---|---|---|
| SAL-40 | PubMedBERT + Qwen, `rct_loose` prompt | 0.547 / 0.873 / 0.673 | `stage_c_sal40_qwen.json` |
| arXiv gold | NLI (τ 0.354) + Qwen, `keep_biased` | 0.513 / 0.854 / 0.641 | `stage_c_nli_r90_verdicts.jsonl` |
| arXiv gold | bge-small + Qwen | 0.601 / 0.687 / 0.641 | `stage_c_full_qwen.json` (summary only, no per-sentence verdicts) |
| FWS | Task 3 Qwen rerun | pending | Task 3 output |

`check` rescores the first two from their saved verdicts and reproduces the numbers exactly.

## Notes and open items

Items 2 and 3 affect only the comparison column, which is recomputed from saved verdicts without new Qwen calls.

1. **FWS prompt: decided, `validate_fws` only**, as the task instructions say. Note for reading the results: the
   paper's current FWS funnel result (NLI + Qwen, high precision F1 0.760, `nli_sal_fws.py`) used the three
   rubrics narrowed by `future_scope()`, so against that funnel the FWS baseline differs in prompt as well as in
   the funnel. The `future_scope` runs are not part of this frozen plan; adding them later would be reported
   under "Changes after test".
2. **Per-mode funnel verdicts.** Matching each baseline prompt with the funnel run on the same prompt needs the
   funnel's kept sentences, as (paper_id, idx), for: arXiv `neutral` and `strict`; SAL-40 NLI in all three modes;
   FWS in all three modes. Only arXiv `keep_biased` and SAL `rct_loose` are in the repo.
3. **bge-small + Qwen** has no per-sentence verdicts in the repo, so no paired CI against it.
4. **Server setup.** Our own environment on `a100-learner`: a venv with `vllm==0.32.0` (the version the funnel's
   server reports at `/version`) and the same model revision downloaded from Hugging Face (about 28 GB; 450 GB
   free). The host reaches PyPI and Hugging Face, and the revision exists there.

## Results

Filled in by `baseline_qwen.py score --split test --write-plan` after the runs.

<!-- results:start -->
| Dataset | Setup | Prompt | Precision | Recall | F1 | F1 minus funnel | Input tokens | Run time | Failed calls | Empty answers |
|---|---|---|---|---|---|---|---|---|---|---|
| SAL-40 | A | keep_biased | 54.2% (46.7–62.5) | 85.4% (80.0–90.1) | 0.663 (0.602–0.724) | -0.010 (-0.062 to +0.040) vs `pubmedbert+qwen_rct_loose` | 346,445 | 1.1 min | 0 | 0 |
| SAL-40 | A | neutral | 56.0% (49.2–63.3) | 84.4% (78.3–89.4) | 0.673 (0.616–0.731) | +0.000 (-0.051 to +0.053) vs `pubmedbert+qwen_rct_loose` | 345,525 | 1.1 min | 0 | 0 |
| SAL-40 | A | strict | 64.1% (56.1–72.7) | 77.6% (71.3–83.4) | 0.702 (0.644–0.759) | +0.029 (-0.031 to +0.090) vs `pubmedbert+qwen_rct_loose` | 347,685 | 1.1 min | 0 | 0 |
| SAL-40 | B | keep_biased | 45.4% (38.7–53.0) | 89.8% (84.3–94.6) | 0.603 (0.536–0.669) | -0.070 (-0.130 to -0.012) vs `pubmedbert+qwen_rct_loose` | 363,130 | 1.5 min | 0 | 32 |
| SAL-40 | B | neutral | 47.9% (40.0–56.4) | 88.8% (83.4–93.5) | 0.622 (0.548–0.696) | -0.051 (-0.116 to +0.015) vs `pubmedbert+qwen_rct_loose` | 360,807 | 1.5 min | 0 | 34 |
| SAL-40 | B | strict | 56.9% (48.2–66.3) | 84.9% (79.2–89.8) | 0.681 (0.610–0.749) | +0.008 (-0.050 to +0.073) vs `pubmedbert+qwen_rct_loose` | 366,261 | 1.4 min | 0 | 39 |
| SAL-40 | C | keep_biased | 39.3% (34.4–44.6) | 93.7% (89.5–97.2) | 0.554 (0.502–0.607) | -0.119 (-0.157 to -0.080) vs `pubmedbert+qwen_rct_loose` | 463,811 | 3.2 min | 0 | 257 |
| SAL-40 | C | neutral | 45.4% (39.5–51.9) | 90.7% (86.5–94.8) | 0.605 (0.547–0.662) | -0.068 (-0.118 to -0.020) vs `pubmedbert+qwen_rct_loose` | 454,634 | 3.1 min | 0 | 264 |
| SAL-40 | C | strict | 50.6% (43.5–58.6) | 87.3% (82.2–91.6) | 0.640 (0.575–0.706) | -0.033 (-0.084 to +0.020) vs `pubmedbert+qwen_rct_loose` | 476,180 | 3.1 min | 0 | 284 |
| arXiv gold | A | keep_biased | 74.6% (63.8–85.6) | 73.6% (68.0–78.1) | 0.741 (0.678–0.795) | +0.101 (+0.027 to +0.172) vs `nli_r90+qwen_keep_biased` | 552,686 | 1.9 min | 0 | 0 |
| arXiv gold | A | neutral | 83.1% (76.8–90.6) | 68.1% (59.8–75.3) | 0.749 (0.696–0.790) | +0.108 (+0.025 to +0.173) vs `nli_r90+qwen_keep_biased` | 552,042 | 1.8 min | 0 | 0 |
| arXiv gold | A | strict | 89.2% (84.7–95.1) | 54.8% (47.0–61.5) | 0.679 (0.620–0.724) | +0.039 (-0.058 to +0.114) vs `nli_r90+qwen_keep_biased` | 553,834 | 1.7 min | 0 | 0 |
| arXiv gold | B | keep_biased | 69.5% (61.6–79.6) | 76.1% (70.0–81.1) | 0.726 (0.673–0.774) | +0.086 (+0.028 to +0.137) vs `nli_r90+qwen_keep_biased` | 579,949 | 2.2 min | 0 | 31 |
| arXiv gold | B | neutral | 71.6% (63.9–81.0) | 72.3% (66.7–76.7) | 0.720 (0.672–0.765) | +0.079 (+0.027 to +0.129) vs `nli_r90+qwen_keep_biased` | 577,258 | 2.2 min | 0 | 33 |
| arXiv gold | B | strict | 84.4% (75.4–93.1) | 59.6% (54.5–64.8) | 0.699 (0.646–0.745) | +0.058 (-0.013 to +0.116) vs `nli_r90+qwen_keep_biased` | 584,746 | 2.0 min | 0 | 42 |
| arXiv gold | C | keep_biased | 61.6% (52.5–71.4) | 81.8% (77.1–85.2) | 0.703 (0.636–0.768) | +0.062 (+0.009 to +0.109) vs `nli_r90+qwen_keep_biased` | 740,338 | 4.6 min | 0 | 310 |
| arXiv gold | C | neutral | 68.2% (60.6–76.4) | 76.9% (72.0–81.0) | 0.723 (0.672–0.773) | +0.082 (+0.026 to +0.136) vs `nli_r90+qwen_keep_biased` | 728,378 | 4.6 min | 0 | 320 |
| arXiv gold | C | strict | 81.5% (73.4–89.4) | 64.5% (58.8–69.5) | 0.720 (0.668–0.762) | +0.080 (+0.001 to +0.145) vs `nli_r90+qwen_keep_biased` | 761,658 | 4.5 min | 0 | 371 |
| FWS | A | validate_fws | 57.7% (50.7–65.0) | 87.3% (80.5–92.7) | 0.694 (0.632–0.751) | pending | 183,250 | 1.5 min | 0 | 79 |
| FWS | C | validate_fws | 60.1% (53.6–66.8) | 88.2% (82.0–93.2) | 0.715 (0.659–0.766) | pending | 100,540 | 0.8 min | 0 | 6 |

Funnel, rescored from its saved verdicts:
- SAL-40, `pubmedbert+qwen_rct_loose`: P 0.547, R 0.873, F1 0.673
- arXiv gold, `nli_r90+qwen_keep_biased`: P 0.513, R 0.854, F1 0.641

arXiv gold, recall by gap type:

| Setup | Prompt | Limitation | Future work | Research gap |
|---|---|---|---|---|
| A | keep_biased | 59.9% | 84.9% | 77.8% |
| A | neutral | 56.6% | 89.0% | 68.9% |
| A | strict | 46.1% | 87.7% | 51.3% |
| B | keep_biased | 61.2% | 94.5% | 79.1% |
| B | neutral | 59.9% | 90.4% | 74.2% |
| B | strict | 46.1% | 89.0% | 59.3% |
| C | keep_biased | 61.2% | 95.9% | 88.7% |
| C | neutral | 55.9% | 93.2% | 83.4% |
| C | strict | 43.4% | 89.0% | 69.2% |
<!-- results:end -->

## Changes after test

- **Scoring display only:** the recall-by-gap-type table also listed the two FWS runs (all zeros except future
  work). It now lists arXiv runs only. No setting, call or number changed.

Test runs: 2026-11-06, 16:30 to 17:15, on the GPU 1 server (vLLM 0.32.0), one call at a time. 3,900 calls,
0 failed.
