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
<!-- results:end -->

## Changes after test

None yet.
