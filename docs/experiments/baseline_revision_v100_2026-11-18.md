# Baseline revision on V100 (18 November 2026)

Implements Hamza's P1/X1/X2 from REVISION_PLAN_2026-11-16.md. No historical verdict is overwritten.

## Prespecified matrix

The old 27B model was **Qwen3.8-27B-FP8**, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, as recorded in baseline_qwen.md.
The same-generation comparison therefore requires an additional **Qwen3.5-27B** run.

For each of Qwen3.5 4B, 9B and 27B, run strict, neutral and keep_biased:

- FWS setup A, using the future-work-scoped rubrics (not validate_fws): 9 configurations.
- SAL setup A and AS (Stage A sentences only): 18 configurations.
- arXiv gold setup A: 9 configurations, including the new same-generation 27B control.
- For 4B and 9B, SAL and arXiv setup C, every sentence in batches of 20: 12 configurations.

Total: **48 configurations, 8,367 test calls**. Setup B is optional in the revision plan and excluded here.
The development smoke matrix uses the first two development papers per dataset and the same configurations.
The paper restriction is applied before batching, so a C smoke batch cannot include a third paper.

## Hardware and serving

The user designated the V100 server. It has two Tesla V100-SXM2-32GB GPUs. Upstream current vLLM
requires compute capability >=7.5; V100 is 7.0. We use a private, loopback-only Ollama 0.40.16
service (llama.cpp backend) under Hamza's account, without altering the other user's server.

This is a **new backend/precision configuration**, separate from historical A100/vLLM/FP8 or bf16 runs.
Use unquantized BF16 GGUF conversions for all three sizes, with conversion repository revisions and
SHA-256 checksums recorded. Verify GPU offload during development before freezing the test plan.
If BF16 cannot run on V100, any FP16 conversion is documented and frozen before test inference.
No quantized model is silently substituted.

All calls: temperature 0, seed 0, thinking false, JSON schema positives array, repeat_penalty 1,
131,072-token context, one call at a time. Output limits stay A/AS 4,096, B 2,048 and C 1,024.
The frozen plan records the full launch command, server version, model manifest digests, template hashes,
data file byte hashes, canonical data/input/prompt hashes, and code hashes (UTF-8 with LF normalization).

## Freeze and execute

Use a fresh output directory for any changed configuration or corrected implementation.
The commands below run from the repository root on the V100. `RUN_ROOT` is a new absolute path.

```bash
python scripts/dataset/fetch_external.py
python scripts/dataset/build_eval_files.py
python scripts/bench/baseline_qwen.py check
python -m pytest tests/test_baseline_qwen.py -q

python scripts/bench/baseline_matrix.py prepare \
  --models qwen3.5-4b-bf16:latest qwen3.5-9b-bf16:latest qwen3.5-27b-bf16:latest \
  --plan docs/experiments/baseline_revision_v100_plan.json \
  --output-root "$RUN_ROOT" --server-command "$EXACT_SERVER_COMMAND"
python scripts/bench/baseline_matrix.py run \
  --plan docs/experiments/baseline_revision_v100_plan.json --split dev

# Review the development evidence; if a fix is needed, create a new plan/output root.
git add -f docs/experiments/baseline_revision_v100_plan.json
git commit -m 'experiments: freeze V100 baseline matrix before test inference'
python scripts/bench/baseline_matrix.py run \
  --plan docs/experiments/baseline_revision_v100_plan.json --split test
python scripts/bench/baseline_report.py "$RUN_ROOT"
```

An individual smoke configuration can also be run through the repaired wrapper:

```bash
python scripts/bench/whole_paper_other_llm.py --backend ollama \
  --model qwen3.5-4b-bf16:latest --dataset sal --setup AS --prompt strict \
  --split dev --limit-papers 2 --output-dir /absolute/fresh/smoke-directory
```

## Evidence and scoring rules

Each verdict file has an immutable configuration manifest and the exact system prompt beside it.
Each call includes original sentence keys, positives, mapped kept keys, invalid ids, final status,
all attempts, raw responses, per-attempt token usage (including rejected output), UTC start and timings.
Unreported usage is null, with known subtotals and the number of missing-cost attempts reported.
Retry exactly twice after the first attempt. Failed calls are retained and not automatically rerun on resume.
Resume validates unit keys and rejects duplicate or foreign records and changed configurations.

Coverage distinguishes requested, answered, failed, unattempted and Stage A excluded sentences.
Every SAL score includes all 7,968 sentences and 205 positives, including AS exclusions as not kept.
P/R/F1 and 95% intervals use 2,000 paper bootstrap draws, seed 0. Paired differences are always
**funnel minus baseline**. A CI containing zero is described as "difference not resolved".

`baseline_report.py` exports baseline scores and accepts identity/mode/dataset/input-matched funnel
exports from the P2 scoring owner. It does not substitute the old PubMedBERT SAL funnel, mix Qwen
generations or infer that different precisions/backends are interchangeable.

Hand over to Osama: prediction JSONLs, prompts, frozen plan/commit, all manifests and summaries,
coverage, failure/attempt logs, known/missing token counts, timings and scorer commit.
Delivery channel/recipient must be identified before sending; the local evidence package is prepared first.
