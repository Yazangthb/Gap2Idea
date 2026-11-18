# V100 baseline run status

18 November 2026, 03:12 Europe/Moscow. Inference is still in progress.

- Runner source: `84ad950`; 29 runner tests pass locally and on the V100 server.
- Small-model freeze: `ee71c86` (36 configurations, 7,416 calls).
- Qwen3.5 27B freeze: `5bf4d60` (12 configurations, 951 calls).
- Historical Qwen3.8 control freeze: `16011ed` (12 configurations, 951 calls).
- All plans, identities, conversion/checkpoint revisions, hashes, settings and commands are pushed
  to `submission-make` before their test inference. The main same-generation matrix has 48 configurations;
  the additional historical-generation controls bring the campaign to 60 configurations and 9,318 calls.

Qwen3.5 4B and 9B each passed all 18 development configurations. Their full test matrices are running
on dedicated V100s. Both completed SAL and FWS whole-paper runs in all three prompt modes.
They are now evaluating arXiv whole papers. At the latest checkpoint, 4B had one output-limit failure
(after the two prescribed retries), with all attempt costs retained; 9B had no failed calls.
No settings were changed in response to that failure.

Both 27B checkpoints are imported. Their development and test phases are queued, in order, after
both small-model jobs release the GPUs. The queue checks full GPU residency after development smoke
and before test inference. A tokenizer check found a maximum 72,285-token gold input, or 76,381 tokens
with the fixed output budget; the 27B context cap is 81,920. Small models use 131,072.

Completed evidence is being committed under
[`baseline_revision_v100_2026-11-18/`](baseline_revision_v100_2026-11-18/README.md).
Each configuration includes predictions, exact prompts, immutable manifests, coverage, failed/retried
attempts, known/missing token counts and timings. Raw evidence is never overwritten; snapshot hashes
are verified against the Git objects. Fresh server evidence is under `~/gap2idea-baselines/runs/`.

The new V100/Ollama configuration is separated from historical A100/vLLM results. The original
Qwen3.8 FP8 checkpoint is dequantized to F16 for V100, explicitly not a bitwise FP8 runtime reproduction.
The three Qwen3.5 sizes are the same-generation comparison. Final accuracy is reported after the
prespecified runs finish. P2 funnel integration uses identity-matched exports and funnel-minus-baseline
paired differences; no older PubMedBERT SAL funnel or different Qwen generation is substituted.
