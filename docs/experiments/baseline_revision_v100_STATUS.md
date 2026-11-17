# V100 baseline run status

18 November 2026, 02:37 Europe/Moscow.

- Runner and evidence code: `84ad950`; 29 runner tests pass locally and on the V100 server.
- Frozen small-model plan: `ee71c86`, including model digests, conversion revisions, prompt/data/code hashes,
  settings, commands and hardware evidence. Pushed to `submission-make` before test inference.
- Qwen3.5 4B: all 18 final development configurations passed coverage and id checks; test matrix started.
- Qwen3.5 9B: short development preflight passed, with full offload to the second V100;
  the complete 18-configuration development smoke matrix is running.
- Qwen3.5 27B: checksum verification, shard merge and import are in progress. Its 12 configurations
  will be frozen and smoke-tested before test inference; it runs after the small models release both GPUs.
- The historical Qwen3.8-27B-FP8 checkpoint at its exact original revision is being downloaded separately
  to assess a V100-compatible representation for additional matched-prompt controls. It is not replaced
  with the unrelated Q4 checkpoint already served by another user's process.

The same-generation Qwen3.5 matrix has 48 configurations and 8,367 planned test calls.
The 4B/9B subset accounts for 36 configurations and 7,416 calls. Whole-paper SAL/FWS uses all three
rubrics; SAL AS retains original ids; C covers every arXiv/SAL sentence in batches of 20.

Fresh evidence roots on the V100 are under `~/gap2idea-baselines/runs/`. On completion, prediction
JSONLs, prompts, manifests, coverage/failure/attempt logs, token counts and timings will be committed
under this repository's experiment directory for Osama. Unknown attempt costs remain null.

No final accuracy results are claimed while inference is in progress. The new V100/Ollama configuration
is explicitly separated from historical A100/vLLM results. Paired funnel integration belongs to P2;
`baseline_report.py` accepts identity-matched exports and reports funnel minus baseline.
