# November baseline campaign status

18 November 2026, 04:42 Europe/Moscow. Inference is still in progress.

The campaign now runs on the V100 and H100 hosts. User-authorized H100 allocation moved the two
pending 27B cohorts to native vLLM and retained additional H100 4B/9B controls. The V100 27B plans
remain as unexecuted provenance: no calls were made under those two plans, and their old queue was
retired. V100 and H100 predictions are separate cohorts.

| Cohort | Model | Recorded test calls | Planned calls | Failed calls | State |
|---|---|---:|---:|---:|---|
| V100 / Ollama | Qwen3.5 4B BF16 | 3,708 | 3,708 | 2 | All 18 configurations recorded |
| V100 / Ollama | Qwen3.5 9B BF16 | 2,272 | 3,708 | 0 | Running; 04:41 snapshot |
| H100 / vLLM | Qwen3.5 27B BF16 | 796 | 951 | 0 | Running |
| H100 / vLLM | Qwen3.8 27B FP8 | 951 | 951 | 0 | All 12 configurations recorded |
| H100 / vLLM | Qwen3.5 4B BF16 | 0 | 3,708 | 0 | Follows Qwen3.5 27B on GPU 0 |
| H100 / vLLM | Qwen3.5 9B BF16 | 0 | 3,708 | 0 | Server starting on GPU 1 |

The V100 4B failures hit the fixed output limit after the prescribed two retries. All attempts and
token costs are retained; settings were not adjusted in response. Both V100 small models passed all
18 development configurations, and both H100 27B models passed their 12 development configurations
before test inference. H100 small controls must pass the same development gate.

The active matrix contains 96 test configurations and 16,734 calls across the two distinct backends.
Runner Python remains frozen at `84ad950`. All 29 runner tests pass locally, on V100, and on H100.
The V100 small plan was committed at `ee71c86`, and the H100 plan at `a3b7c85`, before inference.
All 120 H100 development/test prompt, input and data hashes match corresponding V100 configurations.

Completed configuration evidence is being pushed under
[`baseline_revision_v100_2026-11-18/`](baseline_revision_v100_2026-11-18/README.md) and
[`baseline_revision_h100_2026-11-18/`](baseline_revision_h100_2026-11-18/).
Each includes predictions, exact prompts, immutable manifests, coverage, failed/retried attempts,
known/missing tokens and timings. Raw evidence is not overwritten; snapshot hashes are checked
before import. Fresh server evidence remains in each campaign's `runs/` directory.

The H100 extension uses the original native Qwen3.8 FP8 checkpoint and BF16 Qwen3.5 sizes under
vLLM 0.32.0. The unexecuted V100 Qwen3.8 F16 conversion is not reported as a runtime reproduction.
Final scoring uses all evaluation sentences and 2,000 paper bootstrap draws. Paired P2 funnel
differences require identity-matched exports; older PubMedBERT SAL or other-generation predictions
are not substituted.

See [pause/resume commands](BASELINE_PAUSE_RESUME.md) for Yazan. They preserve saved calls, record a
versioned checkpoint and free only campaign model memory. The helper's operational version 2
amendment is archived separately; no scientific source or settings changed.
