# Adaptive context pilot: GPU 1, 10 development papers

This is a small exploratory pilot on 10 previously studied arXiv development papers, with 682 candidates from the frozen fine-tuned NLI stage and 285 LLM-labeled positive sentences overall. Each request covers targets from one paper; one paper was split into two candidate groups to fit the existing GPU 1 service's 49,152-token context window, giving 11 groups per condition. The one-pass baselines and first-pass uncertainty variants ran on the same Qwen3.8-27B FP8 service (vLLM 0.34.0), with the same groups, candidates, prompts apart from the uncertainty output instructions, and context layouts. No test papers or new human ratings were used.

The first-pass uncertainty prompt offered `keep`, `drop`, and `unsure`, explicitly instructing the model to choose unsure if surrounding original text could change the judgment. A second call would have judged only unsure targets using up to three original neighboring sentences on each side. **Qwen returned zero unsure decisions among all 682 candidates in both layouts. No second-pass context call was made.** The rows labeled uncertainty prompt below therefore measure a changed first-pass prompt, not an effective adaptive-context pipeline.

| First-pass input | Method | Qwen output kept | Precision | Recall | F1 | Qwen input tokens | Qwen output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Target only | One-pass baseline | 272 | 80.51% | 76.84% | 78.64% | 32,527 | 9,712 |
| Target only | Uncertainty prompt | 244 | 84.02% | 71.93% | 77.50% | 33,561 | 10,464 |
| One neighbor each side | One-pass baseline | 264 | 84.85% | 78.60% | 81.60% | 90,452 | 10,148 |
| One neighbor each side | Uncertainty prompt | 228 | 94.30% | 75.44% | 83.82% | 95,707 | 10,721 |

The apparent precision changes are prompt effects on the same small development sample; they are not evidence that targeted follow-up works. There was no token saving from uncertainty routing, because there were no unsure decisions and hence no selective follow-up. The target-only uncertainty prompt uses fewer input tokens than the surrounding-context baseline, but also has lower recall and a different prompt. These are exploratory results, not a reliable comparison on a fresh or human-labeled test set.

One surrounding-context first-pass response omitted a target twice; the incomplete responses were preserved, and the missing target was queried separately with its original immediate neighbors. This recovery changes the batch context for one decision and adds a call; the reported adaptive surrounding token and call totals include both failed attempts and that targeted call. Other calls passed input ID checks. `summary.json`, `plan.json`, `preflight.json`, per-paper records and the runners `scripts/bench/adaptive_context_gpu1_pilot.py` and `scripts/bench/adaptive_context_gpu1_recover_missing.py` preserve the metrics, prompts, failures, and provenance.

The initial uncertainty wording was also attempted on GPU 0 before switching: the first three original development groups (15 papers, 900 candidates) produced zero unsure decisions in both layouts. That run was interrupted at the user's request and is incomplete; its preserved output is in the separate `adaptive_context_development_2026-11-27` directory and is not included in the table.

The user clarified that the GPU 1 service belongs to another user and should be left alone. Our calls used that already-running service but did not stop, restart or modify it. Our SSH tunnel was subsequently closed; no further calls to that service were made after the clarification. A separate boolean-flag probe was prepared locally but never run.
