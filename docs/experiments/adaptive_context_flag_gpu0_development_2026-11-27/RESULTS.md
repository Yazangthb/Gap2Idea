# Context-needed flag: 10-paper GPU 0 development pilot

On the same 10 previously studied arXiv development papers and 682 fine-tuned NLI candidates, Qwen3.8-27B-FP8 on the existing GPU 0 vLLM service returned a provisional `keep` and a separate `needs_more_context` boolean per target. The flag prompt asked for context when surrounding text might change the decision. A flagged candidate would receive up to three original sentences on either side in a second Qwen request; that second decision would replace its provisional keep/drop decision. We tried both target-only and one-neighbor-per-side first-pass inputs, with corresponding one-pass baselines on the same paper/candidate groups. One long paper is split into two candidate groups; all other papers are one group each, for 11 first-pass groups per layout. The denominator for recall is all 285 LLM-labeled gaps in these papers.

| First-pass input | Method | Kept | Precision | Recall | F1 | Qwen input tokens | Qwen output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Target only | One-pass baseline | 259 | 81.08% | 73.68% | 77.21% | 32,527 | 9,728 |
| Target only | Context-needed flag | 299 | 77.93% | 81.75% | 79.79% | 33,462 | 15,128 |
| One neighbor each side | One-pass baseline | 272 | 83.09% | 79.30% | 81.15% | 90,452 | 9,614 |
| One neighbor each side | Context-needed flag | 282 | 81.56% | 80.70% | 81.13% | 95,590 | 15,646 |

**Qwen flagged zero of 682 real candidates in both layouts. No selective follow-up call happened.** Therefore the flag rows are simply different first-pass prompts and schemas, not evidence that adaptive context improved accuracy. The target-only flag prompt used fewer input tokens than the one-pass neighbor baseline, but with different classification instructions and a small, previously studied sample.

A separate diagnostic with four invented sentences did elicit one true context-needed flag for "This could affect the results." It shows the format can express a true flag for at least one example, but does not establish reliable uncertainty detection or prove why the real candidates had no flags. That one diagnostic call was outside the table and its token usage was not recorded.

One surrounding-context flag response omitted a target ID twice; the original partial outputs were preserved and the missing target was queried separately with the same immediate context and system prompt. This changes one decision's batch context and adds a call. The affected flag token total includes the failed calls and targeted recovery. Previously recorded GPU 0 single-paper baselines were reused only after input and prompt hashes matched; both matching baseline calls for the split tenth paper were newly run. The first two flag papers reuse the previously frozen probe. All 10 papers and labels were previously exposed during development, and no independent test, human adjudication or performance guarantee is claimed. No GPU 1 calls were made for this study.

`plan.json`, `preflight.json`, `summary.json`, the 66 per-group records, the preserved failed response and `scripts/bench/adaptive_context_flag_gpu0_development.py` / `adaptive_context_flag_gpu0_recover.py` contain the exact prompts, input hashes, model settings, outputs and token counts. The preceding tri-state prompt also produced zero unsure decisions; see `../adaptive_context_gpu1_pilot_2026-11-27_rev1/RESULTS.md` for its separate GPU 1 result and user clarification.
