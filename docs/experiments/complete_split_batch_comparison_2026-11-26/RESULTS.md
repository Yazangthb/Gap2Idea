# Complete development and test split comparison by batch size

All cells cover the entire original split: 34 development papers and 34 test papers, each with 28 arXiv and 6 biomedical papers. Values are sentence-level precision / recall / F1 (percent). Recall denominators are the 699 development and 664 test LLM-labeled gaps across all original sentences, including losses before Qwen. These are previously studied data with labels not fully human-validated; development has been used for tuning, and test results were examined in earlier experiments. These single-run results are exploratory.

Each column uses matched groups across its five methods. **1** uses 34 one-paper requests per split and method. **Up to 5** reuses the prior eight groups per split, separated by domain, with final groups of three or one papers. **Up to 17** uses four groups of **17, 10, 5, and 2** papers per split, splitting long texts to stay within the context and 65,536-output-token budget; papers are sorted by text length and then ID. All methods in each grouping see the same paper IDs and order, although full-paper input includes all sentences whereas NLI inputs use shortlisted targets and their context. Group composition, ordering and prompt paper-count wording differ across the three grouping conditions, so this does not identify the isolated causal effect of batch size.

## Test — 34 papers

| Method | 1 paper | Up to 5 papers | Up to 17 papers |
|---|---|---|---|
| Full-paper Qwen | 79.32 / 77.41 / 78.35 | 84.66 / 68.98 / 76.02 | 85.36 / 71.99 / 78.10 |
| Normal NLI → Qwen | 77.71 / 76.66 / 77.18 | 72.38 / 75.00 / 73.67 | 76.78 / 76.20 / 76.49 |
| Fine-tuned NLI, current context | 69.54 / 81.48 / 75.03 | 73.57 / 75.45 / 74.50 | 72.92 / 79.07 / 75.87 |
| Fine-tuned NLI, surrounding context | 72.16 / 80.42 / 76.07 | 74.67 / 76.81 / 75.72 | 79.32 / 76.81 / 78.04 |
| Fine-tuned NLI, no context | 70.38 / 78.01 / 74.00 | 69.89 / 76.20 / 72.91 | 72.40 / 78.61 / 75.38 |

## Development — 34 papers

| Method | 1 paper | Up to 5 papers | Up to 17 papers |
|---|---|---|---|
| Full-paper Qwen | 84.87 / 72.25 / 78.05 | 92.87 / 57.80 / 71.25 | 91.54 / 61.95 / 73.89 |
| Normal NLI → Qwen | 83.76 / 70.10 / 76.32 | 81.30 / 69.67 / 75.04 | 82.81 / 67.53 / 74.39 |
| Fine-tuned NLI, current context | 79.91 / 73.96 / 76.82 | 83.93 / 73.25 / 78.23 | 84.21 / 70.96 / 77.02 |
| Fine-tuned NLI, surrounding context | 79.91 / 72.82 / 76.20 | 84.40 / 71.96 / 77.68 | 75.64 / 76.39 / 76.01 |
| Fine-tuned NLI, no context | 76.66 / 72.82 / 74.69 | 85.06 / 65.95 / 74.29 | 80.73 / 69.53 / 74.71 |

Normal NLI uses surrounding context in the new one and up-to-17 calls, while its earlier up-to-five results used the target-then-context layout. The fine-tuned current, surrounding and no-context conditions retain their named layouts across groupings. All use Qwen3.8-27B FP8 and a neutral decision rubric; new output format requires one boolean per target. The prompt states the actual number of papers; older up-to-five calls use their original fixed five-paper wording even for smaller groups.

On the complete test set, up-to-17 fine-tuned NLI with surrounding context uses 259,776 Qwen input tokens versus full-paper's 612,965 (57.6% fewer), with 79.32% versus 85.36% precision and 76.81% versus 71.99% recall. Full paper has F1 78.10 versus 78.04 for the funnel. Qwen input-token counts exclude NLI, preprocessing and output tokens; results do not establish statistical significance.

## Output validation and provenance

Every accepted result has one unambiguous decision for every actual input target. New jobs were frozen before inference; per-call responses and model token usage are in this directory, and the 17-paper group's full/normal-NLI/fine-surrounding responses are reused byte-for-byte from the prior experiment. The up-to-five results are from `five_paper_68_verdicts_2026-11-26` and the context-layout runs; the development full-paper up-to-five counts use the previously documented recovery of two extra invalid IDs.

One development one-paper full response generated 548 additional out-of-input IDs, all marked false, and another generated one additional false ID. The predeclared policy discarded those IDs only after confirming each real input ID appeared exactly once; the original responses and excluded entries are retained. One test one-paper fine-no-context response instead repeated a real ID with conflicting decisions. Its original two failed attempts are preserved; after two additional retry responses were unfortunately not saved, a later retry used an explicit uniqueness reminder and gave an identical duplicate, which was deduplicated after verifying complete, unambiguous coverage. That response is recorded as recovered with the changed instruction. The two unpersisted retry calls are not included in reported token totals, so the affected one-paper no-context token accounting is slightly incomplete. Do not treat the recovered run as strictly identical in prompt or decoding history.

`plan.json`, `preflight.json`, `summary.json`, the 380 run records, prior group results, and scripts provide exact paper IDs, input hashes, model settings, failures, predictions and token usage. The prior 68-paper single-request attempt was stopped at the user's request and is not part of these results.
