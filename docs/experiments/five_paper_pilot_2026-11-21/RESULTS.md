# Five-paper pilot results

Exploratory development-set pilot, not an independent test. Five preselected arXiv papers contain 1,713 sentences and 105 positive labels; labels are not human-validated. The frozen plan records the selection, prompts, model settings, source hashes, and evaluation protocol.

| Method | TP | FP | Labeled positives | Precision | Recall | F1 | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.8-27B-FP8, five whole papers in one call | 56 | 3 | 105 | 0.9492 | 0.5333 | 0.6829 | Completed |
| Qwen3.8-27B-FP8, saved strict funnel | 43 | 3 | 105 | 0.9348 | 0.4095 | 0.5695 | Completed previously |
| Qwen3.5-4B BF16, five whole papers in one call | — | — | 105 | — | — | — | Failed twice |
| Qwen3.5-4B BF16, strict funnel | 73 | 32 | 105 | 0.6952 | 0.6952 | 0.6952 | Completed |

The 4B five-paper call listed consecutive IDs, reaching the 8,192-token output limit on both deterministic attempts before returning valid JSON; this is a failure, not a zero score. Each attempt consumed 96,300 prompt tokens. The 4B funnel processed 304 preselected candidates in 16 successful calls, using 56,641 prompt tokens and 3,010 completion tokens. The funnel token count and time exclude earlier funnel stages, so they are not a complete cost comparison.

The 27B reference comes from the original full-validation funnel run, where batches were formed across the validation set; the new 4B funnel batches only the selected papers. All scores use the same selected sentences and labels, but batch context and runtime differ. The 27B direct call used the existing H100 service; 4B ran on a V100 using Ollama with thinking disabled and a 131,072-token context. The loaded 4B model was `qwen3.5-4b-bf16:latest`, BF16 GGUF, Ollama model digest `23f8c64d937d3062759377d28559235f278fe675fa782da87699e583ddfb40e0`; the precise upstream Hugging Face revision claimed in the plan was not independently verified for this converted GGUF. The adapter and raw call records are included for reproduction.

These five papers provide a feasibility signal only. A full evaluation needs a locked protocol, broader independent data, consistent batching/runtime disclosure, and an explicit policy for invalid outputs; the failed 4B call should not be silently discarded.

## 4B follow-up: prompt retry and one-paper calls

On November 22, a retry of the five-paper call added a reminder and a small invented example after the five papers. It again listed consecutive IDs and reached the 8,192-token output limit, so it did not yield a score. A separate short 40-sentence diagnostic returned valid IDs, indicating that JSON output alone is not sufficient to explain the failure; the specific cause remains unproven.

The fallback sent each complete paper directly to the same 4B model in a separate call, using the original rubric and no funnel. All five calls returned valid outputs: TP 61, FP 20, labeled positives 105, precision 0.7531, recall 0.5810 and F1 0.6559. The 4B funnel's corresponding precision, recall and F1 were 0.6952, 0.6952 and 0.6952. The fallback used 97,094 prompt tokens and 395 completion tokens across five calls; the longest single input used 48,261 prompt tokens. The total call time was 75.91 seconds, while the previously reported funnel call time excludes its earlier stages. This is a different batching strategy from the original five-paper call and was selected after observing failure, so it is exploratory rather than a preregistered like-for-like result. Raw responses and the retry scripts are included.

## Existing neutral and keep-biased prompts on the same five papers

All settings use the same 304 Stage B candidates and the same 105 labeled positives for scoring. The 27B results below rescore the original saved full-validation verdicts; these are not new 27B inference runs. All three 27B prompts used the original full-validation batching. The new 4B calls batch only these five papers, as did the previous 4B strict run. Accordingly, compare prompts within each model; cross-model comparison also changes batching and runtime.

| Stage C model | Prompt | TP | FP | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 27B | Strict | 43 | 3 | 0.9348 | 0.4095 | 0.5695 |
| 27B | Neutral | 79 | 11 | 0.8778 | 0.7524 | 0.8103 |
| 27B | Keep-biased | 87 | 18 | 0.8286 | 0.8286 | 0.8286 |
| 4B | Strict | 73 | 32 | 0.6952 | 0.6952 | 0.6952 |
| 4B | Neutral | 72 | 39 | 0.6486 | 0.6857 | 0.6667 |
| 4B | Keep-biased | 79 | 63 | 0.5563 | 0.7524 | 0.6397 |

Each saved 27B validation verdict file contains all 2,278 unique candidate indices; the saved strict file matches the existing frozen reference. The neutral and keep-biased 4B runs each completed 16 batches with no missing verdicts, using the same BF16 GGUF model and runtime as the original 4B pilot. The prompts were established before this comparison, but some wording was informed by earlier test false positives. The five papers and labels remain development data with LLM-generated, non-human-validated labels; the choice of best prompt here is exploratory and requires confirmation on independent data.
