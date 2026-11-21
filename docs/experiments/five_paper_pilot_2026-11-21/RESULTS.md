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
