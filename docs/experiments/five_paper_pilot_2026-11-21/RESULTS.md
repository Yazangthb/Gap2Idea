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

## 4B-to-27B cascade

The 27B model reviewed only the sentences kept by the existing 4B strict or keep-biased Stage C run; source candidate order, 20-sentence batch maximum, context construction and rubric were retained. It saw no labels or 4B verdicts other than the shortlist itself. All four fresh 27B runs completed without failed batches. Every metric uses the original 105 labeled positives as the denominator.

| Method | Candidates for 27B | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4B strict alone | — | 73 | 32 | 0.6952 | 0.6952 | 0.6952 |
| 4B strict → 27B neutral | 105 | 70 | 9 | 0.8861 | 0.6667 | 0.7609 |
| 4B strict → 27B keep-biased | 105 | 73 | 19 | 0.7935 | 0.6952 | 0.7411 |
| 4B keep-biased → 27B neutral | 142 | 68 | 14 | 0.8293 | 0.6476 | 0.7273 |
| 4B keep-biased → 27B keep-biased | 142 | 76 | 26 | 0.7451 | 0.7238 | 0.7343 |
| 27B neutral without 4B shortlist, saved validation run | 304 | 79 | 11 | 0.8778 | 0.7524 | 0.8103 |
| 27B keep-biased without 4B shortlist, saved validation run | 304 | 87 | 18 | 0.8286 | 0.8286 | 0.8286 |

The strict-to-neutral cascade removed 23 of the 4B shortlist's 32 false positives while losing three of its 73 true positives. It used 18,281 input and 1,164 output tokens in six 27B calls, in addition to the earlier 4B processing and Stage A/B. The earlier 4B strict calls used 56,641 input and 3,010 output tokens. We cannot measure total savings against the old 27B-only run because its saved verdicts do not record token usage. The old 27B-only runs were batched across all validation papers; the new cascade batches only selected candidates from the five-paper pilot, so batch context and runtime differ. These small-sample, LLM-label results are exploratory.

## Higher-recall 4B first-pass prompts

Two new exploratory first-pass prompts (`screen`, `permissive`) tell 4B that a second model will review potential gaps and that uncertain cases should be kept. The prompts were committed before the calls; the same 304 candidates, batching, 4B model, and five-paper evaluation were used. After committing the 4B results, 27B applied the existing neutral prompt to each shortlist. The same 105 labeled positives are the denominator in all rows.

| Configuration | 4B shortlist size | TP after 4B | Final TP | Final FP | Final precision | Final recall | Final F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Previous 4B strict → 27B neutral | 105 | 73 | 70 | 9 | 0.8861 | 0.6667 | 0.7609 |
| **New 4B screen → 27B neutral** | **208** | **89** | **77** | **11** | **0.8750** | **0.7333** | **0.7979** |
| New 4B permissive → 27B neutral | 240 | 87 | 75 | 13 | 0.8523 | 0.7143 | 0.7772 |
| 27B neutral alone, saved validation run | 304 | — | 79 | 11 | 0.8778 | 0.7524 | 0.8103 |

The NLI stage reached 92 of 105 labeled positives; the screen prompt kept 89 of those 92, and 27B retained 77. The new screen setup sent 208 of 304 candidates to 27B, 31.6% fewer candidate items, but also added a 4B call stage. It consumed 52,753 input and 3,830 output tokens in 16 4B calls, then 36,131 input and 2,845 output tokens in 11 27B calls; earlier Stage A/B costs are excluded. The saved 27B-only run did not record token usage, so overall cost savings have not been established. Its batching across the full validation set differs from the new cascade's five-paper-only batching, so the small F1 difference cannot establish equivalence. The prompts were developed iteratively on this sample, which has LLM-generated labels; independent held-out and human evaluation are needed.

## One-sentence 27B prompt, matched batching

Both prompts were run fresh on the same 304 candidates in the same order and 16 batches of up to 20; input formatting, JSON schema, temperature and 27B serving model were identical. The short system prompt was: "For each item, set keep to true if the sentence states a research limitation, a future-work direction, or an open problem in the field or prior work, and return exactly one JSON result with its id and keep value." The structured-output schema and the candidate JSON were still provided separately. Both conditions completed all batches on their first attempt.

| Prompt | TP | FP | Precision | Recall | F1 | Input tokens | Output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One sentence | 84 | 33 | 0.7179 | 0.8000 | 0.7568 | 50,161 | 5,829 |
| Neutral rubric | 75 | 10 | 0.8824 | 0.7143 | 0.7895 | 55,617 | 3,852 |

The shorter prompt found nine additional labeled gaps but added 23 false positives; it used 9.8% fewer input tokens but produced more output tokens. This is an exploratory comparison on the same five development papers and LLM-generated labels, and should be confirmed on an independent sample. The matched neutral result differs from the historical neutral run above because batch composition differs.

## Matched single-request comparison: five full papers vs funnel shortlist

Both methods used one fresh Qwen3.8-27B-FP8 request, the same five papers, neutral system prompt, temperature 0, seed 0, thinking disabled, 8,192-token output limit, JSON schema, paper boundaries, reading order, and original global sentence IDs. The only intended input difference is the included sentences: all 1,713 sentences from the five papers versus the 304 sentences surviving Stage A and the frozen NLI Stage B. The funnel request did not include surrounding non-candidate sentences. One request per method returned valid outputs without retry. The denominator is all 105 original LLM-labeled positives across the five papers, including positives filtered before Qwen.

| Method | Qwen requests | Sentences sent | TP | FP | Precision | Recall | F1 | Input tokens | Output tokens | Qwen call seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Five full papers | 1 | 1,713 | 59 | 4 | 0.9365 | 0.5619 | 0.7024 | 96,215 | 493 | 17.117 |
| Funnel candidates in one request | 1 | 304 | 81 | 34 | 0.7043 | 0.7714 | 0.7364 | 12,513 | 644 | 6.199 |

The funnel call used 87.0% fewer Qwen input tokens (7.69× fewer), found 22 more labeled positives overall, and had 30 more false positives. Of the labeled positives, 57 were found by both methods, two by full papers only, and 24 by the funnel only; one full-only positive was removed before the funnel request. Qwen call time excludes Stage A/B preprocessing, data loading, and total end-to-end system cost. Candidate filtering also reduces context, and the difference in the model's response with shorter input may matter; no causal mechanism has been established. This is one fixed five-paper development sample with LLM-generated labels; the earlier disagreement-only adjudication applied to a different batched 27B run and must not be used to rescore these new predictions without review. Prompts and raw outputs are in the committed runner and result files.
