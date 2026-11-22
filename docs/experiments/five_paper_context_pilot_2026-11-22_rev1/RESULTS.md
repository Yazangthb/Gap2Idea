# 4B context descriptions vs original neighboring text

On the same five-paper development check group (1,543 sentences, 77 LLM-labeled positives), all three conditions use the same 258 Stage A+B candidates, 27B neutral system prompt, ordering, IDs, schema and one request. The only intended difference is the supplementary context after each target sentence. The surrounding-text condition adds the previous original sentence (last 800 characters) and next original sentence (first 800 characters), staying within each paper. The 4B condition adds a short generated note per candidate. The no-context one-request baseline includes only the selected sentences; unlike this experiment, earlier batched funnel experiments did provide local context.

At the user's request, all three 27B conditions ran on the same V100 GPU using Qwen3.8-27B Q4_K_M, Ollama digest `15d962ff762f7d02c565fd78afb81eb514692119fce228b6c429cbbdd28a373e`. This model's precision and runtime differ from the earlier H100 FP8 service; results are compared within the V100 Q4 runs, not presented as directly matched to the older FP8 baseline. The exact upstream revision of the GGUF is unverified.

| Added context | TP | FP | Precision | Recall | F1 | 27B input tokens | 27B output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| None | 54 | 24 | 69.2% | 70.1% | **0.697** | 11,099 | 444 |
| 4B descriptions | 54 | 43 | 55.7% | 70.1% | 0.621 | 15,751 | 548 |
| Original neighboring text | 36 | 1 | **97.3%** | 46.8% | 0.632 | 32,772 | 209 |

The generated descriptions increased 27B input tokens by 41.9% over no additional context, but used 51.9% fewer 27B input tokens than verbatim neighbors. They also required additional 4B inference. Including the initially failed 20-item batch, subsequent partial/repeated-ID attempts and recovery calls, 4B made 25 calls consuming 49,552 prompt and 12,608 output tokens, with 213.5 seconds summed call duration; these figures do not establish dollar cost or end-to-end throughput. The 4B notes averaged 11.7 words. The 27B calls' measured durations were 55.342, 39.873 and 59.214 seconds for no context, 4B notes and neighbors respectively; the first included a cold model load, so they should not be treated as a controlled latency comparison.

Notes were intended to explain local context without judging gaps. They were not independently validated and can be misleading: for paper 2302.04972 sentence 381, a note referred to missing PyTorch support although the target and supplied neighboring text do not mention PyTorch. This is an example, not an estimated hallucination rate. The JSON outputs include the notes and underlying calls for inspection. Original extractions can contain mathematical or table text; the source script fixes the 800-character cap and paper boundaries.

An initial 4B batch returned 19/20 notes. The revised run retained valid partial answers and retried missing IDs, but a later batch repeatedly returned one ID with different descriptions. The recovery script retained only unambiguous IDs and requested the ambiguous one separately, preserving all attempts. The five-paper check was used in previous experiments and has LLM-generated labels. This is an exploratory result, not proof of a causal effect or a publication-grade precision estimate; candidate filtering also makes the maximum achievable recall less than 100%.
