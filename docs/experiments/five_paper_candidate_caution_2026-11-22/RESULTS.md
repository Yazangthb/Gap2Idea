# Candidate-fallibility prompt on the same five papers

The previously repeated zero-shot and fine-tuned shortlist inputs were reconstructed and verified byte-for-byte against their recorded input hashes. The same Qwen3.8-27B Q4_K_M V100 setup, neutral prompt, one request per input, original neighbors, decoding and schema were used. This sentence was appended to the system prompt:

> Some listed candidates may be genuine research gaps, while others may not be; assess each candidate independently and do not assume it is correct merely because it was selected.

| Shortlist | Prompt | TP | FP | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Zero-shot NLI (258 candidates) | Original | 36 | 1 | 97.3% | 46.8% | 0.632 |
| Zero-shot NLI | Added sentence | 47 | 23 | 67.1% | 61.0% | 0.640 |
| Fine-tuned NLI (200 candidates) | Original | 53 | 21 | 71.6% | 68.8% | 0.702 |
| Fine-tuned NLI | Added sentence | 47 | 15 | 75.8% | 61.0% | 0.676 |

The zero-shot run added 12 labeled positives and 22 labeled negatives, while dropping one labeled positive. Fourteen of the added negatives are from paper 2102.03018. The fine-tuned run added one labeled positive while dropping seven labeled positives and six labeled negatives. Thus the instruction improved fine-tuned precision by 4.2 percentage points but lowered recall by 7.8 points; on the zero-shot shortlist it substantially lowered precision. Extra input cost was 33 tokens per request compared to the original requests.

The original requests were repeated three times each with identical output; the amended prompt was run once per input. These five development papers have LLM-generated labels and have already informed multiple iterations, so the comparison is exploratory and apparent false positives need human review. The same Q4 model and hardware were used, not the earlier H100 FP8 setup. Both new requests succeeded on the first attempt; saved outputs include raw responses and metrics.
