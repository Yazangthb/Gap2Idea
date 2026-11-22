# Repeatability of the two Qwen neighbor-context requests

The exact two inputs from the earlier five-paper comparison were reconstructed and checked against their original input hashes: zero-shot NLI's 258-candidate list and fine-tuned NLI's 200-candidate list. Each complete list, with the same original neighboring sentences, was sent three additional times in a fixed alternating order. Each comparison therefore contains the earlier original result plus three new requests.

Every request used the same Qwen3.8-27B Q4_K_M Ollama model and digest, one V100, the same neutral system prompt, temperature 0, seed 0, thinking disabled, 131,072-token context, 8,192-token output cap and JSON schema. The original inputs came from separate earlier server runs; the repeats were conducted in one later session, with potential prefix caching. All six new requests succeeded on the first attempt.

| Input | Requests including original | Accepted sentences in every run | Variable sentence decisions | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero-shot NLI shortlist | 4 | 37 | 0 | 97.3% | 46.8% | 0.632 |
| Fine-tuned NLI shortlist | 4 | 74 | 0 | 71.6% | 68.8% | 0.702 |

For each input, all four runs selected **exactly the same sentence IDs** (every pairwise Jaccard similarity was 1.0), and the raw JSON response text was also identical. Metrics are against the existing LLM labels. Repeats with the exact same request appear stable in this setup, while the two different input lists produce different decisions. This does not establish the cause of the input-list effect or guarantee repeatability across more runs, seeds, model revisions, serving engines, hardware, or the H100 FP8 setup. Token usage and request times are present in the raw files, but cached and uncached calls should not be compared as latency measurements.
