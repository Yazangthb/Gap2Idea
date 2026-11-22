# Five-paper prompt selection and separate development check

Four existing prompts were evaluated on five arXiv development papers (1,947 sentences, 136 LLM-labeled positives), for both one-request full-paper and one-request Stage A+B shortlist inputs. The same Qwen3.8-27B-FP8 model, neutralized shared input format, decoding, 8,192-token output cap, and scoring were used. For each prompt, both methods used that same prompt; the funnel sent 443 candidates. The plan records paper IDs, prompts, ordering and hashes.

| Tuning prompt | Full-paper P / R / F1 | Funnel P / R / F1 |
| --- | --- | --- |
| Neutral | 96.5% / 40.4% / 0.570 | 73.1% / 69.9% / **0.714** |
| Strict | 97.7% / 30.9% / 0.469 | 88.8% / 52.2% / 0.657 |
| Keep-biased | 86.3% / 60.3% / **0.710** | 51.9% / 79.4% / 0.628 |
| Short | 85.0% / 50.0% / 0.630 | 44.0% / 80.1% / 0.568 |

The neutral prompt was selected by the prespecified highest funnel F1, with precision, recall and alphabetical order as tie-breakers, and committed before the separate development check. Selection favored the funnel; the full-paper method's best tuning F1 was with a different prompt. The same prompt for both methods controls prompt differences but is not a comparison of each method's independently optimized prompt.

The selected neutral prompt was then used for one full-paper request and one funnel request on a distinct five-paper development group (1,543 sentences, 77 labeled positives). The funnel passed 258 candidates. Both requests succeeded on the first attempt:

| Method | TP | FP | Precision | Recall | F1 | Qwen input tokens | Qwen output tokens | Qwen call seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Five full papers | 29 | 1 | 96.7% | 37.7% | 0.542 | 80,400 | 181 | 12.713 |
| Funnel, one request | 62 | 31 | 66.7% | 80.5% | 0.729 | 11,098 | 527 | 5.462 |

The funnel used 86.2% fewer Qwen input tokens; the Qwen call times exclude preprocessing and other end-to-end costs. The full-paper input provides non-candidate context; the shortlist omits it. The check group was excluded from this prompt-selection step, but is still part of the historical validation pool used in earlier project experiments, so it is not a new independent test. Both groups have LLM-generated labels without human adjudication; only one five-paper group was used for each stage.

## Protocol revision

An initial neutral shortlist response repeated one valid ID and was rejected by the original validator. Both repeated attempts and the initial full-paper run were saved under `../five_paper_dev_tuning_2026-11-22/`. Before restarting all four prompt comparisons, a revised plan was committed specifying that valid duplicated IDs are deduplicated for every method and prompt, while non-input or invalid IDs still fail. One duplicate occurred in the revised neutral tuning shortlist; no duplicates occurred in the two development-check outputs. No failed outputs were scored as negative predictions.
