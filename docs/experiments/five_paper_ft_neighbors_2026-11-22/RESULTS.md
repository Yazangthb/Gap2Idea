# Fine-tuned NLI + Qwen with neighbors: five-paper pilot

This checks the fine-tuned NLI candidates on the same five development papers as the earlier zero-shot NLI neighbor experiment. Both methods use one Qwen3.8-27B Q4_K_M request on V100, the identical neutral system prompt, paper order, original sentence IDs, temperature 0, seed 0, thinking disabled and output schema. Each candidate is accompanied by the preceding sentence (last 800 characters) and following sentence (first 800 characters), within the same paper. The difference is the candidate set.

| Stage | Zero-shot NLI | Fine-tuned NLI |
| --- | ---: | ---: |
| Sentences selected for Qwen | 258 | 200 |
| Labeled gaps selected by NLI | 66 / 77 | 68 / 77 |
| NLI recall | 85.7% | 88.3% |
| Qwen final TP / FP | 36 / 1 | 53 / 21 |
| Qwen final precision | 97.3% | 71.6% |
| Qwen final recall | 46.8% | 68.8% |
| Qwen final F1 | 0.632 | 0.702 |
| Qwen input tokens | 32,772 | 25,907 |

The fine-tuned shortlist included 62 labeled gaps also in the zero-shot shortlist, six unique labeled gaps, and excluded four that the zero-shot shortlist included. After Qwen, all 37 zero-shot outputs (36 labeled gaps and one labeled negative) were also present in the fine-tuned output; Qwen additionally selected 17 labeled gaps and 20 labeled negatives. Thus the final precision difference is not explained solely by fewer candidates or more NLI-stage positives. Changing the candidate list also changes the overall model input and its decisions; this single comparison does not establish causality.

Of the 20 additional labeled negatives, 17 were already in both NLI shortlists: Qwen rejected them when reviewing the zero-shot shortlist but selected them when reviewing the fine-tuned shortlist. Three were unique to the fine-tuned shortlist. Fifteen of the 20 were from paper `2102.03018`. As a post-hoc diagnostic only, removing that paper changes fine-tuned precision from 53/74 (71.6%) to 40/46 (87.0%); that exclusion is not the result on the five-paper sample. Some additional labeled negatives describe potential method weaknesses, so their true status needs human review. Different whole-request candidate composition can affect Qwen decisions, but this observation alone does not prove why its decisions changed.

Fine-tuned NLI threshold 0.0021033689845353365 is chosen on the entire 34-paper validation set to achieve 90% Stage A+B recall; these five papers belong to that set. The fine-tuned checkpoint was also selected on that validation set. Labels are LLM-generated and some apparent false positives may be labeling disagreements. This is an exploratory development result, not an independent precision estimate. The model is the V100 Q4 GGUF used in the neighbor experiment, not the earlier H100 FP8 model; its exact original HF revision is unverified. Qwen token counts exclude NLI and preprocessing. All five papers' candidates were in one Qwen request, not batches of 20.

Earlier experiments did evaluate fine-tuned NLI followed by Qwen 27B with ±1 context in batches of 20; on 28 arXiv test papers the balanced mode reported F1 0.734. Those older results use different paper sets and call shapes and should not be merged with the five-paper result. See `docs/experiments/llm_size.md` and `scripts/bench/funnel_nli_ft.py`.
