# Precision-oriented prompt development

All calls used fine-tuned NLI, one Qwen3.8-27B-FP8 request per five-paper group, original sentence neighbors (up to 800 characters on each side), identical candidate IDs and order, temperature 0 and seed 0. The tuning group had 136 LLM-labeled positives and 346 candidates. The neutral and existing strict prompts were compared with a new specificity-focused prompt; wording was frozen in the plan before inference.

| Prompt, tuning group | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral | 111 | 62 | 64.2% | 81.6% | 0.718 |
| Strict | 64 | 2 | 97.0% | 47.1% | 0.634 |
| Specific | 104 | 31 | 77.0% | 76.5% | 0.768 |

The specificity prompt was selected as the most precise prompt with at least 60% recall, a prespecified exploratory floor rather than a publication requirement. It was then compared against neutral on five different development papers, 77 labeled positives and 200 candidates:

| Prompt, separate development check | TP | FP | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Neutral | 49 | 15 | 76.6% | 63.6% | 0.695 |
| Specific | 11 | 0 | 100% | 14.3% | 0.250 |

The prompt did not provide a useful precision–recall improvement on the check group; it accepted only 11 sentences. A 100% precision figure based on 11 outputs is not a reliable high-precision claim. The groups are separate for this selection step but are part of the previously studied validation pool; the fine-tuned checkpoint and threshold used all validation papers, and labels are LLM-generated. This is an exploratory result, not a blind or human evaluation.
