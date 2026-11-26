# Context layout comparison on 34 development papers

The three conditions use exactly the same 1,746 fine-tuned NLI candidates, frozen threshold, Qwen3.8-27B FP8 model, neutral system prompt, original global candidate IDs, paper groups, temperature and seed. One request per group, with one boolean verdict per candidate. Eight groups comprise 28 arXiv and 6 biomedical papers; smaller final groups contain three and one papers. Labels are LLM-generated, and all papers have previous development exposure; this is exploratory rather than independent or human-validated evaluation.

| Input format | TP | FP | Precision | Recall | F1 | Qwen input tokens | Qwen output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing: target, then previous and next | 512 | 98 | 83.93% | 73.25% | 0.7823 | 229,786 | 23,595 |
| Surrounding: previous, target, next | 503 | 93 | 84.40% | 71.96% | 0.7768 | 228,040 | 24,581 |
| No context: target alone | 461 | 81 | 85.06% | 65.95% | 0.7429 | 77,185 | 24,581 |

All metrics use the same 699 labeled positives in all original sentences as the recall denominator. Compared to the current layout, no context reduces Qwen input tokens by 66.4%, but finds 51 fewer labeled gaps and 17 fewer false positives. The surrounding layout increases precision by 0.47 percentage points and reduces recall by 1.29 points, using a comparable number of input tokens. No uncertainty estimates or repeated calls were run, so the small precision differences should not be treated as conclusive.

For 28 arXiv papers, existing / surrounding / no-context P-R-F1 is 82.95/73.00/77.66, 83.27/70.50/76.35, and 84.35/64.67/73.21 percent. For 6 biomedical papers, the corresponding values are 90.24/74.75/81.77, 90.91/80.81/85.56, and 89.02/73.74/80.66 percent. The biomedical group is small. Do not compare these development scores directly to the full-paper method's previously measured test scores as though they were on the same papers.

The surrounding request format is `Previous: <previous sentence>`, then `[id] <target sentence>`, then `Next: <following sentence>`. Both context formats clip neighbors to 800 characters. The no-context format omits the previous and next lines. Paper boundaries are retained, and only target IDs are classified. The exact prompt, input hashes, paper IDs, model settings, and raw outputs are saved with this report. Baseline calls were reused from the preceding 68-paper study and were made at a different time with a 65,536-token output cap; new calls used 16,384. All responses completed within their limits and each input ID appeared exactly once; no extra IDs were returned. Token counts include all attempts. Both layouts passed context preflight.
