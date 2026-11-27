# 75/25 NLI score fusion on the complete test split

We evaluated the fusion chosen after the development check: transform the normal and fine-tuned NLI scores to empirical percentiles relative to Stage-A-passing development half-A scores, then use 25% normal and 75% fine-tuned percentiles. The threshold was fixed on half A to achieve at least 90% Stage B recall there. All scores were already computed; no NLI inference was run. This is a static weighted fusion, not a learned router.

The Qwen test sends the fused shortlist once per original test group, covering all 34 papers in matched groups of 17, 10, 5, and 2. It uses the original surrounding-neighbor format, neutral prompt, Qwen3.8-27B FP8 and one boolean verdict per target. The existing fine-tuned funnel and full-paper Qwen responses use the same papers and groups. The original fine-tuned funnel uses its threshold selected on the whole development set; the Stage B table also shows the fine-tuned and normal thresholds selected on development half A. Recall uses all 664 LLM-labeled test gaps, including Stage A losses.

## Stage B, before Qwen

| Stage B configuration | Sentences passed | TP | FP | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Fine-tuned NLI, original threshold | 1,998 | 607 | 1,391 | 30.38% | 91.42% |
| Fine-tuned NLI, half-A threshold | 1,958 | 607 | 1,351 | 31.00% | 91.42% |
| Normal NLI, half-A threshold | 2,579 | 607 | 1,972 | 23.54% | 91.42% |
| 75/25 fusion, half-A threshold | 1,775 | 604 | 1,171 | 34.03% | 90.96% |

Relative to the original fine-tuned shortlist, fusion passes 223 fewer sentences: 3 fewer labeled gaps and 220 fewer labeled non-gaps. This raises measured Stage B precision by 3.65 percentage points and lowers recall by 0.46 points. Running both NLI models adds Stage B computation, not included in Qwen token counts.

## End-to-end, after Qwen

| Method | Qwen-kept | TP | FP | Precision | Recall | F1 | Qwen input tokens | Qwen output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-paper Qwen | 560 | 478 | 82 | 85.36% | 71.99% | 78.10% | 612,965 | 134,727 |
| Fine-tuned NLI → Qwen | 643 | 510 | 133 | 79.32% | 76.81% | 78.04% | 259,776 | 27,734 |
| **75/25 fusion → Qwen** | 701 | 524 | 177 | 74.75% | 78.92% | 76.78% | 224,829 | 21,666 |

Fusion reduced Qwen input tokens by 13.5% versus fine-tuned NLI alone and 63.3% versus full-paper Qwen. Despite its more precise Stage B shortlist, the fused pipeline's final precision fell 4.57 points versus fine-tuned alone, while recall rose 2.11 points: it produced 14 more labeled true positives and 44 more labeled false positives. Fusion changes which candidates share the request; the observed numbers show that Qwen's behavior changed, but do not establish a specific causal mechanism. Qwen token totals do not include either NLI model or preprocessing and are not monetary/runtime costs.

End-to-end results by domain (precision / recall / F1, percent):

| Method | 28 arXiv papers | 6 biomedical papers |
|---|---|---|
| Full papers | 82.28 / 71.35 / 76.42 | 99.03 / 74.45 / 85.00 |
| Fine-tuned funnel | 75.56 / 77.42 / 76.48 | 99.03 / 74.45 / 85.00 |
| Fusion funnel | 70.47 / 79.70 / 74.80 | 99.05 / 75.91 / 85.95 |

The biomedical subset is only six papers and uses LLM-derived three-type labels, not the separate SAL human limitation labels. The test papers were examined in previous experiments, the 75/25 weight was selected after seeing development half B, and labels are not fully human validated. Thus these are exploratory results, not an untouched test or a statistically established difference. One Qwen call per group does not characterize repeat variability.

All four fused Qwen responses passed ID-completeness checks. The exact threshold, test groups, candidate and input hashes, raw responses, domain results, token usage and preflight are saved in this directory; `scripts/bench/nli_fusion_qwen_test.py` is the runner.
