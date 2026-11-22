# Explicitly negated NLI hypotheses on development data

The same 11,201 label-free development sentences were scored on two V100 GPUs: the original ModernBERT-large-NLI checkpoint (`ca476cb9`, SHA-256 `f56738ff...`) and the fine-tuned checkpoint (SHA-256 `15e6b714...`). For each sentence, each model evaluated three positive hypotheses and their explicitly negated counterparts:

- “This is a limitation.” / “This is not a limitation.”
- “This is future work.” / “This is not future work.”
- “This is a research gap.” / “This is not a research gap.”

Each sentence–hypothesis pair was scored independently, with the sentence truncated when the pair exceeded 256 tokens. For each type, positive support is `E_pos/(E_pos+C_pos)`; negative support for a gap is `C_neg/(E_neg+C_neg)`. The scores ignore neutral probability and are not calibrated gap probabilities. We compare taking the maximum type score using positive alone, negated alone, the per-type mean, the per-type maximum (“either”), and the per-type minimum (“both”).

Thresholds were selected on 17 development papers (half A) to retain at least 80%, 90% or 95% of *all* labeled gaps, including cleanup losses. Those fixed thresholds were then applied to the other 17 development papers (half B: 4,907 sentences, 348 labeled gaps). The table's target refers to half A, so actual recall on B can differ. The A/B division is paper-level and stratified, but both halves were previously explored; the fine-tuned checkpoint was also selected using the full development set, including B. The labels are LLM-generated, and this is not an independent human-labeled benchmark.

## Thresholds selected for 90% recall on half A

| Model | Score | Threshold | Half B sentences passed | Half B precision | Half B recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Zero-shot | Positive only | 0.240596 | 1,400 | 22.4% | 89.9% |
| Zero-shot | Negative only | 0.953859 | 2,800 | 11.4% | 91.4% |
| Zero-shot | Mean of both | 0.570139 | 1,523 | 20.8% | 91.1% |
| Zero-shot | Either | 0.956431 | 2,697 | 11.8% | 91.1% |
| Zero-shot | Both | 0.240596 | 1,399 | 22.4% | 89.9% |
| Fine-tuned | Positive only | 0.002487 | 778 | 39.6% | 88.5% |
| Fine-tuned | Negative only | 0.667432 | 3,949 | 8.0% | 90.5% |
| Fine-tuned | Mean of both | 0.499120 | 3,712 | 8.7% | 92.5% |
| Fine-tuned | Either | 0.978875 | 3,871 | 8.1% | 89.7% |
| Fine-tuned | Both | 0.002487 | 778 | 39.6% | 88.5% |

At a development-half-A target of 90% recall, negated wording alone or combined by averaging passed many more candidates for similar half-B recall. Requiring both was effectively identical to positive-only for the fine-tuned model and removed just one candidate for the zero-shot model, with the same true-positive count. This is a test of these specific wordings and scoring combinations, not proof that no negated wording can help.

## Other half-A recall targets

| Model / score | Target on A | B sentences passed | B precision | B recall |
| --- | ---: | ---: | ---: | ---: |
| Zero-shot positive | 80% | 754 | 37.3% | 80.7% |
| Zero-shot negative | 80% | 1,391 | 20.5% | 81.9% |
| Zero-shot mean | 80% | 782 | 35.9% | 80.7% |
| Zero-shot positive | 95% | 2,414 | 13.5% | 94.0% |
| Zero-shot negative | 95% | 3,649 | 9.1% | 95.4% |
| Zero-shot mean | 95% | 3,447 | 9.6% | 95.1% |
| Fine-tuned positive | 80% | 469 | 56.9% | 76.7% |
| Fine-tuned negative | 80% | 3,909 | 7.4% | 82.8% |
| Fine-tuned mean | 80% | 2,050 | 13.9% | 81.9% |
| Fine-tuned positive | 95% | 2,057 | 16.1% | 95.4% |
| Fine-tuned negative | 95% | 3,968 | 8.4% | 95.4% |
| Fine-tuned mean | 95% | 3,907 | 8.5% | 95.1% |

## Are positive and negated outputs complementary?

For the “research gap” pair on half B after cleanup, the original model's positive score and inverted negative score have Pearson correlation 0.277; for the fine-tuned model they have correlation -0.668. These are correlations of the normalized E/(E+C) scores, not calibrated probabilities or logical truth. The fine-tuned model was trained on the positive hypotheses, not their negated forms; its inverted negative scores were heavily concentrated near one even on many non-gap sentences. Model wording sensitivity and ignoring the neutral class are plausible contributors, not a proven causal diagnosis.

Freshly recomputed positive class probabilities closely matched the prior saved validation scores (mean absolute difference at most 0.000336 for zero-shot and 0.000011 for fine-tuned, across the three classes and hypotheses); small numerical differences remain. The raw six-hypothesis class probabilities, thresholds, counts, inputs hashes and inference code are included. Qwen was not run; candidate counts are not token counts or final funnel results.
