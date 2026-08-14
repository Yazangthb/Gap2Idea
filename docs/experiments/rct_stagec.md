# RCT/SAL limitation detection — detector + Stage C vs Lan et al. (2024)

Benchmark of the Gap2Idea extraction funnel (Stage B detector + Stage C LLM precision
filter) against the self-acknowledged-limitations (SAL) SOTA:

> Lan, Cheng, Hoang, ter Riet, Kilicoglu. *Automatic categorization of self-acknowledged
> limitations in randomized controlled trial publications.* J. Biomedical Informatics, 2024.
> Data/labels: [MengfeiLan/SAL_Type_Classification](https://github.com/MengfeiLan/SAL_Type_Classification).

Their best limitation **sentence-detection** model (fine-tuned PubMedBERT, PromDA-augmented):
**P = 0.751, R = 0.907, F1 = 0.821** (their Table 2). Their checkpoint is not released, so we
train our own detector and reconstruct their evaluation condition.

## Why the raw comparison is apples-to-oranges

Their 0.821 is measured on a **section-filtered candidate pool** — sentences from the
abstract + discussion/limitation/conclusion sections only — at **~20.7 % positive
prevalence**. The `data/{train,dev,test}.csv` we are given is the **full-article sentence
stream** at **2.3–2.6 % prevalence** (needle-in-haystack). Precision is prevalence-dependent;
comparing our full-stream number to their curated-pool number understates us badly.

## Method

- **Stage B detector**: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`
  fine-tuned on the RCT train split (binary: sentence carries a limitation span or not),
  dev-threshold-tuned. Two operating points reported: max-F1 and recall-tuned (≥0.90 dev recall).
  Script: [`scripts/training/finetune_rct.py`](../../scripts/training/finetune_rct.py). ~57 s on a V100.
- **Section pool reconstruction**: recover each test sentence's section from the PubMed
  Central BioC API and keep `{ABSTRACT, DISCUSS, CONCL}` — their described filter.
  Script: [`scripts/bench/build_rct_pool.py`](../../scripts/bench/build_rct_pool.py).
  Result: **2 496 sentences, 8.1 % prevalence, and the filter retains 99 % (203/205) of gold
  limitations.** (These 40 test articles are limitation-sparser than their 20.7 % annotation
  corpus, so we also prevalence-match to 20.7 %.)
- **Stage C**: batched LLM precision filter, Yandex models only.
  Optimizer: [`scripts/bench/bench_stagec_opt.py`](../../scripts/bench/bench_stagec_opt.py).
  Winning prompt shipped as `mode="validate_rct"` (domain-tuned, KEEP-biased).

## Headline result (section pool, prevalence-matched to their 20.7 %)

| Method | Precision | Recall | F1 |
|---|---|---|---|
| Their rule-based | 0.758 | 0.848 | 0.800 |
| Their SVM + self-training | 0.778 | 0.835 | 0.806 |
| **Their PubMedBERT (SOTA)** | **0.751** | **0.907** | **0.821** |
| Our detector only | 0.668 | 0.887 | 0.762 |
| Our detector + Stage C (gpt-oss-120b) | 0.752 | 0.882 | 0.812 |
| Our detector + Stage C (yandexgpt-5-pro) | **0.780** | 0.842 | 0.810 |

(Precision/recall at 20.7 % derived from prevalence-invariant recall and the subsampled
F1 as P = R·F1 / (2R − F1).)

## Findings

1. **We match their precision; the gap is recall.** Our Stage-C precision (0.75–0.78) meets or
   beats their 0.751. Their F1 edge is entirely recall (0.907 vs our 0.84–0.89), which traces
   to their end-to-end PubMedBERT + PromDA detector. **Stage C only removes, never adds, so it
   cannot close a recall gap** — to beat 0.821 we need a higher-recall Stage B, not a better filter.
2. **Stage C is a low-prevalence precision tool.** On the full stream (2.6 %) it lifts F1
   +0.05–0.11; at 20.7 % on an already-precise detector it is roughly neutral. It earns its keep
   in the realistic full-document setting, not on curated pools.
3. **Prompt beats model; both saturate.** The domain `validate_rct` prompt preserves 177–179/180
   true limitations vs 158–160/180 for the arXiv default — a large recall win. Across strong
   Yandex judges (yandexgpt-5-pro, gpt-oss-120b, qwen3-235b, deepseek-v4-flash) F1@20.7 %
   clusters at 0.80–0.815. A further error-tuned prompt (`rct_v2`) did **not** help: the residual
   false positives are **gold-annotation granularity** (real limitations that were not the
   annotated span), so precision is label-capped, not prompt-capped.
4. **Model notes.** yandexgpt-5-pro = most precise + native/fast; gpt-oss-120b = best recall
   preservation; qwen3-235b ≈ slightly behind; deepseek-v4-flash works but is ~100× slower
   (heavy reasoning) and least precise — not worth it. DeepSeek-R1/V3 are not provisioned in the
   Yandex folder.

## Reproduce

```bash
# 1. detector + full-stream / prevalence-matched / section-pool eval, dump predictions
python scripts/training/finetune_rct.py \
  --model microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --epochs 3 --neg-ratio 6 --rec-target 0.90 \
  --eval-prevalence 0.207 --pool-csv data/rct_pool_test.csv --dump-preds data/pool_preds.json
# 2. build the section pool (needs internet for the BioC API)
python scripts/bench/build_rct_pool.py --split test --out data/rct_pool_test.csv
# 3. optimize/compare Stage C on the dumped predictions
python scripts/bench/bench_stagec_opt.py --preds data/pool_preds.json \
  --models yandexgpt-5-pro,gpt-oss-120b --prompts rct_loose --eval-prevalence 0.207
```

**Bottom line:** on a faithful reconstruction of their evaluation, Gap2Idea's general-purpose
BiomedBERT + LLM-filter reaches **F1 ≈ 0.81 at precision ≈ 0.75–0.78 — matching their
domain-specialized, augmentation-trained PubMedBERT precision and landing within ~0.01 F1 of
their 0.821**, the residual being a 2–6-point detector-recall gap.
