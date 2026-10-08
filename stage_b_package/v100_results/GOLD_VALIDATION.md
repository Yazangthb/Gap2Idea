# Gold set validation (Stage B benchmark)

Validates `data/bench_gap/gold_sentences_v2.tsv` (the set all Stage-B gold numbers are measured on).
Checked against the full paper text (`data/scibert_prep/gold_papers.jsonl`) and the Stage-A candidate
slice (`data/external/gold_eval.json`). Metric = token-containment (scramble-robust word overlap).

## Composition
- **49 gaps / 10 papers**, 0 duplicate sentences, mean 24 words/gap, none < 6 words.
- Types: **scope 14 · future_work 14 · limitation 11 · assumption 9 · open_problem 1**.
- Gaps/paper: 2–9.

## Validity
| check | result | verdict |
|---|---|---|
| gap sentence present in its source paper (cont ≥ 0.8) | **49 / 49** | ✅ faithful — 0 hallucinated/paraphrased, though GPT-4o-built |
| reachable in Stage-A slice (cont ≥ 0.7) | **42 / 49 (86 %)** | ⚠️ Stage A drops 7 → recall ceiling ≈ 0.86 |
| duplicates / garbled / too-short | none | ✅ |

## Key finding — mixed denominator
Only **25 / 49 (51 %)** gold gaps are `limitation` or `future_work` — the **only two classes the Stage-B head
predicts**. The other 24 (`scope`, `assumption`, `open_problem`) are gap types the pipeline does **not** target.

→ **`recall_all` is misleading on this gold.** A perfect head scores ~0.51 by construction (plus incidental
hits when an untargeted gap's sentence is kept anyway). Our best MiniLM `recall_all` = 0.571 sits just above
that floor for exactly this reason.

## How to use the gold correctly
1. **Report per-class recall, not `recall_all`:** limitation **0.636**, future_work **0.857** (best head). These
   measure what the classifier actually does. Drop `recall_all` or tag it a mixed-denominator number.
   (Alternatively: restrict the benchmark to the 25 limitation+future_work gaps for a clean denominator.)
2. **State the ~86 % Stage-A reachability ceiling** — 7 gaps are unreachable (Stage A sliced them away), so
   Stage B can't recover them regardless of model.

## Caveats (bound claims, don't block use)
- **GPT-4o silver**, not human-verified (sentences are faithful extractions, but class labels are LLM-assigned).
- **Small: 10 papers / 49 gaps** — adequate for *ranking* methods, not for absolute SOTA claims.
- **Upgrade path:** a short human audit of the 49 labels + review of the 7 Stage-A misses would make it a
  defensible standalone benchmark.

## The 7 Stage-A misses (unreachable; cap recall)
Gaps present in the paper (cont_full high) but absent from the slice (cont_slice < 0.7) — e.g.
`2309.09902` scope (slice 0.64), `2002.09564` limitation (slice 0.66). These are Stage-A localization
failures, not gold errors.
