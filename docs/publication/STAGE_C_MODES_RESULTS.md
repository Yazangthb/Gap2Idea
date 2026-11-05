# Stage C modes: results for the paper

Saved 2026-11-05. Project priority: **extracted gaps should be true, so precision comes before recall.** High precision is the main mode; the other two show the trade-off.

**Setup, all modes:** 34 gold test papers, 11,741 sentences, 664 gaps, LLM labels. Stage A junk filter, Stage B short-hypothesis ModernBERT NLI at τ 0.354 (2,522 candidates, fixed on validation), Stage C Qwen3.8-27B with context, batch 20, no thinking, seed 0. The modes differ only in the rubric's final instruction. CIs resample papers (2,000 draws).

## Main table

| Mode | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Kept | True / false |
|---|---|---|---|---|---|
| **High precision** (main) | **86.6%** (79.8–93.4) | 57.4% (49.4–64.3) | 0.690 (0.619–0.748) | 440 | 381 / 59 |
| Balanced | 68.2% (60.0–76.7) | 79.7% (75.8–83.3) | 0.735 (0.678–0.789) | 776 | 529 / 247 |
| High recall | 56.3% (46.9–67.3) | 85.7% (83.5–88.1) | 0.679 (0.604–0.756) | 1,011 | 569 / 442 |

Before Stage C: P 24.0%, R 91.3%, F1 0.380.

## By domain

| Mode | arXiv, 28 papers: P / R / F1 | Biomedical, 6 papers: P / R / F1 |
|---|---|---|
| High precision | **83.4%** (76.0–91.3) / 53.3% / 0.650 | **97.1%** (94.5–99.0) / 73.0% / 0.833 |
| Balanced | 64.2% (56.0–73.6) / 78.4% / 0.706 | 87.2% (82.4–92.2) / 84.7% / 0.859 |
| High recall | 51.7% (42.8–62.7) / 84.6% / 0.642 | 82.6% (80.2–86.1) / 89.8% / 0.860 |

The biomedical papers are a subset of SAL-40. With 6 papers, their CIs understate the real uncertainty.

## Recall by gap type

| Mode | Limitation (220) | Future work (97) | Research gap (347) |
|---|---|---|---|
| High precision | 55.9% | 82.5% | 51.3% |
| Balanced | 71.4% | 84.5% | 83.6% |
| High recall | 79.5% | 86.6% | 89.3% |

## Excluding `research_gap` from scoring

| Mode | Precision | Recall | F1 |
|---|---|---|---|
| High precision | 77.5% (64.5–89.5) | 64.0% (54.4–72.5) | 0.701 (0.593–0.788) |
| Balanced | 49.2% (38.1–62.5) | 75.4% (69.2–81.1) | 0.595 (0.496–0.697) |
| High recall | 36.9% (26.9–51.0) | 81.7% (76.7–86.5) | 0.509 (0.401–0.633) |

## Paired F1 differences

| Comparison | Difference | 95% CI |
|---|---|---|
| Balanced vs high recall | +0.055 | +0.025 to +0.080 |
| High precision vs high recall | +0.011 | −0.054 to +0.060 |
| High precision vs balanced | −0.045 | −0.095 to −0.006 |
| High precision vs bge-small + Qwen (F1 0.665) | +0.025 | −0.037 to +0.087 |
| High precision vs ModernBERT alone (F1 0.614) | +0.076 | +0.020 to +0.132 |

## NLI + Qwen on human-labelled benchmarks

The chosen funnel on datasets with human labels. Each uses one hypothesis ("This is a limitation." or "This is future work."), the threshold rule from gold (recall at least 90% on the benchmark's dev data), and the frozen rubric narrowed to the benchmark's one label. No failed batches. Script: `scripts/bench/nli_sal_fws.py`.

**SAL-40 test** (40 papers, 7,968 sentences, 205 limitations). Stage A section gate keeps 1,885 sentences and 203 limitations. τ 0.7286 from dev, 639 candidates; before Stage C P 29.1%, R 90.7%, F1 0.441.

| Mode | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Kept | True / false |
|---|---|---|---|---|---|
| **High precision** (main) | **66.2%** (59.7–72.8) | 84.9% (80.1–89.3) | **0.744** (0.695–0.789) | 263 | 174 / 89 |
| Balanced | 60.7% (54.6–66.9) | 87.3% (83.0–91.4) | 0.716 (0.670–0.760) | 295 | 179 / 116 |
| High recall | 54.8% (49.0–61.3) | 88.3% (83.9–92.2) | 0.677 (0.627–0.725) | 330 | 181 / 149 |

**SAL dev check** (40 other papers, 7,729 sentences, 180 limitations, no prompt ever seen on them; the threshold was set here). 591 candidates; before Stage C P 27.2%, R 89.4%, F1 0.418. Rounding τ to 0.7286 dropped one candidate from the exact dev recall of 90%.

| Mode | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Kept | True / false |
|---|---|---|---|---|---|
| High precision | 64.3% (55.7–72.8) | 80.0% (74.6–86.0) | 0.713 (0.647–0.774) | 224 | 144 / 80 |
| Balanced | 59.5% (50.7–68.0) | 85.0% (79.9–90.2) | 0.700 (0.628–0.764) | 257 | 153 / 104 |
| High recall | 55.3% (46.9–63.9) | 86.7% (81.6–91.7) | 0.675 (0.602–0.742) | 282 | 156 / 126 |

**FWS test** (209 papers, 1,506 conclusion-section sentences, 220 future-work sentences). All text lowercased, see below. No Stage A. τ 0.1922 from 418 other papers, 400 candidates; before Stage C P 48.2%, R 87.7%, F1 0.623.

| Mode | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Kept | True / false |
|---|---|---|---|---|---|
| **High precision** (main) | **76.5%** (70.0–82.9) | 75.5% (68.3–82.1) | **0.760** (0.705–0.807) | 217 | 166 / 51 |
| Balanced | 73.9% (67.8–80.0) | 78.6% (71.8–85.0) | 0.762 (0.709–0.808) | 234 | 173 / 61 |
| High recall | 69.1% (62.9–75.3) | 83.2% (76.2–89.4) | 0.755 (0.702–0.804) | 265 | 183 / 82 |

**NLI vs PubMedBERT on SAL-40, same mode** (NLI minus PubMedBERT, paired). PubMedBERT was trained on SAL's labels; NLI needs no training labels.

| Mode | Precision difference (95% CI) | F1 difference (95% CI) |
|---|---|---|
| High precision | −8.4 points (−11.8 to −4.5) | −0.021 (−0.053 to +0.014) |
| Balanced | −6.9 points (−11.5 to −2.3) | −0.035 (−0.068 to 0.000) |
| High recall | −9.0 points (−12.7 to −5.1) | −0.059 (−0.089 to −0.027) |

**NLI kept sentences on SAL-40 against our LLM labels (high precision):** precision 93.2% on limitation labels and 95.1% on any gap label. Thirteen of 263 kept sentences have no LLM gap label.

**FWS casing problem:** in `fws_corpus.csv` every positive is entirely lowercase and almost no negative is, so "all lowercase means positive" scores P 99.8%, R 100%. Our file has the same byte size as the published `Corpus_For_FWS_Recognition.csv`, so this appears to come from the dataset, though the two files weren't compared directly. Any FWS result from a case-sensitive model or an LLM that saw the raw text is unreliable, including our earlier F1 0.707.

## Cost: estimated LLM input tokens

Counted with tiktoken `cl100k_base` on prompts and items as sent, high precision mode, batch 20. Output tokens are excluded. Script: `scripts/bench/stage_c_cost.py`. Replaces the earlier 3.74× estimate, which used PubMedBERT and a different counting method.

| Dataset | Candidates | Funnel | Whole paper, one call | Every sentence | Saving vs whole paper | Saving vs every sentence |
|---|---|---|---|---|---|---|
| SAL-40, NLI | 639 | 102,978 | 317,475 | 1,345,409 | 3.08× | 13.07× |
| SAL-40, PubMedBERT | 391 | 62,835 | 317,475 | 1,345,409 | 5.05× | 21.41× |
| FWS, NLI | 400 | 60,410 | 123,445 | 225,015 | 2.04× | 3.72× |
| Gold test, NLI | 2,522 | 450,955 | 562,440 | 2,318,416 | 1.25× | 5.14× |
| Gold arXiv, NLI | 2,137 | 385,897 | 506,753 | 2,087,907 | 1.31× | 5.41× |

On gold, each candidate carries up to 600 characters of context, so the funnel sends nearly as much text as one whole-paper call. Whether the funnel is worthwhile there depends on accuracy, which task 5a measures.

## SAL-40 with PubMedBERT Stage B (comparison)

40 papers, 7,968 sentences, 205 limitations. Stage A section gate, Stage B fine-tuned PubMedBERT (391 candidates, 180 are limitations), Stage C Qwen as above. No failed batches. The rubric is narrowed to limitations, because SAL labels nothing else; this wording was fixed before any SAL verdict.

| Mode | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Kept | True / false |
|---|---|---|---|---|---|
| **High precision** (main) | **74.5%** (68.4–80.7) | 78.5% (72.1–84.1) | **0.765** (0.714–0.811) | 216 | 161 / 55 |
| Balanced | 67.6% (59.6–75.0) | 84.4% (79.3–88.8) | 0.751 (0.691–0.803) | 256 | 173 / 83 |
| High recall | 63.8% (56.9–70.7) | 86.8% (82.5–90.5) | 0.736 (0.682–0.786) | 279 | 178 / 101 |
| Before Stage C | 46.0% (41.0–51.5) | 87.8% (83.7–91.3) | 0.604 (0.554–0.653) | 391 | 180 / 211 |
| Earlier `rct_loose` prompt on Qwen | 54.7% (48.1–61.9) | 87.3% (83.0–91.0) | 0.673 (0.614–0.729) | 327 | 179 / 148 |

| High precision vs | Precision difference (95% CI) | F1 difference (95% CI) |
|---|---|---|
| Before Stage C | +28.5 points (+24.2 to +33.4) | +0.161 (+0.124 to +0.199) |
| `rct_loose` on Qwen | +19.8 points (+15.1 to +25.0) | +0.092 (+0.048 to +0.137) |

**The same kept sentences against our LLM labels:** SAL may under-label, so these show an upper reading, not true precision.

| Mode | Precision, LLM limitation label | Precision, any LLM gap label | Kept sentences with no LLM gap label |
|---|---|---|---|
| High precision | 94.9% (91.2–98.0) | 96.3% (92.8–99.0) | 8 of 216 |
| Balanced | 91.0% (86.7–95.3) | 93.0% (88.7–96.8) | 18 of 256 |
| High recall | 86.4% (80.9–91.6) | 88.5% (83.4–93.2) | 32 of 279 |

**Sensitivity, gold rubric unchanged (all gap types), SAL labels:** high precision P 64.6%, R 81.0%, F1 0.719; balanced P 60.6%, R 84.9%, F1 0.707; high recall P 56.3%, R 86.8%, F1 0.683.

## Rubric wording (final instruction only)

- **High recall:** "Be KEEP-BIASED: when you are genuinely uncertain whether a sentence is a real gap, answer YES. Only answer NO when it clearly is NOT a gap by the rules above."
- **Balanced:** "Judge each sentence on the rules above, without favouring YES or NO."
- **High precision:** "Be STRICT. Answer YES only when the sentence itself explicitly states the limitation, the proposed direction, or the unsolved problem. Also answer NO for: a general property or difficulty of a method or domain given as background or motivation; a statement of why the authors' approach is needed; an observed experimental outcome; and a passing remark that names no specific gap. When genuinely uncertain, answer NO."

## Must disclose

1. All labels here are from the Claude ensemble. Human validation is pending, so don't call 86.6% true precision yet.
2. Balanced and high-precision wording was written after reading false positives from an earlier run on these test papers. Validation gave the same pattern first: P 90.4%, R 52.5% for high precision (`docs/experiments/stage_c_rubric_val.md`). Excluding the 196 sentences used to choose the Stage C setup gives high precision F1 0.675.
3. The three modes were all run and reported. None was picked as best using test F1.
4. Rerunning high recall changed 32 of 2,522 verdicts (1.3%).
5. FWS rubric wording is the frozen rubric narrowed to future work, fixed before any FWS verdict. Dropping FWS's junk filter and lowercasing its text were decided from dev data, before test scoring.
6. SAL-40: the rubric wording was never chosen on SAL results, but the strict wording drew on gold test false positives, and 6 gold test papers are SAL papers. Most of those false positives were arXiv (428 of 455). Six rubric-and-scope runs were made on SAL; report them all.
7. The 74.5% and 94.9% readings disagree on 47 of 216 kept sentences, which LLM labels call gaps and SAL doesn't. Only the human study can settle these. PubMedBERT's selection among 8 classifiers on SAL test still applies.

## Reproduce

SAL-40: `scripts/bench/stage_c_sal40_modes.py` (requires Qwen; outputs `docs/experiments/stage_c_sal40_modes.json` and `stage_c_sal40_modes_*.jsonl`, gitignored). Gold: `scripts/bench/stage_c_rubric_test.py` makes the verdicts (requires Qwen), and `scripts/bench/stage_c_modes_report.py` recomputes every number here except the paired differences, which are in `docs/experiments/stage_c_rubric_test.json`. Outputs are gitignored and local only: `docs/experiments/stage_c_modes.json`, `stage_c_rubric_test.json`, `stage_c_rubric_test_{keep_biased,neutral,strict}.jsonl`.
