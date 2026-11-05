# Stage C on the full gold test split: Qwen3.8-27B (2026-11-05)

First complete A+B+C run on the 230-paper gold corpus. All 1,438 balanced candidates (τ=0.85, tuned on VAL)
from the 34 test papers, in reading order. Self-hosted Qwen3.8-27B-FP8 (vLLM, revision `017b9c7a`), keep-biased
rubric from `stage_c_prompt_vs_model.py`, context on, batch 20, no thinking, temperature 0, seed 0.
72 requests, 183 s, no failed batches. Labels are LLM gold, not human.

Script: `scripts/bench/stage_c_full_qwen.py`. Outputs (gitignored, hold gold labels):
`stage_c_full_qwen.json`, `stage_c_full_qwen_verdicts.jsonl`.

## End to end (recall against all 664 test gaps)

| | Precision | Recall | F1 |
|---|---|---|---|
| A+B balanced | 33.4% | 72.3% | 0.457 |
| **A+B+C** | **64.3%** | **68.8%** | **0.665** |
| 95% CI, papers resampled | 54.4–74.1% | 67.2–70.5% | 0.608–0.715 |

Stage C kept 457 of 480 gaps (95.2%) and removed 704 of 958 gold negatives (73.5%). The recall CI treats the gaps
lost before Stage C as a fixed share, so it is too narrow as a measure of end-to-end recall.

## Against Stage B alone on the same test gold

| Recall by type | n | A+B | **A+B+C** | ModernBERT NLI | bge-small fine-tuned |
|---|---|---|---|---|---|
| limitation | 220 | 66.8% | **61.4%** | 46.4% | 48.6% |
| future_work | 97 | 66.0% | **64.9%** | 44.3% | 83.5% |
| research_gap | 347 | 77.5% | **74.6%** | 74.1% | 77.2% |
| **F1, any gap** | 664 | 0.457 | **0.665** | 0.617 | 0.584 |

The F1 gap to ModernBERT is not established: 0.617 lies inside the CI, and no paired test is possible without
ModernBERT's per-sentence predictions, which remain on the V100 box.

## By domain

| Domain | Candidates | Gaps kept | Precision before | Precision after |
|---|---|---|---|---|
| Biomedical | 243 | 95/99 (96.0%) | 40.7% | 87.2% |
| arXiv | 1,195 | 362/381 (95.0%) | 31.9% | 60.1% |

### End to end per domain, with exact gap totals

`scripts/bench/stage_c_by_domain.py`. Intervals resample papers within the domain. Held-out excludes the 196 selection
candidates and assumes the same share of gaps reaches Stage C.

| Domain | Papers | Gaps | A+B F1 | A+B+C P / R | A+B+C F1 (95% CI) | Held-out F1 | ModernBERT F1 | bge-small FT F1 |
|---|---|---|---|---|---|---|---|---|
| arXiv CS | 28 | 527 | 0.443 | 60.1% / 68.7% | **0.641** (0.570–0.706) | 0.645 | 0.607 | 0.579 |
| Biomedical (SAL papers) | 6 | 137 | 0.521 | 87.2% / 69.3% | **0.772** (0.694–0.851) | 0.765 | 0.661 | 0.607 |

| A+B+C recall by type | arXiv CS | n | Biomedical | n |
|---|---|---|---|---|
| limitation | 55.9% | 152 | 73.5% | 68 |
| future_work | 65.8% | 73 | 62.5% | 24 |
| research_gap | 75.8% | 302 | 66.7% | 45 |

**The biomedical gold is SAL-40.** All 40 biomedical gold papers are the 40 SAL test papers (28 gold train, 6 val, 6 test),
so this is not independent of the SAL benchmark and covers RCT reports only, not biomedicine in general. A gold-trained head
has seen 34 of the 40, so any SAL-40 rerun must use the PubMedBERT candidates, not this head.

Human-label check on those 6 papers: all 243 candidates matched SAL labels, and Qwen kept **29 of 29** human-labelled
limitations. It kept 80 other sentences, but SAL labels limitations only and the rubric also keeps future work and research
gaps, so 26.6% is not a fair precision. Of 51 candidates the LLM gold calls limitation, SAL agrees on 29.

## Reading order vs shuffled (same 80 sentences, same setup)

73/80 verdicts agree. Reading order kept 18 gaps and 18 false positives; shuffled kept 19 and 16. No sign of the
recall inflation seen on SAL (56 → 37 of 60), though this compares batch mixes on only 19 gaps.

## Held out from setup selection

The setup was chosen using 196 of these candidates: the first 120 in file order (Yandex) and the seed-11 sample of 80
(Sonnet, Yandex, Qwen), which overlap by 4. `scripts/bench/stage_c_heldout.py` rescores the saved verdicts without them.
Recall assumes the same 480/664 share reaches Stage C. Intervals resample papers.

| Group | Papers | Candidates | Gaps kept | Precision | Recall | F1 (95% CI) |
|---|---|---|---|---|---|---|
| All | 34 | 1,438 | 457/480 (95.2%) | 64.3% | 68.8% | 0.665 (0.608–0.715) |
| Used for selection | 27 | 196 | 61/64 (95.3%) | 72.6% | 68.9% | 0.707 |
| **Sentences held out** | 32 | 1,242 | 396/416 (95.2%) | **63.2%** | **68.8%** | **0.659 (0.599–0.712)** |
| Whole papers held out | 7 | 142 | 67/71 (94.4%) | 85.9% | 68.2% | 0.760 (0.724–0.782) |

Use the sentence-level row as the held-out figure. The 7 papers are all arXiv, have few candidates and start at 50%
precision, because a random sample is less likely to touch small papers, so they are not representative. Held-out
sentences still share papers with the selection sentences, so this is not paper-independent.

## Limits
- LLM gold only. Some of the 254 false positives may be gaps the gold missed; the human study has to settle that.
- The rubric and model were chosen after looking at 196 candidates from this test split (see above).
- `research_gap` is half the gold and included in the rubric; its scope is still undecided.
- Biomedical has only 99 candidate gaps, so its figures are less stable.
- `LLMGapFilter._batch_messages` still adds the "own work only" instruction; this run bypasses it.
