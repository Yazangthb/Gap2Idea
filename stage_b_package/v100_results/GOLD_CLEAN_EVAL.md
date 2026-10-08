# Clean-gold re-evaluation (limitation + future_work only)

The old gold v2 mixed in scope/assumption/open_problem (gap types the pipeline doesn't target), which
made `recall_all` meaningless. We removed them and **re-extracted a fresh gold** over the same papers'
full text, keeping only the two classes we care about.

## New gold — built by **Claude Sonnet agents** (`data/bench_gap/gold_lf.tsv`, `data/external/gold_eval_lf.json`)
- 10 parallel Sonnet subagents, one per paper, each reading the paper and extracting ONLY limitation /
  future_work (explicitly rejecting scope / assumption / open-problem / prior-work), writing verbatim sentences.
- Each sentence verified verbatim-present (token-containment ≥ 0.85) → **0 dropped, no hallucinations**.
- **19 gaps / 9 papers** — future_work 10, limitation 9.
- Coverage caveat: the 3 largest papers (~200–250k chars) were **not read in full** by their agents
  (they grepped / read sections) → likely under-counts gaps there. Sonnet was also stricter than gpt-4o-mini.

### Gold source comparison (same 10 papers, limitation+future_work only)
| gold builder | gaps | reachable in Stage-A | MiniLM rules+head recall_all | lim | fw |
|---|---|---|---|---|---|
| gpt-4o-mini (full-text, 40k cap) | 33 | 23/33 (70%) | 0.455 | 0.500 | 0.421 |
| **Claude Sonnet agents** | 19 | 14/19 (74%) | **0.632** | 0.444 | 0.800 |

→ The absolute recall swings with the gold (0.455 vs 0.632): the benchmark is **small and gold-dependent**.
Sonnet's gold is stricter/cleaner but under-covers big papers; gpt-4o-mini's is larger but noisier.

## Evaluation on the Sonnet gold (frozen head, trained on Sci-Sentence + FWS + SAL)
| model | recall_all | limitation | future_work | kept |
|---|---|---|---|---|
| **MiniLM + logreg (rules+head)** | **0.632** | 0.444 | 0.800 | 167 |
| bge-small + logreg (rules+head) | 0.632 | 0.444 | 0.800 | 102 |
| MiniLM / bge head-only | 0.579 | 0.444 | 0.700 | 157 / 91 |

`recall_all` is a valid metric here (all gold gaps are target classes).

## Key finding — Stage A is the bottleneck (both golds)
Only **~70–74%** of the clean gaps are reachable in the Stage-A slice; Stage A drops the rest before Stage B
runs. On the Sonnet gold the ceiling is 14/19 = 0.74, and rules+head hits 0.632 — so most of the remaining
loss is **Stage-A localization, not the classifier**. The biggest lever now is Stage A.

## Why lower than the old 0.571 (not a regression)
1. Removed incidental scope/assumption recoveries that inflated the old number.
2. Harder, more complete gold (33 vs 25 target gaps; subtler future_work).
3. Lower Stage-A reachability (70% vs 86% on the old set).

## Honest headline
**limitation recall 0.50, future_work recall 0.42, overall 0.455 — capped at 0.70 by Stage A localization.**
MiniLM frozen remains the best head. Next lever: improve Stage-A reachability (currently loses 30% of real gaps).

## Follow-ups
- Re-run the extraction with gpt-4o (needs OpenRouter credits) + raise the 40k cap to recover `2002.09564` and any long-paper gaps.
- A short human audit of the 33 labels to graduate this from silver to a defensible benchmark.
- Investigate the 10 unreachable gaps → Stage-A improvement (the real ceiling).
