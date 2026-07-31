# Stage B vs prior art — limitation detection on LimGen (final)

> Binary limitation-sentence detection on **LimGen** (ACL papers, CC-BY-4.0),
> the prior art's own data. One held-out test split, leakage-clean, all methods
> trained on the same LimGen-train. Metric = F1 for the limitation class.
> Source: `scripts/bench_research_single.py` (train 8000 / val 2000 / test 5876).

## Same-data comparison (we ran every row)
| method | kind | precision | recall | **F1** |
|---|---|---|---|---|
| **DistilBERT fine-tuned** *(their fine-tuned-BERT approach)* | prior art · fine-tuned | 0.55 | 0.87 | **0.674** ‡ |
| **stacking [bge + tfidf + cue]** *(OURS — shipped Stage B)* | ours · **frozen/cheap** | 0.54 | 0.76 | **0.627** |
| bge-small + logreg *(OURS — base)* | ours · frozen | 0.54 | 0.71 | 0.610 |
| TF-IDF + logreg | classical | 0.59 | 0.64 | 0.610 |
| **BernoulliNB** *(Zhang et al. 2022's method, reproduced)* | prior art · classical | 0.70 | 0.46 | 0.553 |
| cue rules only | ours · lexical | 0.57 | 0.10 | 0.167 |

‡ **DistilBERT fine-tuned reaches 0.674, but does not reliably reproduce on this
CPU box** — torch's multi-threaded CPU backward pass deadlocks randomly (completed
~2 of ~8 attempts; the number is from a completed run, same data/scale). It needs a
GPU (or the planned LLM tier) to be a dependable part of the pipeline. The
isolated/single-threaded worker is `scripts/_distilbert_worker.py` (works on GPU).

## Published numbers — different data/domain/task (context only, NOT reproducible here)
| work (venue) | task | domain | F1 |
|---|---|---|---|
| RCT / PubMedBERT (2024) | limitation detection | **biomedical** (1000s labels) | 0.82 |
| Zhang et al. (J. Informetrics 2022) | **future-work** recognition (binary) | ACL | 0.91 |
| Zhang et al. (2022) | future-work 6-way typing | ACL | 0.73 |

## What we can and cannot claim
- ✅ **We beat the reproducible prior-art on identical data.** Our cheap frozen
  ensemble (**0.627**) beats Zhang et al.'s reproduced BernoulliNB (0.553), the
  classical TF-IDF (0.610), our own base (0.610), and lexical rules (0.167).
- ✅ **We are competitive with fine-tuned BERT at a fraction of the cost.** ~0.05
  F1 behind it on large data (and ahead on small data), while staying frozen,
  CPU-only, and scalable.
- ❌ **We do not beat fine-tuned BERT outright** (0.627 vs 0.674), nor the
  published 0.82 (biomed, different domain + far more labels) or 0.91 (a different,
  easier task — future-work, not limitations). Those are not like-for-like.

## How to close the ~0.05 gap (Stage-B done; these are the next levers)
1. **LLM precision filter (planned Stage C)** — an LLM over the funnel's ~6
   survivors/paper; targets the precision side the recall-heavy fine-tune wins on,
   at near-zero cost. The intended "≥ fine-tuned BERT, far cheaper" claim.
2. **Fused stack [frozen + fine-tuned BERT]** — measured-pending: needs the
   fine-tune to run (GPU). Expected to top 0.674 by combining BERT's recall with
   the cues' precision.
3. **More data + 2-epoch fine-tune** — DistilBERT rose 0.629→0.674 from 2K→8K
   train; full LimGen + more epochs should push higher (GPU).
