# Gap2Idea — Results Registry

> Every measured number in one place, paper-ready. Eval = clean gold
> `data/bench_gap/gold_sentences.tsv` (**19 gaps / 9 papers**) unless noted;
> ±1–2 gaps is within noise. Recall = end-to-end gap recall; limitation/future R
> = per-sentence recall on the Stage-A slice. See [experiment_log.md](experiment_log.md)
> for methods. Reproduce: [README.md](README.md).

## T1 — Headline: Stage B before vs after the ACL-data fix *(shipped)*
| Stage B head | end-to-end recall | preds/paper | **limitation R** | future-work R | type acc |
|---|---|---|---|---|---|
| rules only | 0.32 (6/19) | 2.2 | — | — | 1.00 |
| bge+logreg (before) | 0.42 (8/19) | 4.2 | 0.11 | 0.47 | 1.00 |
| **bge+logreg + 1500 ACL limitations (shipped)** | **0.53 (10/19)** | 6.1 | **0.44** | 0.47 | 0.90 |

## T2 — Stage A localization recall (the ceiling) + load
| containment τ | recall all | future_work | limitation |
|---|---|---|---|
| 0.90 | 0.74 | 0.82 | 0.63 |
| **0.80** | **0.84** | **0.91** | 0.75 |
| 0.70 | 0.89 | 1.00 | 0.75 |

Load: 341 → **62** sentences/paper kept (**82% dropped, free**). Emitted gaps ≈ 6.1/paper.

## T3 — Model does NOT matter (classifier / fine-tuning ablation, before ACL data)
| Stage B method | end-to-end recall | preds | limitation R |
|---|---|---|---|
| rules only | 0.32 | 22 | — |
| bge + logreg (frozen) | 0.42 | 45 | 0.11 |
| DistilBERT (full fine-tune) | 0.37 | 44 | 0.11 |
| SetFit (contrastive) | 0.37 | 28 | 0.11 |
| DistilBERT + *distant data* | 0.53 | **143** | 0.22 |

→ All clean-data methods tie; only *distant data* moved recall, by flooding preds.

## T4 — ACL limitation cap sweep (clean data is the lever)
| ACL limitation sentences added | recall | limitation R | preds |
|---|---|---|---|
| 0 | 0.421 | 0.111 | 42 |
| 600 | 0.526 | 0.333 | 58 |
| **1500 (best, saturates)** | **0.526** | **0.444** | 61 |
| 3000 | 0.526 | 0.444 | 65 |
| 6000 | 0.526 | 0.444 | 64 |

Future-work harvest (`fut_cap=1268`) was net-negative on every row (type_acc → 0.80) → excluded.

## T5 — Encoder does NOT matter either (frozen encoder sweep, ACL-augmented)
| encoder | recall | limitation R | future-work R | type acc | params |
|---|---|---|---|---|---|
| **bge-small-en-v1.5 (shipped)** | 0.526 | 0.444 | **0.467** | 0.90 | 33M |
| bge-base-en-v1.5 | 0.526 | 0.444 | 0.400 | 0.90 | 109M |
| all-mpnet-base-v2 | 0.526 | 0.333 | 0.400 | 0.90 | 110M |
| allenai-specter (scientific) | 0.526 | 0.444 | 0.400 | **1.00** | 110M |

## T6 — Cost & scale (per 1M papers)
| Approach | $/1M papers | wall clock |
|---|---|---|
| Per-paper LLM (gpt-4.1-mini, ~9K tok) | ~$4,000 | days–weeks (rate-limited) |
| **Funnel (Stage A free + bge embed ~62 sents/paper)** | **~$3–31** (local CPU ≈ $0) | hours, offline, shardable |

→ ~128–166× cheaper. Stage A drops 82% before any model runs.

## T7 — Literature (for the comparison; NOT same dataset/metric)
| System (venue) | task | method | reported |
|---|---|---|---|
| Zhang et al. 2022 (J. Informetrics) | future-work recognise | Naive Bayes | Macro-F1 **0.91** |
| Zhang et al. 2022 | future-work 6-way type | SciBERT | wF1 **0.73** |
| RCT/PubMedBERT (PMC11807350) | limitation detect | keyword slice → PubMedBERT | **F1 0.82** (P.75/R.91) |
| Hu & Wan 2015 (arXiv) | future-work extract | regex | "high P/R" |
| FutureGen 2025 (arXiv) | future-work generate | regex → **LLM filter** | per-paper-LLM |
| **Ours** | end-to-end gap retrieval | slice → rules + head | recall 0.53, limitation 0.44 |

Caveat: ours is end-to-end retrieval from scrambled full PDFs on a 19-gap
cross-domain gold; theirs is sentence classification on large in-domain human
labels. Not directly comparable — see [experiment_log.md](experiment_log.md) E4.2.

## Trained heads (artifacts)
| file | what |
|---|---|
| `data/gap_head.joblib` (+ `.meta.json`) | **canonical default** — bge-small + 1500 ACL limitations |
| `data/gap_head_before_acl.joblib` | ablation "before" — bge-small, no ACL data |
| `data/gap_head_acl.joblib` | intermediate (ACL cap 600), superseded |
| `data/gap_head_bge.joblib` | early distant-supervision experiment, superseded |
