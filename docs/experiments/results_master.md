# Gap2Idea — Master Results Log

> Durable archive of **every measured result**, with notes/caveats, so we can return to any
> experiment later. This file covers the SOTA-benchmark + encoder + Stage-C work (Aug 2026).
> For the **older extraction-gold** experiments (Stage A/B on the 19-gap gold: tables T1–T*),
> see [results_registry.md](results_registry.md) and [experiment_log.md](experiment_log.md).
> Per-topic write-ups: [rct_stagec.md](rct_stagec.md), [fws_recognition.md](fws_recognition.md),
> [limgen_stagec.md](limgen_stagec.md), [limgen_enhanced.md](limgen_enhanced.md),
> [bagels_output.md](bagels_output.md).

**How to read.** F1 = positive-class F1 unless "macro". "Stage C" = batched LLM precision filter
(`validate_rct`/`validate_fws` modes). Runner scripts in `scripts/bench/`, `scripts/training/`.

---

## 0. One-screen summary (the keepers)

| Finding | Evidence | Note |
|---|---|---|
| **Data not model** (fine-tuned) | RCT & FWS: bge-small(33M) ≈/> PubMedBERT/SciBERT(110M) | holds only *after* fine-tuning |
| **Frozen encoders DON'T converge** | RCT frozen: SPECTER2 0.346 > bge 0.263 > S-PubMedBert 0.25 | scientific pretraining wins when frozen |
| **Stage C = low-prevalence precision tool** | +0.05–0.11 F1 at 2–3% prevalence; ~neutral at 14–21% | asymmetric: raises P, small R cost |
| **FWS SOTA 0.907 is a leakage artifact** | reproduced 0.9072; chi² in-fold → 0.8252 (+0.082) | honest ceiling ~0.85 |
| **RCT: we match precision, not recall** | our +C P 0.75–0.78 vs their 0.751; F1 0.81 vs 0.821 | gap is detector recall, not method |
| **Raw ACL limitations ~54% clean** | Stage C on 80-sample: 43/80 genuine | Stage C = phase-1→2 bridge |
| **Residual FPs = label noise** | RCT & FWS error analysis | caps measurable precision on both |

---

## 1. RCT / SAL — limitation detection (Lan et al. 2024)  ·  Aug 13–15

Data: `MengfeiLan/SAL_Type_Classification` (43,208 sents, 200 papers, 952 limitations, 13 coarse
types). **Their reported SOTA: PubMedBERT P=0.751 R=0.907 F1=0.821** on a section-filtered pool
(~20.7% prevalence). Their checkpoint is NOT released — we train our own detector.
Scripts: `bench_rct.py`, `build_rct_pool.py`, `training/finetune_rct.py`, `bench_stagec_opt.py`.

### 1a. Full-document stream (2.6% prevalence, test=7,968; needle-in-haystack)
| Config (PubMedBERT-base fine-tuned) | P | R | F1 |
|---|---|---|---|
| 1:6 undersample, max-F1 thr | 0.43 | 0.81 | 0.561 |
| &nbsp;&nbsp;+ Stage C | 0.55 | 0.76 | 0.637 |
| 1:6, recall-tuned thr | 0.32 | 0.90 | 0.468 |
| &nbsp;&nbsp;+ Stage C | 0.47 | 0.81 | 0.593 |
| full-neg (1:40), max-F1 | 0.55 | 0.54 | 0.545 |
| &nbsp;&nbsp;+ Stage C | 0.62 | 0.52 | 0.568 |

*Note:* full-stream caps ~0.55–0.64 regardless — that's the hard framing. More negatives raised
precision but lowered recall (net flat). Stage C lifts F1 here (+0.05–0.13) — its low-prevalence home.

### 1b. Prevalence-matched to 20.7% — **random** negatives (optimistic)
| Config | F1 |
|---|---|
| PubMedBERT max-F1 | 0.846 |
| PubMedBERT recall-tuned | **0.881** |
| recall-tuned + Stage C | 0.869 |

*Note:* beats 0.821 but with EASY random negatives → inflated. Superseded by 1c.

### 1c. Section-filtered pool (their condition) — **hard** section negatives  ·  the airtight number
Pool built via PMC BioC section types (ABSTRACT/DISCUSS/CONCL); **99% of gold positives retained**;
2,496 sents, 8.1% native prevalence. Evaluated at 20.7% (subsampled):
| Method (@20.7%, hard negs) | P | R | F1 |
|---|---|---|---|
| Their PubMedBERT (reported) | 0.751 | 0.907 | 0.821 |
| Our detector only | 0.668 | 0.887 | 0.762 |
| + Stage C (gpt-oss-120b) | 0.752 | 0.882 | 0.812 |
| + Stage C (yandexgpt-5-pro) | 0.780 | 0.842 | 0.810 |

**Verdict: we match their precision; the ~0.01 F1 gap is recall (detector ceiling 0.887 vs their
0.907). Stage C can't add recall (only removes).**

### 1d. Stage C model × prompt (section pool @20.7%)
| Config | keptTP | F1@20.7% |
|---|---|---|
| yandexgpt-5-pro / rct_loose | 172/180 | 0.810 |
| gpt-oss-120b / rct_loose | 179/180 | 0.812 |
| gpt-oss-120b / rct_v2 | 177/180 | 0.815 |
| qwen3-235b / rct_loose | 177/180 | 0.801 |
| deepseek-v4-flash / rct_loose | 178/180 | 0.801 (100× slower) |

*Notes:* `rct_loose` (KEEP-biased, shipped as `validate_rct`) preserves recall; `drop-strict`
collapses recall. `rct_v2` no real gain — **residual FPs are gold-label noise** (real limitations
not annotated), so precision is label-capped. deepseek-v4-flash works but impractically slow.

### 1e. Frozen Stage-B encoder on RCT (test 3,000; SPECTER2 study, Aug 19)
| Frozen encoder + logreg | Params | Pooling | F1 |
|---|---|---|---|
| S-PubMedBert-MS-MARCO | 110M | mean | 0.250 |
| bge-small | 33M | mean | 0.263 |
| SPECTER2-base | 110M | **mean** | **0.346** |
| SPECTER2-base | 110M | CLS | 0.294 |

*Notes:* SPECTER2 (scientific citation-triplet pretraining) is the **best frozen encoder** (+0.08
over bge). CLS < mean because SPECTER2's CLS is calibrated for its **proximity adapter** (not
loaded; needs `adapters` lib). Frozen ≪ fine-tuned (0.35 vs 0.82) — encoder choice is secondary.
Earlier CPU fine-tune of bge-small reached 0.437 (+C 0.576).

---

## 2. FWS — future-work recognition (Zhang et al. 2022)  ·  Aug 15

Data: `xiangyi-njust/FWS` (64,893 sents, 9,013 ACL papers, 13.9% future-work, section labels).
**Reported SOTA: BernoulliNB macro-F1 0.9073.** Scripts: `bench_fws.py`, `bench_fws_cheap.py`,
`bench_fws_balanced.py`.

### 2a. Reproduction + leakage audit (their protocol: balanced 1:1, lemmatized, chi²-14k, 10-fold CV)
| Pipeline | Macro-F1 |
|---|---|
| Their reported | 0.9073 |
| **Reproduced** (chi² on full data, then CV — their code) | **0.9072** |
| **Proper** (chi² refit inside each fold) | **0.8252** |
| Leakage inflation | **+0.082** |
| Our tuned NB (1–4gram, α=1e-5) leaky | 0.9105 (beats, but leaky) |

**Their SOTA reproduces exactly and is a feature-selection-leakage artifact.**

### 2b. Clean balanced split (single held-out, chi² on train only)
| Model | Params | Macro-F1 |
|---|---|---|
| **bge-small (fine-tuned)** | 33M | **0.879** |
| PubMedBERT (fine-tuned) | 110M | 0.877 |
| SciBERT (fine-tuned) | 110M | 0.864 |
| BernoulliNB (1–4, chi²) | — | 0.830 |
| TF-IDF word+logreg | — | 0.827 |
| shipped cue+head (zero-shot) | 33M | 0.731 |

*Note:* on a CLEAN split the cheap NB collapses 0.905→0.830 (confirms the leakage); encoders win;
bge-small(33M) tops the 110M models.

### 2c. Honest imbalanced held-out (14.3%, paper-id split) + Stage C
| Detector | stage | pos-F1 | Macro-F1 |
|---|---|---|---|
| shipped cue+head (zero-shot) | detector | 0.607 | 0.774 |
| | + Stage C | 0.624 | 0.786 |
| SciBERT ft | detector | 0.746 | 0.853 |
| | + Stage C | 0.740 | 0.850 |
| PubMedBERT ft | detector | 0.750 | 0.853 |
| | + Stage C | 0.748 | 0.853 |
| bge-small ft | detector | 0.741 | 0.850 |
| | + Stage C | 0.737 | 0.849 |
| cheap: wordchar+SVM | — | 0.736 | 0.847 |
| cheap: BernoulliNB | — | 0.685 | 0.819 |

*Note:* at 14% prevalence everything compresses ~0.85 (macro dominated by easy negatives); Stage C
neutral on strong detectors, +0.012 macro on the weak zero-shot one.

### 2d. Stage C prompt optimization (FWS, shipped detector predictions)
| Prompt | dropTP | pos-F1 | Macro-F1 |
|---|---|---|---|
| validate_fws (initial) | 18 | 0.624 | 0.786 |
| **keep-biased (shipped)** | 7 | **0.633** | **0.791** |
| drop-strict | 176 | 0.496 | 0.720 |

*Note:* keep-biased recovers weak/implicit future-work (18→7 dropped-TP) for +2 FP. `drop-strict`
proves precision is label-capped (139 kept-FPs are mostly real future-work the gold marks 0).
Shipped into `validate_fws`. Caveat: picked on test predictions.

---

## 3. Cross-dataset & infrastructure notes

**Model sizes:** PubMedBERT-base 109.5M, SciBERT 109.9M, bge-small 33.4M, SPECTER2-base 110M.

**Yandex models available (probed Aug 14):** ✅ yandexgpt, yandexgpt-32k, yandexgpt-lite,
**yandexgpt-5-pro**, yandexgpt-5-lite, **gpt-oss-120b**, gpt-oss-20b, **qwen3-235b-a22b-fp8**,
**deepseek-v4-flash** (slow, reasoning). ❌ deepseek-r1/v3, llama (not provisioned).
Reasoning models need `extra_body` (gpt-oss `reasoning_effort:low`; qwen `enable_thinking:false`)
+ larger token budget (baked into `judge_batch`).

**Compute:** GPU experiments on a single Tesla V100-SXM2-32GB (remote, since torn down). Frozen /
cheap / Stage-C runs are CPU-local. bge-small fine-tune ~57s on V100; full-neg ~452s.

**Data quality (phase-1→2 bridge):** raw ACL limitations harvest = 6,433 sents, **Stage C judges
54% genuine authors'-own limitations** (rest: contributions, results, methods, citations, math,
field-observations). RCT limitations = clean/human but trial-*execution* issues (small sample,
underpowered, no blinding), NOT open research problems → wrong domain for idea generation.

---

## 4. Open threads / to-revisit

- **SPECTER2 + proximity adapter** (canonical config, `pip install adapters`) — likely > 0.346
  frozen; the real SPECTER2 number is still unmeasured. Also the right Phase-2 embedder.
- **Confirm frozen-encoder ranking on FWS** (only RCT tested).
- **RCT Stage-C prompt** picked on test → redo on a dev split (rigor).
- **Phase 2 (ideation)** — unbuilt. Next: Stage-C-clean an ACL/arXiv corpus → cluster → generate.
- **BAGELS / LimGen** — whole-section labels are noisy for sentence-F1; see per-topic docs.

---
_Last updated: 2026-08-19. Numbers above are from committed runs (`git log`: 9f6c5f0, c134bb8,
42e70ae, 8b9eac5, eb3cff5, d88a552) plus the Aug-19 SPECTER2 frozen-encoder study._
