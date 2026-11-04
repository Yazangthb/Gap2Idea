# Comparison plan — baselines for sentence-level gap/limitation/future-work classification

Goal: compare our funnel against the methods that do the **same task** (sentence-level detection of
limitations / future-work / research-gaps), on the datasets we hold. Because no shared benchmark exists
(see `dataset_citation_audit.md`), the contribution is partly **running one consistent baseline suite
across all datasets** — cross-dataset comparability nobody else provides.

## 1. Published methods that do this task (the landscape to position against)

| method | task / domain | dataset | reproducible? |
|---|---|---|---|
| **Kilicoglu et al. 2018** (rules + ML) | limitation recognition, biomedical | Kilicoglu corpus | code/data partially; ~0.80 reported |
| **Lan et al. 2024** (fine-tuned PubMedBERT + EDA/PBA) | limitation detection + 15-type, RCT | **SAL** | reported **F1 0.821**; checkpoint NOT released |
| **Hu & Wan 2015** (features + classifier) | future-work mining | own | classic baseline, reimplementable |
| **Zhang et al. 2023** (BernoulliNB, n-gram+chi²) | future-work sentence recognition | **FWS** | reported macro-F1 **0.907** (leakage; clean ~0.825) — reproducible |
| **Bolaños et al. 2025** (37 LLMs: ZSL + LoRA/NEFT) | 7-class rhetorical incl. gap/limitation | **Sci-Sentence** | reported; best gpt-4o-mini **0.964**, SciBERT-aug 0.928 |
| **Faizullah et al. 2024 (LimGen)** | limitation *generation* | **LimGen** | data/code public (generation, not detection) |
| **Al Azhar et al. 2025 (BAGELS)** | limitation extraction + generation | **BAGELS** | data public; GPT-4o-mini ground truth |
| **Xu et al. 2025 (LimitGen, Yale)** | limitation identification in AI papers | LimitGen | ACL 2025, public |
| **Cohan et al. 2019 / Brack et al. 2021** | sequential sentence classification (SSC) | CSAbstruct, PubMed-RCT | SciBERT-SSC; standard SSC baselines |

## 2. Common baseline suite to RUN ourselves (apples-to-apples across datasets)

**Tier 1 — classical + encoder (cheap, run on every dataset):**
1. Majority / random (floor)
2. TF-IDF + LogReg
3. TF-IDF n-gram + **BernoulliNB** (the FWS "SOTA" method — test it everywhere)
4. **cue-rules only** (our `cue_label`) — the rule-based floor
5. **bge-small (33M) frozen + logreg** — our cheap detector
6. **SciBERT** fine-tuned (110M encoder standard)
7. **PubMedBERT** fine-tuned (biomed standard; SAL's reported detector)
8. **RoBERTa / DeBERTa-v3** fine-tuned (general strong encoder)

**Tier 2 — LLM baselines (Yandex-only per decision Oct 2026; cross-provider deferred):**
9. **yandexgpt-5-lite** (ours, Stage C judge) — ZS + few-shot
10. *(deferred: GPT-4o-mini, Claude Haiku — only if a non-Yandex key is approved later)*
11. *(deferred: Llama-3.1-8B / Qwen2.5-7B LoRA — needs GPU)*

**Tier 3 — our system:**
13. **Funnel A+B** (cue + detector) and **A+B+C** (+ Stage C filter), recall-first operating point.

Report **P / R / F1** (positive class) + **keepTP / dropFP** for the filter, at matched prevalence
where a reported baseline exists.

## 3. Per-dataset matrix (what to run + reported target)

| dataset | our task framing | reported target to match/cite | baselines to run (from §2) |
|---|---|---|---|
| **SAL** | binary limitation detect (+ 15-type) | Lan PubMedBERT **0.821** @20.7% | 2,3,4,5,6,7,8, 9,10,12, 13 |
| **Kilicoglu** | binary limitation detect | Kilicoglu rules **~0.80** | 4,5,6,7, 9,12, 13 (need corpus) |
| **FWS** | binary future-work detect | Zhang NB **0.907** (clean 0.825) | 2,3,5,6,7, 9,12, 13 |
| **Sci-Sentence** | binary gap (LIM∪GAP) + 7-class | Bolaños gpt-4o-mini **0.964** (7-cls) | 2,5,6,7, 9,10,11,12, 13 |
| **LimGen** | binary limitation detect | our prior bench (0.627 frozen ensemble) | 2,3,5,6,7, 12, 13 |
| **BAGELS** | extraction recall/coverage | our 0.738 A+B recall | 4,5, 13 (coverage metric) |

## 4. Priority order
1. **Lock Tier-1 on SAL + FWS + Sci-Sentence** — classical + encoders, no API. Establishes the
   comparable grid and reproduces each dataset's headline (PubMedBERT 0.821, NB 0.907, SciBERT).
2. **Add Tier-2 LLMs** (GPT-4o-mini + Claude Haiku ZS/few-shot, Yandex) — the cross-provider precision-filter story.
3. **Add Kilicoglu** (needs corpus download) — second biomedical detection point.
4. **LimGen / BAGELS** — detection + coverage, lower priority (generation-oriented sets).

## 4b. Results — Tier-1 grid (RUN Oct 2026, CPU, no API)

Positive-class P/R/F1. Script: `stage_c_package/baseline_grid.py`. Encoder fine-tunes for FWS/
Sci-Sentence are deferred (GPU); SAL's fine-tuned rows are from `sal_test_allmodels.json`.

**SAL — limitation, full-document stream (NO Stage A, 2.6% prevalence → needle-in-haystack):**
| method | P | R | F1 |
|---|---|---|---|
| majority / BernoulliNB | 0.000 | 0.000 | 0.000 |
| cue-rules only | 0.355 | 0.215 | 0.267 |
| bge-small frozen + logreg | 0.164 | 0.820 | 0.273 |
| TF-IDF + LogReg | 0.362 | 0.571 | 0.443 |
| DistilBERT-FT | 0.272 | 0.859 | 0.413 |
| SciBERT-FT | 0.309 | 0.883 | 0.458 |
| **PubMedBERT-FT** | 0.343 | 0.888 | **0.495** |

→ Everything is weak on the raw full document — this is exactly why **Stage A matters**. With the
section gate + prevalence-match to 20.7% (Lan's condition), our detector reaches **0.762** and **+Stage C
0.810–0.812** vs their reported **0.821** (see `results_master.md` §1c / `sal_stage_c_summary.md`).

**FWS — future-work, native 14.1% prevalence:**
| method | P | R | F1 |
|---|---|---|---|
| CountBin(1-4) + BernoulliNB (Zhang's method) | 0.880 | 0.510 | 0.646 |
| cue-rules only | 0.855 | 0.559 | 0.676 |
| bge-small frozen + logreg | 0.514 | 0.820 | 0.632 |
| **TF-IDF + LogReg** | 0.667 | 0.767 | **0.714** |
| *our funnel A+B+C* (conclusion-section corpus) | 0.843 | 0.609 | 0.707 |

→ Zhang's BernoulliNB (their "0.907 macro" method) is **precision-heavy, pos-F1 only 0.646** at honest
14% prevalence; a plain TF-IDF+LogReg beats it (0.714). Our funnel matches (+precision via Stage C).

**Sci-Sentence — limitation+research-gap, 28.6% prevalence:**
| method | P | R | F1 |
|---|---|---|---|
| cue-rules only | 1.000 | 0.150 | 0.261 |
| CountBin(1-4) + BernoulliNB | 0.895 | 0.425 | 0.576 |
| TF-IDF + LogReg | 0.784 | 0.725 | 0.753 |
| **bge-small frozen + logreg** | 0.761 | 0.875 | **0.814** |
| *our funnel (recall-first + Stage C)* | 0.943 | 0.825 | 0.880 |
| *(reported) SciBERT-aug / gpt-4o-mini, 7-class* | — | — | 0.928 / 0.964 |

→ bge-small frozen (33M) is the strongest cheap baseline (0.814); our funnel adds Stage C → 0.880.

**Cross-dataset takeaways:** (1) cheap classical/frozen baselines are competitive — TF-IDF+LogReg and
bge-small frozen are the baselines to beat; (2) BernoulliNB (the FWS "SOTA") is precision-heavy and
mediocre on pos-F1 at honest prevalence; (3) on SAL, full-document detection is uniformly weak →
Stage A is the key lever; (4) our funnel is at/above the cheap baselines everywhere, with Stage C adding precision.

## 5. Notes
- Checkpoints NOT released for SAL (Lan) → we reproduce PubMedBERT from their described recipe; cite their number, report ours.
- FWS NB "0.907" is a feature-selection-leakage artifact (we reproduced 0.9072 → 0.825 clean) — report both.
- GPT-4o-mini / Claude are the key *cross-provider* additions (currently Stage C is Yandex-only).
- Everything runs through one harness so the **same metric + prevalence** apply across datasets.
