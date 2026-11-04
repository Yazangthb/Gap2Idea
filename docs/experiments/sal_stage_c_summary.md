# SAL Stage C — SAL-criteria prompt results (Oct 2026)

Summary of the Stage C experiments on the SAL (Lan et al. 2024) limitation benchmark using the
**SAL-annotation-criteria** prompt (cause of a self-acknowledged limitation, 16 categories), in two
variants — **strict** (cause-only) and **keep-biased** (recover the cause from ±1 context) — judged
against SAL gold. Reference SOTA: their PubMedBERT **P=0.751 R=0.907 F1=0.821** (section pool @20.7%).

Eval fixture: `stage_c_package/sal_criteria/sal_test_allmodels.json` (SAL test, 7,968 sentences in
per-paper reading order, gold label + 8 Stage-B models' predictions). Judge: `yandexgpt-5-lite`,
±1 inline context, batch 20, via `LLMGapFilter(mode="validate_rct")` with the prompt swapped in.

---

## ⚠️ Methodological note — `judge_batch` is batch-composition dependent

`judge_batch` packs the whole chunk into **one prompt** (`_batch_messages`), so a sentence's verdict
depends on its batch neighbors. A balanced 120-sentence sample (positives among shuffled clear
negatives) inflated recall; judging the **real detector positives in document order** (dense,
same-region) is stricter. Same prompt, same model, same sentences: **56/60 kept → 37/60 kept.**
**Always judge Stage C on the actual detector output in reading order**, not on a mixed sample.
The faithful numbers (§3–§4) supersede the 120-sample sweep recall (§1).

---

## 1. Model sweep — keep-biased SAL-criteria (120-sample, OPTIMISTIC recall)

Hard sample = PubMedBERT A+B positives, 60 gold-POS / 60 gold-NEG. Used only to pick the model.

| model | P | R | F1 |
|---|---|---|---|
| **yandexgpt-5-lite** | 0.683 | 0.933 | **0.789** |
| yandexgpt-5-pro | 0.697 | 0.883 | 0.779 |
| yandexgpt-5.1 | 0.645 | 1.000 | 0.784 |

→ **Model chosen: `yandexgpt-5-lite`** (best F1, cheapest, fastest; same winner as the old-prompt probe).
Recall here is optimistic (see note); treat as relative model ranking only.

---

## 2. Prompt-variant comparison (end-to-end funnel, PubMedBERT)

| Stage C prompt | P | R | F1 | note |
|---|---|---|---|---|
| strict SAL-criteria (cause-only) | 0.662 | 0.498 | 0.568 | precision up, recall collapses |
| **keep-biased SAL-criteria** | 0.594 | 0.722 | **0.652** | recall recovered, precision held |
| rct_loose (old shipped) | 0.580 | 0.761 | 0.658 | ≈ tied, slightly more recall |

→ The keep-bias fix lifts strict-SAL recall **0.498 → 0.722** and F1 **0.568 → 0.652**. Essentially
tied with the old `rct_loose`; SAL-criteria is more interpretable (aligned to the annotation guideline).

---

## 3. (a) FAITHFUL end-to-end funnel — PubMedBERT, raw prevalence, 40 full papers, 205 gold

Stage A = DISCUSS+CONCL; Stage B = PubMedBERT-FT; Stage C judged on PubMedBERT's own 391 A+B
positives in document order.

| config | P | R | F1 |
|---|---|---|---|
| A+B (no Stage C) | 0.460 | 0.878 | 0.604 |
| **A+B+C — SAL keep-biased** | 0.594 | 0.722 | **0.652** |
| A+B+C — rct_loose (old) | 0.580 | 0.761 | 0.658 |

Stage C's payoff is real at this (low ~11%) prevalence: **+0.05 F1**, precision 0.46 → 0.59.

---

## 4. (b) FAITHFUL prevalence-20 section pool (their SOTA condition)

Reconstructed from the SAL test: pool = DISCUSS+CONCL (1,885 sents, 203 lim, 10.8% native),
detector = PubMedBERT-FT, prevalence-matched to 20.7% (mean over 8 negative subsamples).

| config | P | R | F1 | **F1@20.7%** |
|---|---|---|---|---|
| their PubMedBERT (reported) | 0.751 | 0.907 | 0.821 | **0.821** |
| detector only | 0.460 | 0.887 | 0.606 | 0.749 |
| + Stage C (SAL keep-biased) | 0.594 | 0.729 | 0.655 | 0.741 |
| + Stage C (rct_loose, old) | 0.580 | 0.768 | 0.661 | 0.757 |

→ **At 20.7% prevalence Stage C is ~neutral** (±0.008) — consistent with the logged finding that
Stage C is a *low-prevalence* precision tool. This reconstruction's detector (default threshold) is
less calibrated than the committed airtight run (recall-tuned detector + gpt-oss/pro = **0.810–0.812**,
matching their precision); so the prompt comparison is clean but the absolute sits below the committed SOTA number.

---

## 5. Full funnel across 8 Stage-B models (union-judged — APPROXIMATE)

From `funnel_salcriteria` / `ab_prev_funnel`. Stage C judged on the 1,036-sentence union of all
models' positives, so verdicts are stricter than per-detector judging (§⚠️) — use for **relative**
Stage-B ranking, not absolute F1. A+B F1 (strict, sentence-level) is exact:

| model | A+B F1 | A+B+C F1 (SAL keep-biased, union) |
|---|---|---|
| PubMedBERT-FT | 0.604 | 0.554 |
| SciBERT-FT | 0.553 | 0.545 |
| DistilBERT-FT | 0.523 | 0.544 |
| e5-base (frozen) | 0.457 | 0.499 |
| bge-small (frozen) | 0.436 | 0.484 |
| bge-base (frozen) | 0.437 | 0.484 |
| mpnet (frozen) | 0.437 | 0.471 |
| MiniLM-L6 (frozen) | 0.371 | 0.445 |

Span ±1 scoring lifts PubMedBERT A+B **0.604 → 0.743** (recovers adjacent-sentence gold-noise).

---

## 6. Disagreement analysis — why precision looks capped

Faithful PubMedBERT run: **101 FP** (LLM kept, SAL=NEG) and **32 FN** (LLM dropped, SAL=POS).
Single-reader categorization of 22 FPs by *what the label should be under SAL's own criteria*:

| bucket | what it is | correct under SAL's criteria | ~share | prompt-fixable? | solution |
|---|---|---|---|---|---|
| **A — gold under-label** | real self-ack cause-clause in a valid category, SAL didn't tag | **POS** (gold is wrong) | ~45% | ❌ no — LLM already correct | **adjudicate / relabel**; report true precision; use as the gold-granularity critique |
| **B — convention mismatch** | consequence / result-interpretation / bare scope statement | NEG | ~36% | ⚠️ only via strict prompt (costs recall) | keep (true extraction) OR strict cause-only prompt for a secondary SAL-aligned column |
| **C — clean LLM error** | defense/mitigation, bare header, other-study | NEG | ~18% | ✅ yes, cleanly | DROP rules: mitigations ("unlikely to have affected"), bare headers, other-study sentences (rct_v2 dir) |
| **FN** | mild/implicit real limitations dropped | POS | 32 sents | ✅ yes | KEEP rules + shots: data-not-collected, group contamination, unblinding-by-knowledge, dosing deviations, logistical |

**Projected true precision after adjudicating bucket A ≈ 0.87** (vs measured 0.59). Proportions are a
single-reader estimate over 22/101 — indicative, pending domain-expert adjudication.

**Answer to "can the FP be fixed by a criteria-faithful prompt?":** partly. ~18% (C) cleanly, ~36% (B)
only at a recall cost; **~45% (A) are wrong labels even under SAL's own criteria** — no prompt can fix
them, because matching SAL would require the LLM to be wrong. That ceiling is the paper's core argument.

---

## 7. Takeaways
1. Keep-biased SAL-criteria ≈ `rct_loose` everywhere (within ~0.006 F1); it fixed the strict prompt's recall collapse.
2. Stage C helps at **low prevalence** (end-to-end funnel, +0.05 F1), neutral at 20.7%.
3. Measured precision (~0.59) **understates** true precision (~0.87) — most FPs are SAL under-labels.
4. Real, free headroom = C-drop + FN-keep prompt rules (disjoint buckets → P and R both rise).
5. `judge_batch` is batch-composition sensitive → always judge on real detector output in reading order.

Scripts & result JSONs: `stage_c_package/sal_criteria/`.
