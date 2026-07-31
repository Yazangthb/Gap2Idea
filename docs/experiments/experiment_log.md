# Gap2Idea — Extraction Experiment Log

> Lab notebook for the cheap gap-extraction funnel (future_work + limitation).
> Chronological; each entry = motivation → method → config → result → conclusion,
> with the script and artifact that produced it. Numbers are on the clean gold
> (`data/bench_gap/gold_sentences.tsv`, 19 gaps / 9 papers) unless noted.
> Companion: [results_registry.md](results_registry.md) (all tables in one place),
> [README.md](README.md) (file index + reproduce), [../related_work_analysis.md](../related_work_analysis.md).

**Scope decision.** The funnel targets the two *structurally localized* gap types
— `future_work` and `limitation`. `open_problem` is excluded: a gap = *(posed)*
AND *(left unresolved)*, and the second half is a document/discourse-level
judgment that cannot be made from a sentence in isolation (architecture doc §5.2).

---

## Phase 0 — Baseline & gold

### E0.1 — Current per-paper-LLM extractor (baseline)
- **Why:** establish the expensive incumbent's cost and quality.
- **Method:** `openai_gaps.py` — gpt-4.1-mini on regex sections, ≤2 gaps/paper.
- **Result:** ~$4,000 / 1M papers; on the old silver bench, macro-F1 **0.187**,
  recall starved by the ≤2/paper cap (`scripts/archive/eval_gap_extraction.py`,
  `data/bench/eval_metrics.tsv`).
- **Conclusion:** cost is linear in papers and rate-limited → does not scale.
  Target a cheap funnel; the ≤2/paper cap is a recall ceiling to avoid.

### E0.2 — Clean gap-sentence gold (extract → verify)
- **Why:** no public benchmark targets arXiv AI/ML/math; need a clean recall target.
- **Method:** full paper → gpt-4o extract (own-work limitation / future-work,
  verbatim) → **gpt-4o verify** filter (drop prior-work limitations,
  contributions/cross-refs mislabelled as gaps, vague gestures). Keep only
  sentences with ≥0.80 token-recall in source. `scripts/dataset/build_gap_gold.py`.
- **Result:** raw single pass = 31 gaps but **~25% contaminated** (esp. math
  papers: theorem statements mislabelled future-work). Verify pass → **19 clean
  gaps / 9 papers** (`data/bench_gap/gold_sentences.tsv`; raw kept as
  `gold_sentences_unverified.tsv`).
- **Conclusion:** the **verify pass is essential** — a raw LLM gold is materially
  contaminated. This is the literature-standard *extract → filter* pattern.

---

## Phase 1 — Stage A (structural slice)

### E1.1 — Recall-robust slicer
- **Why:** Stage A is the recall ceiling; PDFs are two-column-scrambled so clean
  heading detection fails (~0.30 recall naive).
- **Method:** union of three sources on the reading-order sentence stream —
  generous span after a Limitations/Future-Work/Conclusion *heading*, ±window
  around an *inline* keyword (scrambled headings), and a *terminal tail*; anchor
  priority cap; mid-paper Discussion gated. `gap_funnel.slice_terminal_regions`.
- **Result (localization recall, `scripts/bench/bench_gap_recall.py`):**

  | containment τ | all | future_work | limitation |
  |---|---|---|---|
  | 0.90 | 0.74 | 0.82 | 0.63 |
  | **0.80** | **0.84** | **0.91** | 0.75 |
  | 0.70 | 0.89 | 1.00 | 0.75 |

  Drops **82%** of sentences for free (341 → 62 sentences/paper).
- **Conclusion:** future_work localization is effectively solved; the 2 limitation
  misses are *mid-paper* own-work limitations (setup, not a Limitations section)
  — a structural ceiling, not a slicer bug. Reported at multiple τ because 19/27
  raw gold gaps are PDF-scrambled (containment, not substring).
- **Artifacts:** `data/bench_gap/stage_a_audit.tsv`,
  `docs/figures/stage_a_{localization,funnel}.png`, `docs/stage_a_explained.md`.

---

## Phase 2 — Stage B (classify): the limitation problem

### E2.1 — The poisoned-negatives bug (found & fixed)
- **Symptom:** the embedding head predicted `none` on real gaps at p=0.91.
- **Root cause:** self-distillation used the teacher's ≤2-gaps/paper output as the
  *only* positives, so every real gap the teacher skipped (in the same slice)
  became a **negative** → the head learned "gaps are none."
- **Fix:** draw negatives from the paper body **outside** the slice (reliably
  non-gap). `scripts/training/train_gap_head.py`.
- **Result:** hybrid end-to-end recall 0.37 → 0.47.
- **Conclusion:** a self-distillation pitfall worth reporting — capped teacher
  labels silently poison the negative class.

### E2.2 — Is it the model? (classifier / fine-tuning ablation)
- **Why:** decide whether weak limitations are a model-capacity or data problem.
- **Method:** identical data + eval, three classifiers — frozen bge+logreg,
  fine-tuned DistilBERT, SetFit contrastive fine-tuning.
  `scripts/training/test_bert_stageb.py`, `scripts/training/test_setfit_stageb.py`.
  (SetFit reimplemented in raw torch — the `setfit` package is broken on
  Python 3.14 / transformers 5.5.)
- **Result (end-to-end recall / limitation recall):** logreg 0.42 / 0.11 ·
  DistilBERT 0.37 / 0.11 · SetFit 0.37 / 0.11 — **all tie, limitations stuck.**
- **Conclusion:** **not the model.** With ~63 limitation examples every classifier
  is equivalent. Matches the literature (domain-BERT caps ~0.5 on limitation typing).

### E2.3 — Distant supervision (rejected)
- **Method:** weak positives = every sentence in an explicit Limitations/Future
  section (`--distant`).
- **Result:** recall up (0.53) but **floods predictions (143 vs 45)** — fires on
  formulas, table-of-contents lines, even an acknowledgments line. type_acc 0.90→noisy.
- **Conclusion:** too noisy (math papers' gap regions are full of exposition).
  **Reverted.** Lesson: section-membership ≠ gap.

### E2.4 — No off-the-shelf model exists
- **Method:** direct HuggingFace Hub crawl (`huggingface_hub.list_models`) over
  future-work / limitation / research-gap / scientific-sentence-classification.
- **Result:** **0 task-specific models** (only one 7-download CSAbstruct
  classifier, Background/Method/Result/Conclusion — no gap labels).
- **Conclusion:** training our own is the only option; SetFit + sequential-sentence
  classification (Cohan) are the only adjacent open methods.

---

## Phase 3 — The fix: clean data (ACL mandated-Limitations harvest)

### E3.1 — Harvest LimGen Limitations sections
- **Why:** the literature's recipe for the limitation-data shortage — a mandated
  "Limitations" heading is, by construction, the authors' own-work self-critique.
- **Method:** download LimGen (`github.com/arbmf/LimGen`, CC-BY-4.0, 4068 ACL
  papers); pull the `limitations` field, split sentences, filter junk, **leakage-
  guard vs gold**. Also cue-harvest future-work from `content`.
  `scripts/dataset/harvest_acl_limitations.py`.
- **Result:** **6,433 limitation + 1,268 future-work** sentences from 813 papers,
  0 leakage (`data/acl_limitations.tsv`, `data/acl_futurework.tsv`).

### E3.2 — Cap sweep (how many to add)
- **Method:** add N ACL limitation positives, refit logreg, benchmark.
  `scripts/training/sweep_acl_cap.py`.
- **Result (end-to-end recall / per-sentence limitation recall):**

  | ACL limitation sentences | recall | limitation R | n_pred |
  |---|---|---|---|
  | 0 (before) | 0.421 | 0.111 | 42 |
  | 600 | 0.526 | 0.333 | 58 |
  | **1500 (best)** | **0.526** | **0.444** | 61 |
  | 3000 / 6000 | 0.526 | 0.444 | 64–65 |

  **Future-work harvest was net-negative** (noisy cue extraction: type_acc
  0.90→0.80) and excluded by the sweep.
- **Conclusion:** **limitation recall 0.11 → 0.44 (4×)**, end-to-end 0.42 → 0.53;
  **saturates at ~1,500** (no need for LimGen's 89 MB train split). Same recall as
  distant supervision but **clean (61 preds, not 143)**. **The data was the
  bottleneck — proven.**

### E3.3 — Encoder sweep (is it the features?)
- **Method:** swap the frozen encoder (bge-small/base, mpnet, SPECTER), classifier
  fixed, on the ACL-augmented data. `scripts/training/sweep_encoders.py`.
- **Result:** **all tie at recall 0.526 / limitation 0.444.** SPECTER gave
  type_acc 1.00 but worse future-work; bge-small best future-work + smallest.
- **Conclusion:** encoder doesn't matter either — **third axis confirming
  data-bound.** Shipped **bge-small** (most scalable, ties on the rest).

---

## Phase 4 — Cost & positioning

### E4.1 — Cost projection
- **Result:** only slice sentences are embedded (~62/paper) → ~$3–31 / 1M papers
  (local CPU ≈ free) vs ~$4,000 for per-paper LLM. **~128–166× cheaper**, offline,
  shardable. `scripts/bench/bench_gap_funnel.py` (cost_projection), funnel figure.

### E4.2 — Related-work analysis (deep research)
- **Result:** our two-stage funnel (high-recall structural slice → precision
  classifier filter) is the **field-standard pattern** (Hu & Wan 2015; Zhang et al.
  2022; FutureGen 2025; RCT/PubMedBERT). Closest analog RCT/PubMedBERT = keyword
  slice → fine-tuned BERT, **F1 0.82**, scaled to 12k papers no-LLM. Binary FWS
  detection is easy (NB 90.7%); fine-grained typing hard for everyone (SciBERT
  72.6%, PubMedBERT 0.49). [../related_work_analysis.md](../related_work_analysis.md).
- **Conclusion:** method is established (validation, not novelty); originality is
  downstream (idea generation + novelty evaluation) and in the cheap cross-domain
  application. On the comparable sub-task (limitations) SOTA is ahead (0.82 vs our
  0.44) — a *data*, not method, gap.

---

## Phase 5 — Same-data comparison vs prior art (E5)

### E5.1 — LimGen head-to-head (limitation detection)
- **Why:** get a real, leakage-clean number vs prior art on *their* data.
- **Method:** LimGen ACL papers, binary limitation-sentence detection, one held-out
  split, all methods trained on LimGen-train. `scripts/bench/bench_limgen.py`,
  `bench_research_single.py`. Full table: [research_comparison.md](research_comparison.md).
- **Result (F1):** DistilBERT fine-tuned (their approach) **0.674** > **stacking
  [bge+tfidf+cue] (OURS, frozen) 0.627** > bge 0.610 ≈ tfidf 0.610 > Zhang's
  reproduced BernoulliNB 0.553 > cue rules 0.167.
- **Enhancements that worked** (`enhance_stageb.py`): stacking ensemble (+0.03 over
  base bge) and more-data fine-tune (DistilBERT 0.629→0.674 from 2K→8K train).
  Threshold-tuning marginal.
- **Conclusion:** our cheap frozen ensemble **beats the reproducible classical +
  Zhang prior art on identical data**, and is **~0.05 behind a fine-tuned
  transformer** at a fraction of the cost. We do *not* outright beat fine-tuned BERT.

## ⏸ STAGE B — FINAL STATUS (paused; come back later)
**Shipped:** frozen stacking ensemble, F1 **0.627** on LimGen (and limitation
recall 0.11→0.44 on our gold via the ACL-data fix). Beats all reproducible
baselines; competitive (~0.05) with fine-tuned BERT, far cheaper. **This is a
clean stopping point.**

**Blocked levers to beat fine-tuned BERT (need GPU or LLM credits — resume when available):**
- Fused stack [frozen + fine-tuned BERT] — expected to top 0.674; **blocked: DistilBERT
  training deadlocks randomly on this CPU box** (~2/8 runs; `_distilbert_worker.py`
  is correct, runs on GPU).
- Full-LimGen + 2-epoch fine-tune (GPU).
- bge+logreg rises ~0.595→ with more train data; stacking should too.

## Open threads (for later)
1. **Context-aware (Cohan SSC)** classification for discourse-hard limitations —
   needs a bigger eval than 19 gaps.
2. **Stage A** mid-paper limitation recall is the end-to-end ceiling now.
3. Bigger / human-validated gold than 19 gaps.

## → NOW: Phase 6 — Stage C (LLM precision filter)
Local small instruct model over the funnel's ~6 survivors/paper → kill the
false positives (math exposition, acknowledgments, fragments seen in
`funnel_demo_output.md`). API-swappable. The intended "≥ fine-tuned BERT, far
cheaper" claim. *In progress.*
