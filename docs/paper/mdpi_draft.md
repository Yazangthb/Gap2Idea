<!--
MDPI-style working draft — Gap2Idea.
Convention in this file:
  ✅ = done & measured (real numbers from our experiments)
  ⟨PENDING: ...⟩ = not yet done; blank to fill as the work is completed
Front matter (authors/affiliations/funding) are placeholders.
-->

# From Research Limitations to Research Ideas: A Two-Phase Retrieval-Augmented System for Mining and Operationalizing Self-Acknowledged Gaps in the Scientific Literature

**Authors:** ⟨PENDING: Author 1, Author 2, …⟩
**Affiliation:** ⟨PENDING: Department, University⟩
**Correspondence:** ⟨PENDING: email⟩

---

## Abstract

The volume of scientific literature makes it infeasible for researchers to manually track the
research gaps—self-acknowledged limitations and stated future work—reported across a field.
We present **Gap2Idea**, a two-phase system that (i) **extracts** gap statements from papers at
scale and (ii) **operationalizes** them into ranked, evidence-grounded research ideas.
Phase 1 is a cheap, scalable extraction funnel: a structural section slice (Stage A), a
lightweight embedding-plus-cue detector (Stage B), and a batched large-language-model (LLM)
precision filter (Stage C). We benchmark Phase 1 against two published sentence-classification
SOTAs. On self-acknowledged limitations in randomized controlled trials (RCT/SAL), our
general-purpose detector plus Stage C **matches the reported precision (0.75–0.78 vs. 0.751)
and reaches F1 ≈ 0.815 vs. the reported 0.821** on a faithful reconstruction of the original
evaluation condition. On future-work-sentence recognition, we **reproduce the reported
macro-F1 of 0.907 exactly (0.9072) and show it to be a feature-selection-leakage artifact**:
refitting selection inside cross-validation folds drops the same model to 0.825, and under
honest evaluation a 33 M-parameter general encoder matches or exceeds 110 M domain-specific
transformers. Across datasets we find that (a) model capacity and domain-pretraining wash out
after fine-tuning ("data not model"), and (b) the LLM precision filter is a *low-prevalence*
tool—valuable on the realistic full-document stream, roughly neutral on curated pools.
Phase 2 (ideation) is designed and specified here; empirical results are ⟨PENDING⟩.
**Main quantitative results for Phase 2: ⟨PENDING⟩.**

**Keywords:** research gap mining; self-acknowledged limitations; future work sentences;
scientific idea generation; retrieval-augmented generation; text classification;
reproducibility; scientometrics.

---

## 1. Introduction

Scientific papers routinely state their own **limitations** and propose **future work**. In
aggregate these "self-acknowledged gaps" are a map of what a field knows it cannot yet do—an
ideal substrate for surfacing open problems and generating research directions. Two obstacles
have kept this from being exploited at scale: gaps must first be **extracted** reliably and
cheaply from millions of documents, and they must then be **operationalized**—normalized,
de-duplicated, scored, and turned into concrete, novel, feasible ideas—rather than merely
summarized.

This paper makes the following contributions:

1. **A cheap, scalable gap-extraction funnel (Phase 1, ✅ implemented and benchmarked)** whose
   final precision filter is a *batched* LLM call, keeping cost roughly constant per document.
2. **A reproducibility audit of two published SOTAs.** We match one (RCT limitation detection)
   under a faithfully reconstructed protocol and show the other (future-work recognition) to be
   inflated by evaluation leakage. ✅
3. **A cross-dataset "data-not-model" result**: after fine-tuning, a 33 M general encoder
   matches or beats 110 M domain-specific models on three datasets. ✅
4. **A specification of the ideation phase (Phase 2)**: normalization, entity linking,
   canonicalization, gap graph with addressed-edge mining, scoring, three idea-generation
   mechanisms, grounded synthesis, and novelty/feasibility verification. Implementation and
   evaluation are ⟨PENDING⟩.

---

## 2. Related Work

- **Self-acknowledged limitations.** Lan et al. (2024) annotate limitation sentences and their
  types in RCT publications and report a fine-tuned PubMedBERT detector (P = 0.751, R = 0.907,
  F1 = 0.821). We use their corpus and labels as an evaluation target.
- **Future-work sentences.** Zhang et al. (2022) build the ACL FWS-RC corpus and report a
  Bernoulli Naive Bayes recognizer at macro-F1 = 0.9073. We reproduce and audit this result.
- **Mandated limitation sections and LimGen / BAGELS.** Whole-section harvesting (every
  sentence in a "Limitations" section labeled positive) is convenient but noisy; we quantify
  the resulting label noise.
- **Idea generation and RAG.** ⟨PENDING: position vs. FutureGen (2025) and analogical/idea-
  generation literature; cross-domain transfer; novelty estimation⟩.

⟨PENDING: full related-work prose and citations; see `references.bib`.⟩

---

## 3. Materials and Methods

### 3.1. System overview

Gap2Idea has two phases. **Phase 1 (extraction)** converts full-text papers into a stream of
gap sentences. **Phase 2 (ideation)** converts the gap stream into ranked research ideas.
Phase 1 is implemented and benchmarked (Section 4); Phase 2 is specified here and ⟨PENDING⟩
in implementation.

### 3.2. Phase 1 — Gap extraction funnel ✅

A three-stage funnel trades a large amount of cheap recall for a small amount of expensive
precision:

- **Stage A — structural slice.** Parse the document (GROBID sections) and keep only
  gap-bearing regions (limitations, future-work, discussion, conclusion), discarding the bulk
  of the paper before any model runs.
- **Stage B — cheap detector.** A rule layer of lexical cues plus an embedding head
  (`bge-small-en-v1.5`, 33 M parameters, frozen encoder + logistic regression, or fine-tuned)
  classifies each candidate sentence as *limitation*, *future_work*, or *none*. Tuned for
  **recall**.
- **Stage C — LLM precision filter.** Surviving candidates are judged in **batches** (one
  structured-output call per ~40 sentences, `json_schema`-constrained) by an LLM. Domain
  prompt modes (`validate_rct`, `validate_fws`) reject prior-work critiques, contribution
  claims, results restatements, acknowledgments, and citations. Provider abstraction supports
  YandexGPT (incl. `yandexgpt-5-pro`, `gpt-oss-120b`, `qwen3-235b`) and OpenRouter.

### 3.3. Phase 2 — Ideation pipeline ⟨PENDING⟩

The following steps are specified but not yet implemented/evaluated.

1. **Normalization into a schema.** Each gap → `{limitation_type, target_entity, cause, scope,
   evidence_span, paper_id, year}` via a small fine-tuned classifier (not a per-record LLM).
   Status: taxonomy available for RCT (13 coarse types); general extractor ⟨PENDING⟩.
2. **Entity linking.** Resolve `target_entity` to an ontology node (CSO / MeSH / Papers-with-
   Code). Expected coverage ~40–60 %; embedding-cluster fallback otherwise. ⟨PENDING⟩.
3. **Canonicalization.** Embed → approximate-nearest-neighbor index → density clustering
   (UMAP + HDBSCAN) → canonical limitations; cluster size = frequency signal. ⟨PENDING⟩.
4. **Gap graph + addressed edges.** `(paper)-[states]->(limitation)-[about]->(entity)` plus
   `(paper)-[addresses]->(limitation)` mined from **citation contexts**; a mined `addresses`
   edge is a strong *negative* signal (already solved). ⟨PENDING; requires citation-context
   corpus (S2ORC/OpenAlex)⟩.
5. **Gap scoring.** Composite of frequency, **persistence** (field-volume-normalized year
   trend), **unaddressed ratio**, breadth, and actionability. Note: raw frequency is
   down-weighted (the commonest gaps are the most generic). ⟨PENDING⟩.
6. **Idea generation (three mechanisms).** intra-domain matching (low novelty, high
   feasibility); **cross-domain analogical transfer** in a *structure* space, not a topic space
   (high novelty); limitation composition (addressing 2–3 co-occurring gaps). ⟨PENDING⟩.
7. **Grounded synthesis.** LLM produces `{hypothesis, method_sketch, experimental_design,
   expected_contribution, resolves}` constrained to retrieved evidence. ⟨PENDING⟩.
8. **Novelty verification.** Dense retrieval of nearest existing abstracts + an **entailment**
   check ("does an existing paper already do this?"), not a cosine threshold. ⟨PENDING⟩.
9. **Feasibility filter.** Data availability, compute, evaluability, expertise;
   rank by novelty × feasibility × gap_score. ⟨PENDING⟩.
10. **Retrospective validation.** Truncate the corpus at year *T*, generate, and check whether
    *T+1…T+3* papers implemented the ideas—**using an LLM whose pretraining cutoff ≤ T** to
    avoid contamination. ⟨PENDING⟩.

### 3.4. Datasets

| Dataset | Task | Size | Labels | Use |
|---|---|---|---|---|
| RCT/SAL (Lan 2024) | limitation detection | ~43 k sentences, 200 papers, 952 limitations | human, 13 coarse types | ✅ benchmark + Phase-2 sandbox |
| FWS (Zhang 2022) | future-work recognition | 64.9 k sentences, 9 k papers | human, binary | ✅ benchmark |
| LimGen / BAGELS | limitation sentences | whole-section harvest | noisy (~half non-limitation) | ✅ noise analysis |
| ACL limitations (harvest) | limitation sentences | 6 433 | whole-section, ~54 % clean | ✅ Phase-1→2 bridge corpus |

### 3.5. Implementation

Python; `sentence-transformers` (bge-small), `scikit-learn`, `transformers` (BiomedBERT/
SciBERT fine-tuning); YandexGPT via an OpenAI-compatible client with model-slug rewriting;
GPU experiments on a single Tesla V100. ⟨PENDING: release/version details⟩.

---

## 4. Results

### 4.1. RCT limitation detection vs. Lan et al. (2024) ✅

Their reported best: **P = 0.751, R = 0.907, F1 = 0.821** (PubMedBERT, evaluated on a
section-filtered pool at ~20.7 % positive prevalence). We reconstruct that pool via PubMed
Central BioC section types (retaining 99 % of gold positives) and evaluate at their prevalence
(hard section-negatives):

| Method (section pool @ 20.7 %) | Precision | Recall | F1 |
|---|---|---|---|
| Their PubMedBERT (reported) | 0.751 | 0.907 | 0.821 |
| Our detector only | 0.668 | 0.887 | 0.762 |
| Our detector + Stage C (gpt-oss-120b) | 0.752 | 0.882 | 0.812 |
| Our detector + Stage C (yandexgpt-5-pro) | 0.780 | 0.842 | 0.810 |

**We match their precision; the residual ~0.01 F1 is a recall gap, not a method gap.** On the
realistic full-document stream (2.6 % prevalence), Stage C lifts F1 by +0.05–0.11; at 20.7 %
it is roughly neutral—establishing it as a **low-prevalence precision tool**.

### 4.2. Future-work recognition vs. Zhang et al. (2022) ✅

| Pipeline | Macro-F1 |
|---|---|
| Their reported | 0.9073 |
| Reproduced (their code: chi² selection on full data, then 10-fold CV) | **0.9072** |
| Proper (chi² refit inside each fold) | **0.8252** |
| Leakage inflation | **+0.082** |

The reported SOTA reproduces exactly and is a **feature-selection-leakage artifact**. Under an
honest, clean split:

| Model (balanced clean split) | Params | Macro-F1 |
|---|---|---|
| bge-small (fine-tuned) | 33 M | **0.879** |
| PubMedBERT (fine-tuned) | 110 M | 0.877 |
| SciBERT (fine-tuned) | 110 M | 0.864 |
| BernoulliNB / TF-IDF+logreg | — | 0.830 / 0.827 |

### 4.3. Cross-dataset findings ✅

- **Data not model.** A 33 M general encoder matches/beats 110 M domain models on RCT and FWS;
  domain-pretraining (PubMedBERT) provides no advantage after fine-tuning.
- **Stage C is prevalence-sensitive.** Clear F1 lift at 2–3 % prevalence; neutral at 14–21 %.
- **Published SOTAs regress to honest baselines** once leakage, prevalence, and label noise are
  controlled (~0.85 macro-F1 on FWS; matched precision on RCT).

### 4.4. Extraction-to-ideation bridge: input quality ✅

Stage C judges only **54 %** of the raw whole-section ACL limitation harvest to be genuine
authors'-own limitations; the remainder are contributions, results, methods, citations, or
field-level observations. Stage C therefore doubles as the cleaning step that produces the
Phase-2 corpus (~3 500 clean limitations from 6 433 raw).

### 4.5. Phase 2 — ideation results ⟨PENDING⟩

⟨PENDING: canonical-limitation counts and cluster quality; gap-ranking table; generated-idea
samples; novelty-filter precision; feasibility distribution; retrospective-validation recall
against T+1…T+3 papers; expert Likert ratings.⟩

---

## 5. Discussion

Our extraction results argue for a **cost-first design**: a small encoder plus a batched LLM
filter matches specialized transformers at a fraction of the cost, and the LLM's value is
concentrated where prevalence is low—exactly the realistic full-document setting. The
reproducibility findings (a leakage-inflated SOTA; a precision-matched but recall-limited SOTA)
motivate reporting under honest, prevalence-aware, leakage-free protocols.

**Threats to validity.** (i) Our detectors are trained on the target corpora's train splits
(disjoint by paper id), whereas the shipped pipeline is zero-shot arXiv-trained; the two must
not be conflated. (ii) Stage-C prompt selection for RCT was made on test predictions—a dev-
split reselection is ⟨PENDING⟩. (iii) The RCT prevalence-matched pool uses random negatives in
one table (matching the original protocol) and section negatives in another; only the latter is
apples-to-apples. (iv) Residual "false positives" in both datasets are dominated by
**annotation-label noise**, capping measurable precision.

**Limitations of the study.** Phase 2 is unimplemented; RCT limitations are trial-*execution*
issues rather than open research problems, so idea generation must target multi-domain,
research-problem corpora (arXiv). ⟨PENDING: quantify Phase-2 novelty/feasibility.⟩

---

## 6. Conclusions

We presented Gap2Idea, a two-phase system for mining self-acknowledged research gaps and
operationalizing them into ideas. Phase 1 is implemented and benchmarked: it matches or audits
two published SOTAs and yields a reusable "data-not-model" and "low-prevalence precision-filter"
characterization of the design space. Phase 2 is specified and ⟨PENDING⟩; the immediate next
steps are to (a) Stage-C-clean an ACL/arXiv limitation corpus, (b) canonicalize via clustering,
and (c) implement grounded, novelty-verified idea generation with retrospective validation.

---

**Author Contributions:** ⟨PENDING⟩.
**Funding:** ⟨PENDING⟩.
**Data Availability Statement:** RCT/SAL and FWS corpora are public; code and derived corpora
⟨PENDING: release URL⟩.
**Conflicts of Interest:** The authors declare no conflict of interest.

## References

⟨PENDING: format per MDPI. Key entries already in `docs/thesis_latex/references.bib`:
Lan et al. 2024 (RCT/SAL); Zhang et al. 2022 (FWS); LimGen; BAGELS; FutureGen 2025;
bge-small (Xiao et al. 2023); GROBID; SPECTER; Leiden.⟩
