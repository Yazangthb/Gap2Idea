# Publication plan — gap extractor (IEEE Access / MDPI)

Target: a complete, rigorous, reproducible system paper. Both venues judge on **soundness,
completeness, reproducibility** (not novelty). Critical path = **Phase 1** — nothing else is
trustworthy until the metric is fixed. Est. ~3–4 focused weeks to submission-ready.

Each item: **what it means** → **what to do** → **why it matters**.

## Phase 1 — Fix the evaluation (gating; ~1 wk)
1. **Span/paragraph-level scoring**
   - *Means:* stop scoring a prediction wrong just because it's an adjacent sentence of a limitation
     SAL tagged on a different sentence.
   - *Do:* a kept sentence counts as correct if it's within the same limitation span/paragraph (±1
     sentence) as any gold positive; re-score SAL + Kilicoglu (strict W=0 vs span W=1/2).
   - *Why:* removes the documented gold-noise artifact → honest, defensible precision.
2. **Human adjudication**
   - *Means:* a person decides who's right on the LLM-vs-gold disagreements.
   - *Do:* hand-label ~150–200 disagreement sentences; report Stage C's **true precision** (expect ~0.85–0.9).
   - *Why:* converts "precision looks low" into a credited, trustworthy number.

## Phase 2 — Reproduce baselines (~1 wk)
3. **Faithful Lan SAL reproduction** — run PubMedBERT on *their* exact split + binary task; side-by-side
   with the funnel. *Why:* a real head-to-head vs the 0.821 SOTA.
4. **Run Kilicoglu's rule-based tool** — reproduce the 0.80 baseline instead of citing it.
5. **LLM-on-full-paper baseline** — prompt an LLM on whole papers; measure recall/precision + cost.
   *Why:* the baseline the funnel must beat *on cost* — the core selling point.

## Phase 3 — The headline result (~2–3 days)
6. **Cost/efficiency analysis** — tokens + $ per paper: funnel (~6 sentences to the LLM) vs full-paper
   LLM; plot cost vs recall. *Why:* "≈same recall at a fraction of the cost" is the practical contribution.

## Phase 4 — Breadth & consolidation (~1 wk)
7. **Multi-domain evidence** — report arXiv/ACL (FWS, Sci-Sentence, PDF demo) alongside biomedical SAL.
8. **Consolidate ablations** — Stage A on/off, Stage B (frozen vs fine-tuned, in- vs cross-domain),
   Stage C on/off + config, cue optimization, cross-dataset → clean tables/figures.
9. **Error analysis + annotation critique** — qualitative examples + write up the finding that span-based
   limitation benchmarks systematically under-measure. *Why:* the intellectual contribution.

## Phase 5 — Write-up & submit (~1 wk)
10. **Paper draft** — Intro (+ contributions) → Related Work (LimGen/LimTopic/Lan/Kilicoglu) → Method →
    Experiments/Results → Discussion (critique + limitations) → Conclusion.
11. **Reproducibility package** — the `minimal` repo + one-command reproduce script + Data Availability.
12. **Formatting & admin** — journal template (IEEE double-column / MDPI), abstract + keywords, author
    bios/ORCID (IEEE), cover letter + suggested reviewers, APC budget.

## Contributions (state honestly)
1. A cost-efficient, recall-first extraction funnel matching full-paper-LLM recall at a fraction of the cost.
2. A cross-domain evaluation (fine-tuned-in-domain vs frozen-cross-domain detector behavior).
3. A finding that span-based limitation benchmarks systematically under-measure, with a corrected protocol.
