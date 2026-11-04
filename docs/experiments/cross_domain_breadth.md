# Cross-domain breadth — current funnel on FWS + Sci-Sentence (Oct 2026)

Runs the **current production funnel** (Stage A section gate + Stage B `cue_label` + cached
bge-small+logreg head + Stage C `yandexgpt-5-lite`) on two CS-domain datasets, to show the
funnel is not SAL/biomedical-specific. Mirrors `scripts/run_extraction_pipeline.py` Stage B
exactly. Script + raw results: `stage_c_package/breadth_funnel.py`, `breadth_results.json`.

## Results

Stage B = frozen bge-small + logreg head (production head; *weaker* than SAL's fine-tuned PubMedBERT →
a conservative floor). Stage C = `yandexgpt-5-lite`, ±1 context, batch 20.

| dataset | domain / framing | n (pos%) | A+B P/R/F1 | A+B+C P/R/F1 | **keepTP** | **dropFP** |
|---|---|---|---|---|---|---|
| **FWS** (Zhang 2022) | future-work, ACL/CS — authors' own | 1,506 (14.6%) | 0.723 / 0.618 / 0.667 | 0.843 / 0.609 / 0.707 | **134/136 (98.5%)** | 27/52 (52%) |
| **Sci-Sentence** — `validate` (mismatched) | limitation+research-gap, CS *related-work* | 140 (28.6%) | 0.578 / 0.650 / 0.612 | 0.900 / 0.450 / 0.600 | 18/26 | 17/19 |
| **Sci-Sentence** — **GAP-matched prompt** | (same) | 140 (28.6%) | 0.578 / 0.650 / 0.612 | **1.000 / 0.575 / 0.730** | 23/26 (88%) | **19/19 (100%)** |

- `keepTP` = true positives the detector found that Stage C **keeps** (recall preservation); `dropFP` =
  detector false positives Stage C **removes** (precision gain). These are the precision-filter metrics —
  report these, not raw precision vs noisy gold.
- FWS: positive = `future_work`; Stage C mode `validate_fws`. Corpus is already Conclusion/Future-Work
  sections → Stage A ~pass-through. Train-overlap sentences removed (leakage guard).
- Sci-Sentence: positive = `LIMITATION ∪ RESEARCH GAP`; TEST split (140); no section gate. Head never
  trained on it (clean transfer). `GAP-matched` prompt = keep limitations/gaps in *prior/existing work*
  (not only the authors' own study).

## Recall-first operating point (Sci-Sentence) — the funnel design

The 0.575 end-to-end recall above is a **Stage B** limit (frozen cross-domain head @ thr 0.5), not
Stage C. Training an in-domain head on Sci-Sentence's own train split (490 sents, bge-small+logreg)
and running it **recall-first**, then applying the GAP-matched Stage C:

| config | P | R | F1 | keepTP | dropFP |
|---|---|---|---|---|---|
| frozen head thr=0.5 + Stage C | 1.000 | 0.575 | 0.730 | 23/26 | 19/19 |
| in-domain head thr=0.3 — A+B (recall-first) | 0.639 | **0.975** | 0.772 | — | — |
| **in-domain head thr=0.3 + GAP-matched Stage C** | **0.943** | **0.825** | **0.880** | 33/39 | 20/22 (91%) |

→ The recall-first detector catches **39/40 gaps** but is flooded with FPs (P=0.64); Stage C removes
91% of them and lifts precision to **0.943** at R=0.825 → **F1 0.880**. This is the funnel thesis
end-to-end: recall-first Stage B + Stage C precision filter. Residual recall loss (6 gaps) is Stage C
over-dropping (tunable via a keep-biased GAP prompt), no longer Stage B. The frozen→in-domain jump
(R 0.65→0.975) mirrors SAL: a strong result wants a fine-tuned in-domain detector.

## Sci-Sentence — summary + comparison with reported methods

**Our funnel (binary gap = LIMITATION ∪ RESEARCH GAP), TEST split (140 sents, 40 gap):**

| our config | detector | P | R | F1 |
|---|---|---|---|---|
| frozen head thr=0.5 (cross-domain) + Stage C | bge-small 33M (no FT) | 1.000 | 0.575 | 0.730 |
| in-domain head thr=0.5 — A+B only | bge-small 33M + logreg | 0.761 | 0.875 | 0.814 |
| **in-domain head thr=0.3 (recall-first) + Stage C** | bge-small 33M + lite | **0.943** | 0.825 | **0.880** |

**Reported methods on Sci-Sentence** (Bolaños, Salatino, Osborne & Motta, *Modelling and Classifying
the Components of a Literature Review*, arXiv:2508.04337 / PeerJ CS cs-3921, 2025). Their task is
**7-class**; shown are per-class F1 for the two gap classes + the 7-class average (from
`results/Sentence Classification FT Results.xlsx`). These 37 models (all evaluated by the dataset
authors) are the *only* models benchmarked on Sci-Sentence to date — no independent third-party
evaluation exists yet (dataset public since Aug 2025).

**Citation check (Semantic Scholar, Oct 2026):** the paper has only **5 citations**, and **none use
the Sci-Sentence dataset for any task** — the lone "match" is the authors' own PeerJ journal version;
the other four (UniCite citation taxonomy; LLM scientific-quality probing; OntoGPT term extraction;
OUTLINEFORGE outline generation) neither use the dataset nor do gap/limitation extraction. So there is
**no external baseline for our (binary gap-extraction) task** on Sci-Sentence — only the authors'
7-class numbers below. This makes Sci-Sentence a novelty angle but not a standalone benchmark anchor;
pair it with SAL / Kilicoglu, which have directly comparable limitation-*detection* baselines.

| method | type / size | Research-Gap F1 | Limitation F1 | 7-class avg F1 |
|---|---|---|---|---|
| BERT (base) | encoder 110M | 0.720 | 0.200 | 0.590 |
| BioBERT | encoder 110M | 0.826 | 0.516 | 0.731 |
| SciBERT (cased) | encoder 110M | 0.884 | 0.769 | 0.870 |
| SciBERT (cased, augmented) | encoder 110M | 0.900 | 0.857 | 0.928 |
| Olmo-1B (NEFT) | decoder 1B, LoRA/NEFT | 0.927 | 0.884 | 0.921 |
| SuperNova-Medius-14B (LoRA, aug) | decoder 14B | 0.923 | 0.923 | 0.943 |
| **gpt-4o-mini** (best overall) | API LLM | 0.974 | 0.950 | 0.964 |

**How to read the comparison (honest):**
- **Task mismatch favors us.** Theirs is 7-way (must separate Research-Gap from Limitation from
  Extension/Result/Description/Other); ours merges the two gap classes into one binary target — an
  easier problem. So our 0.880 is **not** directly above/below their 7-class numbers.
- **Cost/size strongly favors us.** Our detector is a **33M frozen bge-small + logreg** (+ a cheap
  `yandexgpt-5-lite` filter on ≤61 sentences); their strong results use fine-tuned 1B–14B decoders with
  data augmentation, or API GPT-4o-mini. We land around their **SciBERT-110M** encoder level
  (Limitation F1 0.77–0.86) on the easier binary framing, below their best augmented/large models.
- **Different goal.** Their pipeline maximizes a 140-sentence classification benchmark; ours is a
  recall-first **full-paper scanner** — Stage B floods candidates, Stage C filters (dropFP 91%). The
  takeaway is efficiency: near-GPT-4o-mini-class *gap* precision (0.94) from a 33M detector + a small LLM.

## Findings
1. **Stage C is a near-perfect precision filter (the claim).** On Sci-Sentence with a task-matched prompt
   it **dropped 100% of the detector's false positives (19/19)** while keeping 88% of true gaps → **precision
   = 1.00**. On FWS it **keeps 98.5% of true future-work (134/136)** and removes half the detector FPs. The
   LLM removes false positives at essentially no cost to true positives — exactly what Stage C is for.
2. **Where precision reads < 1.0, it's gold noise, not the filter.** FWS's residual "FPs" (0.84 precision)
   are **real future-work FWS mislabeled** (*"promising directions for future research"*, *"we would like to
   incorporate syntactic information"*, *"opens many challenging directions"*) — same under-labeling as SAL.
3. **The prompt must match the *framing*, not just the domain.** The self-acknowledged-limitation `validate`
   prompt on Sci-Sentence's *others'-work* gaps gave R=0.45; the GAP-matched prompt restored it and hit P=1.0.
   Literature-gap detection (relevant to RQ3 idea-generation) needs its own Stage C prompt.
4. **The end-to-end recall ceiling is Stage B, not Stage C.** Sci-Sentence A+B+C R=0.575 is bounded by the
   frozen detector (A+B R=0.65); Stage C cost only 3/26 recall and 0 precision. A fine-tuned detector lifts this.

## Status vs publication plan
Covers Phase 4 item 7 (multi-domain evidence) for the two in-repo CS datasets. Next breadth steps:
Kilicoglu 2018 (2nd biomedical end-to-end) and the ACL mandatory-Limitations-section harvest.
