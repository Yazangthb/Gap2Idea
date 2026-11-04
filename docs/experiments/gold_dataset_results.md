# Gap gold dataset — build results (Oct 2026)

A new per-sentence **research-gap** benchmark built with a 2-judge Claude ensemble, validated by
inter-model agreement and (pending) a human pass. See `gold_dataset_plan.md` for the methodology.

## Corpus
- **230 papers, 68,719 sentences**: 190 arXiv CS/ML (`paper_texts.jsonl`) + 40 SAL biomedical.
- Labels per sentence: `limitation` / `future_work` / `research_gap` / none (binary **gap** = any type).

## Method
- **Judge 1** `claude-opus-5-5`, **Judge 2** `claude-sonnet-5-5`, independent, whole paper as numbered
  sentences → JSON `{type:[{i,q}]}`, mapped by index and verified by quote.
- **Agreement = confidence.** Both judges, same type → high-confidence gold.
- **Cost: $31.90** total (230 papers × 2 judges), 0 judge errors. (~$68 of the $100 budget remains.)

## Inter-model agreement (Cohen's κ, pooled over 68,719 sentences)
| decision | κ |
|---|---|
| **any gap** | **0.778** |
| limitation | 0.748 |
| future_work | 0.869 |
| research_gap | 0.725 |

→ κ=0.778 for "is this a gap" **matches the human inter-annotator ceiling** on this task (SAL reported
Gwet/α ≈ 0.78; Kilicoglu α ≈ 0.78). Two independent strong models agree about as much as humans do with
each other — the core validity signal for the LLM-labeled gold. `future_work` is nearly unambiguous (0.869).

## Built-in SAL validity check (40 SAL papers, 205 known gold limitations)
- Ensemble-both `limitation` vs SAL gold: **P=0.544, R=0.844**.
- SAL gold caught as **any** agreed gap type: **recall 0.868**.
- The low precision is the documented SAL under-labeling (the ensemble flags real limitations SAL didn't
  tag — confirmed on the sample paper); the high recall (0.84–0.87) shows the ensemble reproduces SAL's gold.

## Dataset statistics (intersection gold)
- **2,960 high-confidence gaps** (both judges agree on the same type) — the core gold.
- **4,697 sentences flagged by ≥1 judge** (includes disagreements, kept for adjudication/human review).
- Gap prevalence: Opus 6.2%, Sonnet 5.1% of all sentences.

## FINAL gold — recall-boosted (union + adjudication)
To raise recall, the candidate set is the **union** of both judges (SAL recall 0.956 vs 0.868 for the
intersection); the 1,737 one-sided/type-mismatch candidates were **adjudicated** by `claude-opus-5-5`
(high effort, ±3 context). Confirmed ones join the gold.
- **4,462 gap sentences**: research_gap 2,234 · limitation 1,543 · future_work 685.
- Composition: **2,960 high-confidence** (both agree) + **1,502 adjudicator-confirmed** one-sided (235 rejected).
- Adjudication cost **~$4.1** (total build so far ≈ **$36**).

| gold version | SAL limitation recall | SAL binary-gap recall |
|---|---|---|
| intersection | 0.844 | 0.868 |
| **union + adjudication (final)** | **0.927** | **0.956** |

Precision vs SAL = 0.456 (SAL under-labeling artifact — the gold now includes real limitations SAL never
tagged); **true precision comes from the human pass**. The final gold is `data/gold/gap_gold_final.jsonl`
(per-sentence: label, confidence = high|adjudicated, sal_gold). Human worksheet refreshed (210 rows:
high-conf, adjudicated-kept, adjudicated-dropped, negatives — validates both the gold and the adjudicator).

## Files
- `data/gold/gap_gold.jsonl` — per-paper: each judge's votes (indices per type) + agreement + cost.
- `data/gold/gap_gold_sentences.jsonl` — flattened per-sentence: text, section, opus/sonnet labels,
  `agree`, `confidence` (high/med/low), `gold_gap` (both agree), `sal_gold`.
- `data/gold/human_validation_worksheet.csv` — **210 stratified rows** (70 both-agree positives,
  60 one-sided disagreements, 20 type-mismatches, 60 negatives) with ±1 context + blank human columns.
- Scripts: `scripts/dataset/gold_label_full.py` (labeler), `gold_postprocess.py` (report + flatten + worksheet).

## Human validation (the remaining step to make it a published benchmark)
1. ≥2 annotators label the 210-row worksheet blind against the guideline.
2. Report **inter-annotator agreement** (Cohen's κ / Krippendorff α) and **human vs ensemble-gold** κ.
3. If human–ensemble κ ≥ ~0.75, the ensemble gold is validated for the full corpus; disagreement patterns
   become the error-analysis section. Adjudicate residual disagreements (optional 3rd Opus-max pass).

## Status
Dataset built + inter-model-validated at human-level agreement (κ=0.778). Ready for the human pass.
Optional next: a 3rd `claude-opus-5-5` (max effort) adjudication pass over the ~1,600 one-sided/mismatch
disagreements to lift them to resolved labels (~$5–8 of remaining budget).
