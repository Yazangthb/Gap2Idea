# SciBERT-FT on the gold papers — notes

> Quick experiment (after the LimGen win: SciBERT-FT F1 = 0.743 there). Drop the
> fine-tuned 3-class SciBERT into the funnel as Stage B in place of bge+logreg,
> rerun on our 10 gold papers, compare apples-to-apples.

## Setup
- **Same training data** as the bge+logreg head: runs/\* self-distilled + ACL
  mandated-Limitations harvest (1500 cap), 3 classes (none / limitation /
  future_work).
- **3-class fine-tune** on V100 (2 epochs, lr 3e-5, bs 24, ~25K → 8395 balanced
  rows; epoch losses 0.61 → 0.35).
- **Identical Stage A** for both heads (the slice is independent of the
  classifier). The only change is Stage B's classifier.
- Wrapped as `BertGapHead` that duck-types `EmbeddingGapHead.predict` so it drops
  into `extract_gaps()` with no other changes.
- Script: `scripts/training/test_scibert_gold.py`. Output:
  `data/scibert_prep/scibert_gold_{summary.md, gaps.tsv}`.

## Result (10 gold papers, 19 gold gaps)
| head | preds | recall | precision_floor | F1_floor |
|---|---|---|---|---|
| bge + logreg (current) | 62 | 0.526 | 0.194 | 0.283 |
| **SciBERT-FT** | **49** | 0.526 | **0.245** | **0.335** |

- **+5 pts precision floor, −13 predictions, same recall.**
- True precision is *higher* than the floor — several "extras" are clear
  limitations gpt-4o silver gold didn't extract.

## Observations
1. **Stage A is the recall ceiling on this gold** — neither head clears 0.526,
   the slice already caps it. Improving end-to-end recall on our gold needs Stage
   A changes (mid-paper limitation coverage), not Stage B.
2. **SciBERT-FT generalises from ACL training to arXiv AI/ML/math eval**: the
   improvement transfers despite the train↔eval domain mismatch.
3. **Future-work coverage stays weak** — only 116 future_work training examples
   (vs 1500 limitation + ACL harvest). The LimGen-style harvest doesn't help
   future_work. Fix: harvest future-work sentences from a future-work-mandated
   corpus, or add cue-protected future-work samples.
4. **The LimGen number (0.743) and the gold-paper number (0.335 floor) measure
   different things** — LimGen is binary classification of clean sentences; gold
   is end-to-end retrieval from scrambled full PDFs against a partial silver
   reference. Both are valid; report each in its own context.

## Implication for the paper
- **Ship SciBERT-FT as the default Stage B** for the paper version of the
  pipeline. The LimGen benchmark gives the "vs prior art" headline (0.743);
  the gold-paper qualitative figure shows cleaner output (49 vs 62 preds).
- **Honest framing:** "SciBERT-FT raises precision on the silver-gold paper
  test by ~5 points at no recall cost; the recall ceiling is Stage A
  (localization), not Stage B (classification)."
- **The LLM filter (Stage C) is now genuinely optional** — it hurt SciBERT-FT
  on LimGen (over-rejection). Mention as a tier for users with a weaker
  classifier; not part of the default pipeline.

## What I would try next (later)
- **Future-work data harvest** to fix the class imbalance (e.g. cue-confirmed
  sentences from a different corpus, peer-review meta-data).
- **Stage A recall on mid-paper limitations** — the real next bottleneck.
- **Soft LLM ensemble** instead of hard filter (LLM probability + SciBERT
  probability) — might push the LimGen number above 0.75.
- **Save the SciBERT head as `data/gap_head_scibert/`** and add a `--head-type`
  flag to the CLI so it's selectable alongside the bge+logreg default.
