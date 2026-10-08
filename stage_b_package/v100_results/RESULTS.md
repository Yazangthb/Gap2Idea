# Stage B experiment — Sci-Sentence + FWS (run on V100)

**Date:** 2026-10-06 · **Host:** `sam` (2× Tesla V100-SXM2-32GB) · env: torch 2.5.1+cu121, sentence-transformers 6.1.0, transformers 5.18.

## Setup
- **Training data:** Sci-Sentence `Based_Version.xlsx` + FWS `fws_corpus.csv`.
  - Label map: Sci-Sentence `LIMITATION → limitation`, `EXTENSION → future_work`, `RESEARCH GAP → future_work`, others → `none`; FWS `1 → future_work`, `0 → none`.
  - FWS capped at 2,000 future_work + 4,000 none (paper-grouped sample).
  - **6,700 training rows**: none 4,400 · future_work 2,199 · **limitation 101** (the thin class — Sci-Sentence is the only limitation source; FWS has none).
  - Leakage guard: drop any training sentence with ≥0.8 token-containment to a gold gap (0 dropped).
- **Split:** paper-disjoint (GroupShuffleSplit, 15% held out) for in-dataset F1.
- **End-to-end eval:** our 10 gold papers (49 gaps). Stage-A candidates precomputed locally; recall = gold gap recovered (token-containment ≥ 0.7) among kept sentences.

## Results

Fine-tunes are **full fine-tuning** (all weights, AdamW lr 2e-5, 3 epochs, class-weighted loss).

| | Frozen (bge-small + logreg) | DistilBERT FT (full) | SciBERT FT (full) |
|---|---|---|---|
| **in-dataset macro F1** | 0.633 | 0.745 | **0.846** |
| in-dataset limitation F1 | 0.317 | 0.533 | **0.800** |
| in-dataset future_work F1 | 0.743 | 0.805 | **0.833** |
| in-dataset none F1 | 0.838 | 0.897 | **0.905** |
| **gold end-to-end recall (rules+head)** | **0.429** | 0.306 | 0.327 |
| gold limitation recall | **0.455** | 0.182 | 0.273 |
| gold future_work recall | **0.786** | 0.714 | 0.714 |
| gold head-only recall | **0.347** | 0.204 | 0.204 |
| time | embed 6.7k in 3.1 s | FT 27.6 s | FT 47.0 s |

(Full numbers: `frozen_result.json`, `bert_result.json` = DistilBERT, `scibert_result.json`.)
SciBERT shipped legacy `.bin`; converted to safetensors on-box to run under torch 2.5.1.

## Findings
1. **Fine-tuning wins in-dataset but loses on our gold — monotonically.** As model capacity/fit rises
   (frozen → DistilBERT → SciBERT), in-dataset macro F1 climbs 0.633 → 0.745 → **0.846**, yet gold
   end-to-end recall stays **below** the frozen head (0.429 → 0.306 → 0.327). Both full fine-tunes
   overfit the Sci-Sentence+FWS distribution and transfer worse to arXiv (gold limitation head-only
   recall = 0.0 for both). **The frozen bge head is the best for the real task.** (Consistent with the
   old "data not model" result, E20.)
2. **FWS fixed future_work** — gold future_work recall 0.786 (frozen), the strongest class.
3. **Limitation is the bottleneck** — only 101 limitation training examples. Gold limitation recall 0.455
   (frozen). This is the next thing to fix.
4. **SciBERT ≥ DistilBERT** everywhere (scientific vocab helps), but neither beats frozen on gold. If a
   fine-tuned model is wanted, try a *regularized* variant (LoRA / freeze encoder + train head / 1 epoch)
   to curb the overfit — full fine-tuning on 6.7k sentences is too much capacity for this data.

## Next steps
- Add a **limitation** source to lift the starved class: SAL/RCT (true limitation labels + negatives, biomedical — cap it) and/or cleaned ACL. Re-run frozen head.
- SciBERT fine-tune not yet run: its checkpoint is legacy `.bin` and transformers 5.18 blocks `torch.load` on torch < 2.6 → needs a torch upgrade (≥2.6). Scientific vocab *might* transfer better than DistilBERT, but expect the same overfit-vs-gold pattern.
- Numbers are on a 10-paper gold set (small sample): good for ranking methods, not a large-scale claim.
