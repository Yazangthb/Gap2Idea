# Funnel v2 — recall-first Stage A (junk-filter + bge-small head) on the gold

Exact per-sentence eval on the gold-aligned TEST sentences. Stage-A head threshold tuned on VAL for target recall 95%. Gap = any type (lim/fw/research_gap).

| config | recall | precision | F1 | retention |
|---|---|---|---|---|
| junk-filter only (ceiling) | 97.0% | 6.7% | 12.5% | 81.8% |
| **A+B recall-first (tau=0.111)** | **96.5%** | 8.8% | 16.1% | 62.1% |
| A+B balanced (max-F1 tau=0.85) | 72.3% | 33.4% | 45.7% | 12.2% |

**Recall-first per-domain:** arxiv_cs: R=96.2% P=8.0% · biomed_sal: R=97.8% P=14.4%
**Recall-first per-type recall:** future_work: 96.9% (n=97) · limitation: 95.5% (n=220) · research_gap: 97.1% (n=347)
**Type accuracy (detected gaps):** 73.0%

Gold gaps lost to the junk-filter: 20 / 664 (= the 3.0pt gap below 100% recall).

Candidates saved for Stage C: 7297 -> `funnel_v2_candidates.json`.

### Notes
- Exact per-sentence metrics (gold-aligned); no token matching.
- recall-first = Stage A tuned high-recall; low precision is BY DESIGN — Stage C restores it.
- junk-filter = free regex + biomed boilerplate sections; no GROBID/sections needed.
