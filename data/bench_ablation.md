# Extraction-bench ablation (N=10 papers, unarXive gold)


## Stage 1 — regex section parser (higher is better)

| variant | rouge1_f | rouge2_f | rougeL_f |
|---|---:|---:|---:|
| v1 (text + old regex) | 0.246 | 0.125 | 0.158 |
| v2a (text + new vocab) | 0.246 | 0.125 | 0.158 |
| v2b (PDF + style + new vocab) | 0.459 | 0.359 | 0.411 |

## Stage 2 — LLM gap extraction (recovery higher better, hallucination lower better)

| variant | n_gaps | mean_sim_to_gold | recovery@0.6 | hallucination@0.6 |
|---|---:|---:|---:|---:|
| v1 (text + old regex) | 1.10 | 0.515 | 0.400 | 0.100 |
| v2a (text + new vocab) | nan | nan | nan | nan |
| v2b (PDF + style + new vocab) | 1.70 | 0.739 | 0.650 | 0.150 |

## Per-paper rouge1_f, v1 → v2b

| paper_id | gold title (truncated) | v1 | v2b | Δ |
|---|---|---:|---:|---:|
| `0803.2303` | Open Questions | 0.128 | 0.230 | +0.102 |
| `0805.3456` | Conclusion and future work | 0.137 | 0.137 | -0.001 |
| `0806.2938` | Open questions. Conclusions | 0.104 | 0.893 | +0.789 |
| `0807.0023` | Future Work | 0.593 | 0.314 | -0.280 |
| `0807.1134` | Limitations of the Calculations | 0.368 | 0.325 | -0.043 |
| `0808.2057` | Open problems | 0.316 | 0.347 | +0.031 |
| `0811.2508` | Improvements and Future Work | 0.243 | 0.262 | +0.019 |
| `0811.3859` | Conclusion and Open Problems | 0.249 | 0.757 | +0.508 |
| `cs/0610128` | Conclusion and Future Work | 0.264 | 0.858 | +0.594 |
| `quant-ph/0402095` | Oracle Limitations, Open Problems | 0.057 | 0.472 | +0.415 |