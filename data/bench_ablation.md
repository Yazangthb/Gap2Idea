# Extraction-bench ablation (N=10 papers, unarXive gold)


## Stage 1 — regex section parser (higher is better)

| variant | rouge1_f | rouge2_f | rougeL_f |
|---|---:|---:|---:|
| v1 (text + old regex)  [N=10] | 0.246 | 0.125 | 0.158 |
| v2a (text + new vocab) [N=10] | 0.246 | 0.125 | 0.158 |
| v2b (PDF + style)      [N=10] | 0.459 | 0.359 | 0.411 |
| v2a (text + new vocab) [N=100] | 0.322 | 0.213 | 0.245 |
| v2b (PDF + style)      [N=100] | 0.506 | 0.419 | 0.454 |

## Stage 2 — LLM gap extraction (recovery higher better, hallucination lower better)

| variant | n_gaps | mean_sim_to_gold | recovery@0.6 | hallucination@0.6 |
|---|---:|---:|---:|---:|
| v1 (text + old regex)  [N=10] | 1.10 | 0.515 | 0.400 | 0.100 |
| v2a (text + new vocab) [N=10] | nan | nan | nan | nan |
| v2b (PDF + style)      [N=10] | 1.70 | 0.757 | 0.700 | 0.150 |
| v2a (text + new vocab) [N=100] | 1.34 | 0.569 | 0.435 | 0.045 |
| v2b (PDF + style)      [N=100] | 1.63 | 0.693 | 0.630 | 0.113 |

## Pipeline gaps vs Oracle gaps (gold section fed straight to LLM)

Two systems, same Stage-2 LLM. Oracle skips Stage 1.

| metric | mean | meaning |
|---|---:|---|
| n_oracle_gaps | 1.760 | gaps the LLM produces on the gold section |
| mean_sim_pipe_to_oracle | 0.577 | avg cosine: each pipeline gap → closest oracle gap |
| mean_sim_oracle_to_pipe | 0.575 | avg cosine: each oracle gap → closest pipeline gap |
| recovery_at_0.6 | 0.469 | fraction of pipeline gaps that match an oracle gap |
| coverage_at_0.6 | 0.475 | fraction of oracle gaps the pipeline reproduced |

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