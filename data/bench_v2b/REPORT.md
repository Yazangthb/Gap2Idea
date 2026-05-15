# Extraction-quality benchmark (N=10 papers, unarXive gold sections)

Reference: author-titled `future work` / `limitations` sections from unarXive 2023.


## Sampled papers

- `quant-ph/0402095`  titles: ['Oracle Limitations', 'Open Problems']  (gold 27021 chars, full 100328 chars)
- `cs/0610128`  titles: ['Conclusion and Future Work']  (gold 1274 chars, full 100720 chars)
- `0803.2303`  titles: ['Open Questions']  (gold 798 chars, full 11842 chars)
- `0805.3456`  titles: ['Conclusion and future work']  (gold 947 chars, full 43390 chars)
- `0806.2938`  titles: ['Open questions. Conclusions']  (gold 3721 chars, full 47806 chars)
- `0807.1134`  titles: ['Limitations of the Calculations']  (gold 2403 chars, full 216709 chars)
- `0807.0023`  titles: ['Future Work']  (gold 3002 chars, full 51840 chars)
- `0808.2057`  titles: ['Open problems']  (gold 3240 chars, full 56727 chars)
- `0811.2508`  titles: ['Improvements and Future Work']  (gold 1958 chars, full 25324 chars)
- `0811.3859`  titles: ['Conclusion and Open Problems']  (gold 2553 chars, full 114633 chars)

## Aggregate metrics (mean ± std)

| stage | metric | mean | std | n |
|---|---|---:|---:|---:|
| llm_gap | hallucination_at_0.5 | 0.050 | 0.158 | 10 |
| llm_gap | hallucination_at_0.6 | 0.150 | 0.337 | 10 |
| llm_gap | hallucination_at_0.7 | 0.200 | 0.350 | 10 |
| llm_gap | mean_sim_to_full | 0.897 | 0.163 | 10 |
| llm_gap | mean_sim_to_gold | 0.757 | 0.228 | 10 |
| llm_gap | n_gaps | 1.700 | 0.483 | 10 |
| llm_gap | recovery_at_0.5 | 0.800 | 0.350 | 10 |
| llm_gap | recovery_at_0.6 | 0.700 | 0.422 | 10 |
| llm_gap | recovery_at_0.7 | 0.650 | 0.412 | 10 |
| pipeline_vs_oracle | coverage_at_0.5 | 0.600 | 0.459 | 10 |
| pipeline_vs_oracle | coverage_at_0.6 | 0.600 | 0.459 | 10 |
| pipeline_vs_oracle | coverage_at_0.7 | 0.500 | 0.471 | 10 |
| pipeline_vs_oracle | mean_sim_oracle_to_pipe | 0.700 | 0.296 | 10 |
| pipeline_vs_oracle | mean_sim_pipe_to_oracle | 0.652 | 0.278 | 10 |
| pipeline_vs_oracle | n_oracle_gaps | 1.700 | 0.483 | 10 |
| pipeline_vs_oracle | recovery_at_0.5 | 0.600 | 0.459 | 10 |
| pipeline_vs_oracle | recovery_at_0.6 | 0.600 | 0.459 | 10 |
| pipeline_vs_oracle | recovery_at_0.7 | 0.500 | 0.471 | 10 |
| regex_section | pred_chars | 5411.200 | 3815.419 | 10 |
| regex_section | rouge1_f | 0.459 | 0.275 | 10 |
| regex_section | rouge2_f | 0.359 | 0.296 | 10 |
| regex_section | rougeL_f | 0.411 | 0.312 | 10 |

## Interpretation

- `regex_section.rouge*_f`: how much of the gold span our regex section parser recovers (lexical overlap).
- `llm_gap.recovery_at_τ`: fraction of LLM-extracted gap sentences whose max cosine to a gold-section sentence ≥ τ — high = the LLM points at real future-work content.
- `llm_gap.hallucination_at_τ`: fraction whose max cosine to *any* sentence in the source paper < τ — high = the LLM is inventing content not in the paper.