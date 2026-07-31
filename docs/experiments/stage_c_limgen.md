# Stage C on the LimGen yardstick (small sample)

LimGen limitation detection, held-out test sample (196 sents, 49 limitation). Same task/metric as the prior-art table. Stage C = LLM filter (Qwen/Qwen2.5-1.5B-Instruct), rule-protected.

| method | precision | recall | F1 |
|---|---|---|---|
| Stage B (bge+logreg+cue) | 0.561 | 0.755 | 0.643 |
| **Stage B + Stage C** | 0.786 | 0.673 | **0.725** |

ΔF1 +0.082; precision 0.561→0.786, recall 0.755→0.673. 61 LLM judgments.
