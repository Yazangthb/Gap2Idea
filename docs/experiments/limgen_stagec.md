# LimGen Stage-C lift (limitation detection, held-out test, leakage-clean)

Stage B = bge-small+logreg trained fresh on LimGen-TRAIN (2400 sents); test 1856 sents (464 limitations). Stage C = batched LLM precision filter (yandex) over Stage-B positives.

| stage | precision | recall | F1 |
|---|---|---|---|
| Stage B | 0.57 | 0.744 | 0.645 |
| + Stage C | 0.845 | 0.504 | 0.632 |

Stage C judged 605 positives in 16 calls; dropped 328 (217 false positives removed, 111 true limitations lost).
