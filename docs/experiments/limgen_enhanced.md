# LimGen Stage-B enhancements (limitation detection, held-out test)

Targets the recall gap vs fine-tuned DistilBERT (0.629). LimGen own split, leakage-clean. Rows appended as computed.

| method | precision | recall | F1 | note |
|---|---|---|---|---|
| baseline bge+logreg @0.5 | 0.528 | 0.691 | 0.599 |  |
| #2 bge+logreg, tuned threshold | 0.607 | 0.599 | 0.603 | thr=0.6 |
| #3 stacking [bge+tfidf+cue] @0.5 | 0.55 | 0.734 | 0.629 |  |
| #3 stacking, tuned threshold | 0.569 | 0.717 | 0.635 | thr=0.525 |
| #1 DistilBERT fine-tuned (8000 train) | 0.549 | 0.874 | 0.674 | 1 epoch(s) |
