# LimGen head-to-head — limitation sentence detection

Same data & split as LimGen (ACL papers). Binary: limitation-section sentence vs other. Trained on LimGen-TRAIN (2000 sents), evaluated on held-out LimGen-TEST (3976 sents, pos:neg 1:3). All methods trained fresh on the same split (the shipped head — trained on test+valid — is deliberately excluded to avoid leakage).

| method | precision | recall | F1 |
|---|---|---|---|
| DistilBERT fine-tuned (their approach) | 0.494 | 0.866 | 0.629 |
| bge-small + logreg (OURS) | 0.526 | 0.65 | 0.582 |
| tfidf + logreg (classical) | 0.546 | 0.558 | 0.552 |
| SPECTER frozen + logreg (domain encoder) | 0.452 | 0.641 | 0.53 |
| cue rules (ours, lexical) | 0.585 | 0.097 | 0.166 |

_Published references (different splits/domains, NOT directly comparable):_ RCT/PubMedBERT limitation detection F1 0.82 (biomed); Zhang et al. future-work recognition Macro-F1 0.91 (ACL).
