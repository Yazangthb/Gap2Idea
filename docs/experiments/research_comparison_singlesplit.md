# Airtight single-split comparison — LimGen limitation detection

One split for ALL methods (leakage-clean). train=1600 val=400 test=2108 (527 limitation / 1581 other, 1:3).

| method | kind | precision | recall | F1 | note |
|---|---|---|---|---|---|
| cue rules (lexical) | ours | 0.517 | 0.087 | 0.149 |  |
| BernoulliNB (Zhang repro) | prior | 0.678 | 0.224 | 0.337 |  |
| tfidf + logreg | classical | 0.56 | 0.626 | 0.591 |  |
| bge + logreg (OURS current) | ours | 0.561 | 0.715 | 0.629 |  |
| stacking[bge+tfidf+cue] (OURS) | ours | 0.729 | 0.562 | 0.635 | thr=0.68 |
