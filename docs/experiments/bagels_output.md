# BAGELS extraction benchmark (ACL, verbatim gold coverage)

Backend api (openai/gpt-4o-mini); 968 judged in 139 calls. Match = token-containment >= 0.8 (lower bound on BERTScore-coverage).

| subset | papers | gold sents | coverage before | coverage after | preds before | preds after |
|---|---|---|---|---|---|---|
| ACL_23_with_limitation | 47 | 382 | 0.822 | 0.508 | 424 | 240 |
| ACL_24_with_limitation | 80 | 830 | 0.699 | 0.400 | 757 | 404 |
| **ALL** | 127 | 1212 | 0.738 | 0.434 | 1181 | 644 |

**Extraction recall (Stage A+B): 0.738** of verbatim gold limitation sentences recovered — the BAGELS-comparable coverage number.

Stage C then drops 45% of predictions (1181->644) and coverage falls to 0.434. This gold is limitations-only (no negatives), so Stage C — a precision filter — can only cost coverage here; its FP-removal value shows on full-paper benchmarks, not this one.

Caveats: sections are cleanly labelled (Stage-A localization is easier than on raw PDFs); verbatim token-containment is a lower bound on BAGELS' BERTScore-coverage; BAGELS gold includes non-canonical limitations-section sentences (method observations) that a precision filter drops.