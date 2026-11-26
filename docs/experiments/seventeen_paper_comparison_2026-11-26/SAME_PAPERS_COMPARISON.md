# Five-paper versus 17-paper calls on the same test papers

Post-hoc comparison on exactly the 17 shorter test papers chosen for the 17-paper run. The earlier five-paper results are filtered to those papers; their original calls also included other papers, while the 17-paper calls group all 17 selected papers and reorder them by length. The fine-tuned NLI comparison uses the previous five-paper **surrounding-context** layout, matching the 17-paper run. The zero-shot five-paper results used the earlier target-then-context layout, so output differences there confound grouping and context layout. The full-paper format was the same. All runs used one Qwen verdict per target, Qwen3.8-27B FP8, temperature 0 and seed 0.

| Method | Grouping | TP | FP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|
| Full-paper | Up to 5 | 216 | 17 | 92.70% | 71.52% | 0.8075 |
| Full-paper | 17 | 219 | 21 | 91.25% | 72.52% | 0.8081 |
| Fine-tuned NLI | Up to 5 | 230 | 21 | 91.63% | 76.16% | 0.8318 |
| Fine-tuned NLI | 17 | 224 | 17 | 92.95% | 74.17% | 0.8250 |
| Zero-shot NLI | Up to 5 | 230 | 28 | 89.15% | 76.16% | 0.8214 |
| Zero-shot NLI | 17 | 225 | 23 | 90.73% | 74.50% | 0.8182 |

For fine-tuned NLI, of the changed predictions on these papers, the 17-paper request gained 9 labeled true positives and lost 15; it added 5 labeled false positives and removed 9. Thus it gained precision at the expense of recall relative to the same papers in the earlier groups, and its F1 decreased. The large difference from the earlier 34-paper five-group funnel precision of 74.67% is principally a sample-composition issue: the 17 selected shorter papers had 91.63% precision even in the previous five-paper context. This is an arithmetic decomposition, not proof of why particular papers are easier or whether batch size alone caused decision changes.

These comparisons are exploratory: the data are previously examined and LLM-labeled; the 17-paper subset is length-selected; group composition and ordering change; each condition is a single run; and 5-paper contexts included papers outside the subset. The four biomedical and 13 arXiv papers should not be treated as representative or as independent of their batch context.
