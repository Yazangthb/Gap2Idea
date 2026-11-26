# Matched 17-paper requests

We compare full-paper Qwen, zero-shot NLI + Qwen, and fine-tuned NLI + Qwen using one request per method for the same 17 papers. Because all 34 full papers in either split exceed the model context even in two 17-paper batches, the 17 shortest papers by total sentence characters were selected separately within the original development and test splits, with ties broken by paper ID. Selection did not use labels but creates a shorter-paper bias. The development group contains 11 arXiv and 6 biomedical papers; the test group contains 13 arXiv and 4 biomedical papers.

All methods used Qwen3.8-27B FP8, the same neutral rubric and one boolean verdict per input sentence. Full-paper calls contain all sentences. Both NLI calls use their original Stage A and fixed NLI candidates, with previous and next original sentence around each target (neighbors clipped to 800 characters). Recall is measured over all LLM-labeled gaps in the 17 papers, including candidates removed before Qwen. Each of the six calls passed exact completeness validation on its first attempt with no extra IDs. Input and output token counts exclude NLI and other preprocessing.

## Test: 17 shorter papers

| Method | TP | FP | Precision | Recall | F1 | Sentences sent | Qwen input tokens | Qwen output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full paper | 219 | 21 | 91.25% | 72.52% | 0.8081 | 3,139 | 149,655 | 33,424 |
| Normal NLI → Qwen | 225 | 23 | 90.73% | **74.50%** | 0.8182 | 750 | 91,814 | 11,790 |
| Fine-tuned NLI → Qwen | 224 | 17 | **92.95%** | 74.17% | **0.8250** | 621 | 75,545 | 9,748 |

The fine-tuned funnel has 5 more true positives and 4 fewer false positives than full-paper Qwen, with 49.5% fewer Qwen input tokens. This is a small descriptive difference on a length-selected, previously examined test subset; do not claim it establishes a robust or significant advantage. The normal NLI has the highest recall at 74.50%.

For the **13 arXiv test papers**, full / normal / fine precision-recall-F1 is 87.95/71.22/78.71, 87.72/73.17/79.79, and 90.64/75.61/82.45 percent. For the **4 biomedical test papers** it is 98.65/75.26/85.38, 97.40/77.32/86.21, and 98.57/71.13/82.63 percent. The biomedical subset is too small to generalize; these are LLM-generated gap labels, not the separate SAL human limitation labels.

## Development: 17 shorter papers

| Method | TP | FP | Precision | Recall | F1 | Qwen input tokens |
|---|---:|---:|---:|---:|---:|---:|
| Full paper | 174 | 20 | **89.69%** | 70.16% | **0.7873** | 162,261 |
| Normal NLI → Qwen | 178 | 32 | 84.76% | 71.77% | 0.7773 | 97,829 |
| Fine-tuned NLI → Qwen | 201 | 82 | 71.02% | **81.05%** | 0.7571 | 81,275 |

The fine-tuned method did not increase precision on the development sample, illustrating sensitivity to paper groups. The selected test and development papers have prior exposure to experiments and development was used to select the model and prompt; both splits use LLM-generated labels and do not constitute fresh human-validated evidence. They include shorter papers only, and the changes in paper selection, composition, order and batch size mean these results do not isolate batch size from those changes.

`plan.json` contains paper IDs, prompt, model and input hashes; `preflight.json` records context checks; raw replies and `summary.json` contain per-domain counts and token use. A preceding attempt to put all 68 NLI-filtered papers in one request was stopped at the user's request before this matched comparison; no result from it is used here.
