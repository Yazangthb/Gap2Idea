# Matched context-layout and full-paper comparison on 34 test papers

These are the same 34 gold test papers (28 arXiv and 6 biomedical) and the same eight paper groups from the preceding 68-paper comparison. The full-paper and current fine-tuned NLI results are reused unchanged from that run. Two new calls per group test the same fine-tuned NLI candidates with surrounding neighbors and with no neighbors. Every method uses Qwen3.8-27B FP8, the same neutral classification instructions, a boolean decision for each input ID, and one request per paper group; groups contain up to five papers. The fine-tuned NLI and Stage A are unchanged. Surrounding layout uses the previous sentence, the numbered target sentence, then the next sentence; neighbors are clipped to 800 characters as in the baseline. Full-paper input includes all numbered sentences without separate neighbor fields.

| Method | TP | FP | Precision | Recall | F1 | Target sentences | Qwen input tokens | Qwen output tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full papers | 458 | 83 | **84.66%** | 68.98% | **0.7602** | 11,741 | 611,338 | 144,104 |
| Fine NLI, current context | 501 | 180 | 73.57% | 75.45% | 0.7450 | 1,998 | 262,899 | 28,861 |
| Fine NLI, surrounding context | 510 | 173 | 74.67% | **76.81%** | 0.7572 | 1,998 | 260,900 | 24,660 |
| Fine NLI, no context | 506 | 218 | 69.89% | 76.20% | 0.7291 | 1,998 | 81,128 | 30,781 |

There are 664 labeled gap sentences across all original test sentences; recall includes losses before Qwen. Surrounding context found nine more labeled gaps and produced seven fewer false positives than the current context format, using 0.8% fewer input tokens. Versus full-paper Qwen it used 57.3% fewer Qwen input tokens and had 7.83 percentage points higher recall, but 9.99 points lower precision. No-context input saved 69.1% of Qwen input tokens compared to the current funnel layout but precision fell 3.68 points; compared with full-paper Qwen its precision fell 14.77 points. Input token counts exclude NLI and other preprocessing. The higher precision observed with no context on development did not repeat on test.

## Domain breakdown

| Domain | Papers | Full P / R / F1 | Current P / R / F1 | Surrounding P / R / F1 | No context P / R / F1 |
|---|---:|---|---|---|---|
| arXiv | 28 | 81.61 / 69.07 / 74.82 | 68.77 / 74.38 / 71.47 | 70.26 / 76.66 / 73.32 | 65.56 / 78.37 / 71.39 |
| Biomedical | 6 | 98.95 / 68.61 / 81.03 | 98.20 / 79.56 / 87.90 | 98.15 / 77.37 / 86.53 | 98.94 / 67.88 / 80.52 |

The biomedical sample is only six papers; labels here are LLM-generated three-type labels, not the separate SAL human limitation labels. The prior 34-paper development comparison yielded 83.93% / 73.25% precision/recall for the current funnel; its higher precision than the 73.57% / 75.45% test result is a real split difference but its cause is not established. Development choices were tuned using those papers; all of these test papers were also examined in earlier experiments. Thus the test is exploratory, not a fresh blind evaluation, and the LLM-derived labels are not human validated. One call per condition does not establish significance or characterize variability.

The original full/current calls were made previously with a 65,536 output-token cap; the new calls used 16,384 and all completed with one decision for every input ID. The exact prompt, group identities, input hashes, preflight, all raw responses, token usage and split/domain counts are recorded alongside this report. The desktop was restarted during execution; saved responses were validated and reused when continuing.
