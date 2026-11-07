| Dataset | Setup | Prompt | Precision | Recall | F1 | F1 minus funnel | Input tokens | Run time | Failed calls | Empty answers |
|---|---|---|---|---|---|---|---|---|---|---|
| SAL-40 | A | keep_biased | 54.2% (46.7–62.5) | 85.4% (80.0–90.1) | 0.663 (0.602–0.724) | -0.010 (-0.062 to +0.040) vs `pubmedbert+qwen_rct_loose` | 346,445 | 1.1 min | 0 | 0 |
| SAL-40 | A | neutral | 56.0% (49.2–63.3) | 84.4% (78.3–89.4) | 0.673 (0.616–0.731) | +0.000 (-0.051 to +0.053) vs `pubmedbert+qwen_rct_loose` | 345,525 | 1.1 min | 0 | 0 |
| SAL-40 | A | strict | 64.1% (56.1–72.7) | 77.6% (71.3–83.4) | 0.702 (0.644–0.759) | +0.029 (-0.031 to +0.090) vs `pubmedbert+qwen_rct_loose` | 347,685 | 1.1 min | 0 | 0 |
| SAL-40 | B | keep_biased | 45.4% (38.7–53.0) | 89.8% (84.3–94.6) | 0.603 (0.536–0.669) | -0.070 (-0.130 to -0.012) vs `pubmedbert+qwen_rct_loose` | 363,130 | 1.5 min | 0 | 32 |
| SAL-40 | B | neutral | 47.9% (40.0–56.4) | 88.8% (83.4–93.5) | 0.622 (0.548–0.696) | -0.051 (-0.116 to +0.015) vs `pubmedbert+qwen_rct_loose` | 360,807 | 1.5 min | 0 | 34 |
| SAL-40 | B | strict | 56.9% (48.2–66.3) | 84.9% (79.2–89.8) | 0.681 (0.610–0.749) | +0.008 (-0.050 to +0.073) vs `pubmedbert+qwen_rct_loose` | 366,261 | 1.4 min | 0 | 39 |
| SAL-40 | C | keep_biased | 39.3% (34.4–44.6) | 93.7% (89.5–97.2) | 0.554 (0.502–0.607) | -0.119 (-0.157 to -0.080) vs `pubmedbert+qwen_rct_loose` | 463,811 | 3.2 min | 0 | 257 |
| SAL-40 | C | neutral | 45.4% (39.5–51.9) | 90.7% (86.5–94.8) | 0.605 (0.547–0.662) | -0.068 (-0.118 to -0.020) vs `pubmedbert+qwen_rct_loose` | 454,634 | 3.1 min | 0 | 264 |
| SAL-40 | C | strict | 50.6% (43.5–58.6) | 87.3% (82.2–91.6) | 0.640 (0.575–0.706) | -0.033 (-0.084 to +0.020) vs `pubmedbert+qwen_rct_loose` | 476,180 | 3.1 min | 0 | 284 |
| arXiv gold | A | keep_biased | 74.6% (63.8–85.6) | 73.6% (68.0–78.1) | 0.741 (0.678–0.795) | +0.101 (+0.027 to +0.172) vs `nli_r90+qwen_keep_biased` | 552,686 | 1.9 min | 0 | 0 |
| arXiv gold | A | neutral | 83.1% (76.8–90.6) | 68.1% (59.8–75.3) | 0.749 (0.696–0.790) | +0.108 (+0.025 to +0.173) vs `nli_r90+qwen_keep_biased` | 552,042 | 1.8 min | 0 | 0 |
| arXiv gold | A | strict | 89.2% (84.7–95.1) | 54.8% (47.0–61.5) | 0.679 (0.620–0.724) | +0.039 (-0.058 to +0.114) vs `nli_r90+qwen_keep_biased` | 553,834 | 1.7 min | 0 | 0 |
| arXiv gold | B | keep_biased | 69.5% (61.6–79.6) | 76.1% (70.0–81.1) | 0.726 (0.673–0.774) | +0.086 (+0.028 to +0.137) vs `nli_r90+qwen_keep_biased` | 579,949 | 2.2 min | 0 | 31 |
| arXiv gold | B | neutral | 71.6% (63.9–81.0) | 72.3% (66.7–76.7) | 0.720 (0.672–0.765) | +0.079 (+0.027 to +0.129) vs `nli_r90+qwen_keep_biased` | 577,258 | 2.2 min | 0 | 33 |
| arXiv gold | B | strict | 84.4% (75.4–93.1) | 59.6% (54.5–64.8) | 0.699 (0.646–0.745) | +0.058 (-0.013 to +0.116) vs `nli_r90+qwen_keep_biased` | 584,746 | 2.0 min | 0 | 42 |
| arXiv gold | C | keep_biased | 61.6% (52.5–71.4) | 81.8% (77.1–85.2) | 0.703 (0.636–0.768) | +0.062 (+0.009 to +0.109) vs `nli_r90+qwen_keep_biased` | 740,338 | 4.6 min | 0 | 310 |
| arXiv gold | C | neutral | 68.2% (60.6–76.4) | 76.9% (72.0–81.0) | 0.723 (0.672–0.773) | +0.082 (+0.026 to +0.136) vs `nli_r90+qwen_keep_biased` | 728,378 | 4.6 min | 0 | 320 |
| arXiv gold | C | strict | 81.5% (73.4–89.4) | 64.5% (58.8–69.5) | 0.720 (0.668–0.762) | +0.080 (+0.001 to +0.145) vs `nli_r90+qwen_keep_biased` | 761,658 | 4.5 min | 0 | 371 |
| FWS | A | validate_fws | 57.7% (50.7–65.0) | 87.3% (80.5–92.7) | 0.694 (0.632–0.751) | pending | 183,250 | 1.5 min | 0 | 79 |
| FWS | C | validate_fws | 60.1% (53.6–66.8) | 88.2% (82.0–93.2) | 0.715 (0.659–0.766) | pending | 100,540 | 0.8 min | 0 | 6 |

Funnel, rescored from its saved verdicts:
- SAL-40, `pubmedbert+qwen_rct_loose`: P 0.547, R 0.873, F1 0.673
- arXiv gold, `nli_r90+qwen_keep_biased`: P 0.513, R 0.854, F1 0.641

arXiv gold, recall by gap type:

| Setup | Prompt | Limitation | Future work | Research gap |
|---|---|---|---|---|
| A | keep_biased | 59.9% | 84.9% | 77.8% |
| A | neutral | 56.6% | 89.0% | 68.9% |
| A | strict | 46.1% | 87.7% | 51.3% |
| B | keep_biased | 61.2% | 94.5% | 79.1% |
| B | neutral | 59.9% | 90.4% | 74.2% |
| B | strict | 46.1% | 89.0% | 59.3% |
| C | keep_biased | 61.2% | 95.9% | 88.7% |
| C | neutral | 55.9% | 93.2% | 83.4% |
| C | strict | 43.4% | 89.0% | 69.2% |
