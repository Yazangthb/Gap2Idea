# What the paper says, and whether it holds up

Status on 2026-11-05. Full history: `PROJECT_INVENTORY.md`. Checks: `scripts/bench/audit_leakage.py`, `scripts/bench/audit_sensitivity.py`.

**Title:** Scalable Extraction of Author-Stated Research Gaps: Towards Automated Idea Generation
**Target:** MDPI *Machine Learning and Knowledge Extraction*. Authors: Yazan Alnakri, Osama Orabi, Hamza Aldaghstany; Rustam to be added.

## 1. The idea

Authors state limitations, future work and open problems in their papers, but sending whole papers to an LLM is costly and, in our test, missed most of them. We use a recall-first funnel in which cheap stages narrow the text and an LLM only filters what remains.

| Stage | Job | SAL-40 track | arXiv track |
|---|---|---|---|
| A. Localise | Drop text unlikely to hold gaps | Keep Discussion and Conclusion sections | Junk filter (no section gate, because sections don't help on arXiv) |
| B. Classify | Flag candidate sentences | ModernBERT NLI, "This is a limitation." | ModernBERT NLI, three short per-type hypotheses |
| C. Filter | LLM drops false positives, in three modes; high precision is the main one | Qwen3.8-27B, rubric narrowed to limitations | Qwen3.8-27B, full rubric |

NLI needs no training labels, so one method runs on every dataset; thresholds come from each dataset's dev data. Fine-tuned PubMedBERT is the supervised comparison on SAL-40: more precise (74.5% vs 66.2%) but trained on SAL's labels. Stage C is one self-hosted, pinned model that played no part in making the labels; the prompt matches each benchmark's labels.

**Contributions**
1. The funnel, evaluated end to end.
2. A 230-paper, three-type gold corpus, LLM-labelled with human validation pending.
3. Stage C as a tunable precision filter: one instruction moves gold precision from 56.3% to 86.6% (`STAGE_C_MODES_RESULTS.md`). The earlier "almost no recall cost" claim only holds for high recall mode, and high precision is now the main mode.
4. A negative result: section recovery doesn't lift arXiv recall.
5. An audit showing existing gap datasets are rarely reused for evaluation.

## 2. Datasets

| Dataset | Use in paper | Labels | Evaluation size |
|---|---|---|---|
| **SAL** (Lan 2024) | Main, human-labelled track | Human | 40 test papers, 205 limitations |
| **Our gold, arXiv part** | Main, scale and types track | Claude Opus + Sonnet + adjudicator | 28 test papers, 527 gaps |
| Our gold, biomedical part | Cross-check only: **same 40 papers as SAL** | Same LLM ensemble | 6 test papers, 137 gaps |
| **FWS** (Zhang 2023) | Supporting, CS future work | Human | 209 papers, 1,506 sentences |
| Sci-Sentence (Bolaños 2025) | Only if redone cleanly | Human | 140 sentences, 40 positive |

**Leave out:** LimGen, BAGELS, Kilicoglu (data not found), Aug section-pool 0.81 (predictions lost), early small golds, clustering and idea generation (next paper).

## 3. Results

| Experiment | Result | Evidence |
|---|---|---|
| **SAL-40 Stage A** | Keeps 1,885 of 7,968 sentences and 203 of 205 limitations | Row-level |
| SAL-40 Stage B, 8 classifiers | PubMedBERT 0.604 > SciBERT 0.553 > DistilBERT 0.523 > e5 0.457 > bge-base/mpnet 0.437 > bge-small 0.436 > MiniLM 0.371 | Row-level |
| **SAL-40 + Stage C (Qwen)** | **P 46.0→54.7%, R 87.8→87.3%, F1 0.604→0.673**, 179/180 kept | Verdicts saved |
| **SAL-40, three modes (Qwen, frozen rubric narrowed to limitations)** | **High precision P 74.5%, R 78.5%, F1 0.765**; balanced P 67.6%, R 84.4%, F1 0.751; high recall P 63.8%, R 86.8%, F1 0.736. High precision vs `rct_loose`: F1 +0.092 (CI +0.048 to +0.137). See `STAGE_C_MODES_RESULTS.md` | Verdicts saved |
| SAL-40 + Stage C (Yandex) | P 58.0%, R 76.1%, F1 0.658; paired difference vs Qwen not significant | Verdicts saved |
| **arXiv + Stage C (Qwen)** | **P 31.9→60.1%, R 72.3→68.7%, F1 0.443→0.641** (CI 0.570–0.706) | Verdicts saved |
| All gold test + Stage C (Qwen) | F1 0.457→0.665; **0.659 excluding the 196 selection candidates** | Verdicts saved |
| **Gold test, NLI Stage B + Qwen (candidate design)** | **P 55.8%, R 86.4%, F1 0.678**; F1 tied with bge-small + Qwen, 17.6 points more recall. arXiv F1 0.641, R 85.4% | Verdicts saved |
| Gold test, NLI + Qwen, rubric as a fixed setting | Keep-biased P 56.3%, R 85.7%, F1 0.679; **neutral P 68.2%, R 79.7%, F1 0.735** (arXiv 0.706); strict P 86.6%, R 57.4%, F1 0.690. Neutral vs keep-biased +0.055 (CI +0.025 to +0.080). Wording informed by test false positives: disclose | Verdicts saved |
| Biomedical gold cross-check | F1 0.521→0.772 (6 papers); 0.861 with NLI Stage B | Verdicts saved |
| Stage B alone on gold | ModernBERT NLI 0.617, fine-tuned bge-small 0.584, frozen bge-small 0.453 | Totals only |
| Section recovery, arXiv | LaTeX sections: recall 55.6→55.2%; 82% of misses in Methods, Experiments, Results | Log only |
| **SAL-40, NLI + Qwen (chosen funnel)** | **High precision P 66.2%, R 84.9%, F1 0.744**; dev check P 64.3%, R 80.0%, F1 0.713. Precision is 8.4 points below PubMedBERT (CI −11.8 to −4.5) | Verdicts saved |
| **FWS, NLI + Qwen, lowercased text** | **High precision P 76.5%, R 75.5%, F1 0.760**. Earlier Yandex F1 0.707 saw raw casing, which reveals the label: don't use it | Verdicts saved |
| Cost, NLI funnel vs one whole-paper call | About 3.08× fewer input tokens on SAL-40, 2.04× on FWS, 1.25× on gold (estimate). Replaces 3.74× | Row-level arithmetic |

## 4. Audit

### Clean: checked against code and data

| Check | Result |
|---|---|
| SAL paper overlap, train/dev/test (120/40/40) | **0** |
| SAL test limitations whose exact text is a train limitation | **0** |
| PubMedBERT training | Train split only, fixed 3 epochs, threshold 0.90 from dev, no test access (`h100_allmodels.py`, Oct 8) |
| Gold paper overlap, train/val/test (162/34/34) | **0** |
| Gold near-duplicate papers | **0** pairs sharing 5%+ of sentences |
| Gold sentences repeated across train and test | 39, all boilerplate; none has a gap label |
| Gold head and thresholds | Trained on train; encoder and τ chosen on val |
| ModernBERT and fine-tuned bge | Same paper split; τ from val |
| Stage C inputs | `gap_type` is predicted, not gold (matches gold on 363/480 gaps). No labels in the input |
| SAL labels in gold labelling | Not sent to the judges |
| `pipeline_head` training vs Sci-Sentence test, FWS, SAL-40 | **0** shared sentences |
| Sci-Sentence train vs test | **0** shared sentences |
| SAL fixture (7,968) vs `test.csv` (8,882) | The 914 dropped rows are under-4-word fragments; **0 are limitations** |
| Dropping abstracts from SAL Stage A | Harmless: worst case F1 0.673→0.665 |

### Must fix before submission

| # | Problem | Why it matters | Fix |
|---|---|---|---|
| 1 | `rct_loose` was **selected on the SAL test pool** (recorded Aug 14) | Both 0.658 and 0.673 use a prompt picked on those papers | Use the three-mode SAL results as the headline: their wording wasn't chosen on SAL results. Still worth running on SAL's 40 **dev** papers as a check. Retrain PubMedBERT first: checkpoints are gone, about 3 minutes |
| 2 | Sci-Sentence 0.880: **threshold and prompt were both chosen on test**; 40 positives | The strongest-looking number is the least valid | Drop it, or choose the threshold by cross-validation on Sci-Sentence train and use the frozen rubric |
| 3 | Whole-paper baseline is weak: 10 papers, 1 cheap model, 2 empty replies scored as no predictions, no output limit set | "Funnel beats whole-paper LLM" rests on it. Its A+B+C row also uses `validate_rct`, not the headline prompt | Rerun on all 40 papers on Qwen, separating failures from empty answers |
| 4 | **No human labels on the arXiv track**; 10 blind-study items are in a public CSV | Every arXiv number rests on labels from the Claude ensemble | Run the 2+1 study; replace the 10 items and decide on the CSV |
| 5 | ~~FWS result is on Yandex~~ Done 2026-11-05 | Rerun on Qwen with lowercased text, because raw casing reveals the label | See `STAGE_C_MODES_RESULTS.md`; disclose the casing problem |
| 6 | PubMedBERT and ModernBERT can't currently be regenerated | Saved outputs exist, but referees can't reproduce them. No paired test against ModernBERT is possible | Copy `h100_allmodels.py` into the repo; rerun ModernBERT and save per-sentence predictions |

### Disclose in the paper

| Issue | Measured effect or wording |
|---|---|
| Stage C setup chosen on 196 gold test candidates | F1 0.665→0.659 without them |
| **`research_gap` flatters "any gap" scores** | Without it, F1 falls 0.665→**0.515**; arXiv 0.641→**0.445** (P 35.7%, R 59.1%). The rubric wasn't aimed at this narrower scope. Report per-type results |
| Adjudicated labels are inclusive (51% of SAL extras confirmed by a third model) | High-confidence labels only: F1 0.628, P 53.8%, R 75.5% |
| Biomedical gold is SAL-40, RCT reports only | Say "CS papers and RCT reports", not general "cross-domain" |
| Rubric wording mirrors the gold guideline | Qwen is independent of the labels; the prompt's definitions aren't |
| PubMedBERT chosen among 8 on test | Report all 8; it's also Lan's published choice |
| Lan's 0.821 uses a different pool and prevalence | Not comparable; the replay is a sensitivity only. Don't place neighbour-credit 0.821 beside it |
| FWS has no real Stage A, and head training rows have no paper IDs | 0 shared sentences, but same-paper overlap can't be ruled out |
| Short fragments removed from SAL before scoring | 914 rows, none positive |

### Not verified

- Whether ModernBERT's hypothesis wording was tuned using test results.
- Which papers supplied `pipeline_head`'s training sentences.
- Which data the `DROP_SEC` section list was derived from.
- The `validate_fws` prompt being hand-written and untuned rests on an Aug 14 session note.

## 5. Decisions still open

0. Decided 2026-11-05: precision comes before recall. High precision is the main Stage C mode; report all three modes.
1. Decided 2026-11-05: Stage C is Qwen3.8-27B and Stage B is short-hypothesis ModernBERT NLI. The SAL-40 headline is NLI + Qwen in high precision mode: P 66.2%, R 84.9%, F1 0.744. PubMedBERT and bge-small are baselines.
2. Keep `research_gap` in the main claim. The title supports it, so report it with per-type results and the 0.515 sensitivity.
3. The public label CSV.
4. Abstract rewrite, after the must-fix runs.
