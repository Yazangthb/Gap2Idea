# Revision plan before submitting to MAKE

Written 16 November 2026. Base: branch `submission-make` at `909d865`.
Source: the readiness review of 16 November (independent reviewer read + checks against the repository).
This version includes the points from a second, independent review of the plan itself (OpenAI Codex,
`gpt-6.1-sol`, high reasoning effort, 16 November). Section 7 lists what that review changed.

**Goal.** Submit a version that a strict reviewer would send back with *minor* revision, not *major*.
**What that takes.** One day of preparation (code, ethics, missing files), three required experiment
blocks (no new labelling), the human study, the text changes and the admin items. Existing raw results
are never rerun or replaced.

**People.** The split follows who wrote what:

| Person | Role in this revision | Why |
|---|---|---|
| **Hamza** | Baseline runner and baseline experiments (P1, X1, X2) | He ran the whole-paper baselines (task 5a) and knows `baseline_qwen.py` |
| **Yazan** | Funnel experiments (X3, X5), human study (P3, P4, H1, X4), missing provenance files (P5), final regeneration of tables | He wrote the funnel code, owns the server setup, and has the human-study key |
| **Osama** | Scoring code (P2), manuscript text, baselines that need no GPU, data and licence audit, admin and submission package, integration of all new results | He did the Task 5b baselines and the last text revision |

Decisions about authors and CRediT roles need **all** authors, not one person (see P6).

---

## 0. Rules for everyone

1. **Freeze before you run.** For every new experiment, commit a short plan note first: what runs,
   which papers, which metrics, and the hashes of the prompts and data files, the model revision and the
   commit of the code. `baseline_qwen.py` only checks that its plan file is committed, not that code and
   prompts match it, so the note must carry these hashes.
2. **Raw evidence is never overwritten.** Each new run gets a fresh folder. `baseline_qwen.py` reuses a
   file name and resumes from it, so never point a new configuration at an existing folder. Existing
   verdict files stay as they are. The *generated* manuscript tables may change; that is expected.
3. **Same settings as the paper.** Same prompts, temperature 0, seed 0, no reasoning mode, all three
   modes, same thresholds (gold τ = 0.354). No tuning on the new data.
4. **Everything is reported.** Whatever these runs show goes into the paper, including results that
   weaken a claim. Inclusion rules are fixed before a result is seen. We do not add experiments in order
   to reach significance.
5. **Scoring stays the same.** Sentence-level precision, recall and F1; 95% intervals from 2,000
   resamples of papers; paired differences on identical sentences, always written as **funnel minus
   baseline**.
6. **Careful conclusions.** A new run tells us about the configurations we tested. The baselines differ
   from the funnel in more than one way (type information, context, answer format; the answer format
   alone changed 7.6–9.3% of verdicts in `stage_c_call_shape.md`). So we do not write "X is the cause"
   unless a control isolates X. "No significant difference" is not "equal".
7. Work on `submission-make`. Push small commits. Don't push the human-study key or answers.

---

## 1. Day 0 preparation (before any experiment or rating)

| # | Who | What | Why |
|---|---|---|---|
| P1 | Hamza | **Make the runner able to run the new matrix.** `whole_paper_other_llm.py` hardcodes setup A, seven dataset/prompt pairs and the old `validate_fws` prompt for FWS; it loads `baseline_qwen.py` from another git ref and skips the plan check. Give it parameters for model, setup, prompt, dataset and output folder; use the local, pinned `baseline_qwen.py`; keep the plan check. Smoke-test each new configuration on a few **development** papers. Write the exact commands into the plan note. | Without this, X1 and X2 cannot run for 9B and 4B, and an edit to the local script would not reach the code that actually runs |
| P2 | Osama | **Scoring and table code, before the runs.** `baseline_qwen.py score` compares SAL with an older funnel (PubMedBERT + `rct_loose`) and gold with high-recall only, and prints baseline minus funnel; `llm_size_compare.py` hardcodes the old FWS file. Write one scorer that pairs by dataset, model, mode and input type, and prints funnel minus baseline. For the SAL "Stage A sentences only" baseline, score on **all 7,968 sentences and 205 limitations**, counting sentences outside Stage A as not kept. Test it on the existing 27B files: it must reproduce the published numbers. | Otherwise the new results are compared with the wrong funnel |
| P3 | Yazan | **Ethics and consent first.** Ask the institution whether the rating task needs ethics review or is exempt, and get that in writing. Prepare a short information and consent text for raters. No recruiting before this. | MAKE asks for the approval or the exemption basis, and for a consent statement. `main.tex` says "Not applicable" for both today |
| P4 | Yazan | **Human-study files.** Find `data/gold/human_study/`, check the key hash starts with `f8f55f1b`, and commit the analysis amendment (see H1) before any answer arrives. Freeze the X4 design now too, if X4 is done. | The bundle is on one machine only |
| P5 | Yazan | **Missing provenance files.** Push `docs/experiments/nli_variants.json` (table scripts fail without it) and `extras_triplecheck.json` with the model actually used for each item (needed for T8). If a file cannot be published, write down the exact way to rebuild it. | The tables must rebuild from a fresh clone on a second machine, not only on one laptop |
| P6 | All | Author list (supervisor?), corresponding author, CRediT roles; what Rubetek's support was (Funding or Acknowledgments); whether the Zenodo preprint was posted (DOI). | Needed for the back matter and the cover letter |

---

## 2. Hamza: baseline experiments

### X1. Like-for-like whole-paper baselines (required; a few GPU-hours)

*Why:* today the FWS whole-paper baseline uses a different, keep-biased prompt, and on SAL the funnel
sees section labels that the baseline does not. A reviewer cannot tell what the comparison measures.

| Step | What to do | Output |
|---|---|---|
| X1.1 | Add the FWS rubric runs to the frozen plan in `docs/experiments/baseline_qwen.md` (it is listed there as "open question 1") and to `RUN_PLAN`. Commit. | plan commit |
| X1.2 | FWS, setup A (whole excerpt), prompts `strict`, `neutral`, `keep_biased` (the files `prompts/fws_*.txt` already exist), for 27B, 9B and 4B. 27B: `python scripts/bench/baseline_qwen.py run --dataset fws --setup A --prompt strict --split test` (repeat per prompt). 9B/4B: with the runner from P1. | 9 verdict files |
| X1.3 | SAL, new setup "A, Stage A sentences only": the whole-paper call receives only the sentences in Discussion and Conclusion sections (the same sentences the funnel keeps after Stage A), numbered with their original ids. Small code change in `baseline_qwen.py`; add it to the plan and commit before running. Three prompts × three models. | 9 verdict files |
| X1.4 | Score with the P2 scorer. | result rows |

*Done when:* Table 8 can show FWS with one whole-paper result **per mode**, and SAL has a second
whole-paper row "with Stage A sentences only".

### X2. The LLM alone on short inputs, for the small models

*Why:* this tests the paper's central claim. With the 27B model, the chunked and every-sentence
baselines beat the funnel on arXiv (F1 0.70–0.73 against 0.64–0.71). They were never run for 9B and 4B.
Without them, the small-model gain could simply come from shorter inputs.

| Step | What to do | Priority |
|---|---|---|
| X2.1 | Add the runs to the plan and commit. | required |
| X2.2 | Setup **C** (20 consecutive sentences per call), datasets `gold` (arXiv) and `sal`, three prompts, models 9B and 4B: 12 runs. This is the closest match to the funnel's 20 candidates per call. | **required** |
| X2.3 | Setup **B** (chunks of 100 sentences), same grid: 12 runs. | optional; drop first if time is short |
| X2.4 | Score with the P2 scorer. Log input tokens, output tokens and time. Failed and retried calls: record their tokens when the server returns them; otherwise mark the cost as "missing", not zero (today tokens are summed over successful calls only). | required |

*Done when:* for each of 9B and 4B we can state the paired difference (funnel minus "the same LLM on
every sentence in batches of 20") with its interval, on arXiv and SAL-40. If the interval contains
zero, we report "difference not resolved", not "equal".

**Hand-over to Osama:** the verdict files, the run manifest (hashes from rule 1), a coverage check
(every sentence answered, number of failed calls) and the scorer commit. Not only result rows.

---

## 3. Yazan: funnel experiments, human study, final regeneration

### X5. Call-format control: zero-shot funnel with one call per paper (required; small)

*Why:* the paper's one-call-per-paper funnel changes two things at once (fine-tuned Stage B, and one
call per paper). The reviewer asked for the one-call format with the zero-shot classifier.

- **Already done, not yet in the paper:** 27B on the gold test split
  (`docs/experiments/stage_c_per_paper/test_paper_{keep_biased,neutral,strict}.jsonl`, described in
  `stage_c_call_shape.md` §2; 34 papers, 2,522 candidates). Reuse these verdicts; do not rerun.
- **Run only what is missing:** the same variant for 9B and 4B on gold, and for all three sizes on
  SAL-40 and FWS. Candidates only, so the cost is low.
- Plan note first, as always.

### H1. Human study (required; about 2.5 rater-hours)

Follow `docs/publication/HUMAN_STUDY_INSERT.md`, after P3 and P4.
1. Recruit two raters and one tie-breaker. No authors; nobody who has seen the labels or the funnel output.
2. Send one page per rater, collect answers, build the tie-breaker page, run the analysis.
3. Hand `docs/experiments/human_study_results.md` to Osama.

**Analysis amendment, committed before any answer arrives.** The frozen sample and the two primary
tests stay exactly as they are. The amendment only adds:
- The weights use the original group sizes (B2 281, S1 174, S2 77), but sampling was from the eligible
  items (267, 161, 69). State the assumption that excluded items behave like eligible ones, and add a
  sensitivity bound for the excluded items.
- The current variance is zero when all answers in a group agree. Add a conservative interval for that case.
- These are labelled supplementary. Nothing is changed after ratings are seen.

### X3. Retrospective check on the 134 arXiv training papers (recommended; about 1 GPU-day)

*Why:* the arXiv test set has only 28 papers, and it was used during development (the 196 candidates;
the high-precision wording was written after reading false positives from all 34 test papers). The 134
arXiv training papers already have LLM labels.

*What it is not:* a clean held-out set. `stage_a_latex_sections.md` shows a development analysis over
175 of the 190 arXiv papers that informed the decision to drop section filtering on arXiv. So:

| Step | What to do |
|---|---|
| X3.0 | **Exposure audit.** List every experiment note and script that read these 134 papers or their labels, and what decision it informed. Commit it with the plan note. The paper describes X3 with exactly this audit. |
| X3.1 | Plan note: the 134 papers; Stage A junk filter; zero-shot NLI with τ = 0.354; Stage C unchanged; three modes; 27B, 9B, 4B; whole-paper setup A with the same prompts. The fine-tuned Stage B is **not** evaluated here, because it was trained on these papers. This needs a new split/export in the loaders; smoke-test first. |
| X3.2 | Zero-shot NLI scores (`nli_score_remote.py`). |
| X3.3 | Main funnel (20 candidates per call) × 3 modes × 3 models. |
| X3.4 | Whole-paper prompting × 3 modes × 3 models. |
| X3.5 | Score with the P2 scorer; output in a new folder, e.g. `docs/experiments/retro_train134/`. |

*Done when:* there is a second arXiv table (134 papers) with the same columns as Table 8.
*Power:* a rough projection from resampling the 28 test papers suggests the 9B difference has about a
1-in-3 chance of a significant interval with 28 papers and a much better chance with 84 or more. This
is an approximation, not a promise. Osama commits the script (`power_arxiv.py`) so the number can be
checked; the paper does not predict significance.

### X4. Disagreement audit: funnel versus whole-paper prompting (optional; about 2.5 rater-hours)

*Why:* with 27B on arXiv, whole-paper prompting scores higher than the funnel, but the labels were made
by LLMs reading whole papers. People can judge the sentences where the two systems disagree.

*Scope:* a **descriptive audit**. It reports the share judged a real gap in each group, with intervals.
It cannot show that the F1 advantage is "real" or not, because the groups have different sizes and the
audit does not re-estimate recall. Do it only if the design below is frozen on Day 0 (P4):
- exact funnel variant; the models (27B and 4B); how many sentences per model and direction
  ("only the funnel kept", "only whole-paper kept"); the size of each group in the population;
- no sentence that is already in H1; duplicates removed; same context and question as H1; same
  two-rater and tie-breaker rule; seed and sample hash committed;
- given to the same raters only **after** they finish H1.

### Final regeneration

After all included results are integrated (not before), regenerate Tables 8–10. Then one other person
rebuilds them from a **fresh clone on a second machine** (this is why P5 matters).

---

## 4. Osama: scoring code, text, baselines, data audit, admin, integration

### T. Text changes that can start today

| # | Change | Where |
|---|---|---|
| T1 | **Abstract:** say "34 test papers" of the 230-paper corpus; "on full papers" after "7–8 times fewer input tokens" (it is 2.7× on FWS); "equal or higher observed F1" instead of "matches or improves"; add one concluding sentence; stay ≤ 200 words. | `main.tex`, `preprint.tex` |
| T2 | **Small-model claim:** replace "small models gain most" and "its advantage grows as the model gets smaller". The gain appears with the two smaller Qwen3.5 models and does not grow from 9B to 4B (one-call variant on arXiv: +0.28/+0.39/+0.20 for 9B, +0.19/+0.15/+0.26 for 4B). Mention format differences too (27B: FP8 with speculative decoding; 9B/4B: bf16). | Abstract, Intro, Discussion, Conclusions |
| T3 | **Mode control:** "on every dataset" → "with the 27B model". Report as a finding that the order breaks for smaller models in the one-call format (precision 4B: 46.4 / 40.3 / 46.3%; 9B: 51.9 / 55.3 / 53.7% for high recall / balanced / high precision). | Abstract, Intro, Results 4.3, Discussion |
| T4 | **"Untouched papers" is wrong.** The high-precision wording drew on false positives from all 34 test papers. Call them "the 7 papers that supplied none of the 196 development candidates". | Results 4.3, Limitations |
| T5 | **Statistics wording:** "within 0.021 F1" → "no significant difference; the interval rules out a deficit larger than 0.053". Remove or flag the intervals on the 6 biomedical papers. Say the length split has no intervals. | Results, Intro, Conclusions |
| T6 | **RQ3 wording:** "In our data they could not" → the two label sets disagree, and the data do not show which one is right. Soften the heading of Section 5.4 (it rests on an analysis that was not repeated). | Conclusions, Discussion |
| T7 | "The same funnel runs on biomedical and computer-science papers": add that τ needs development data for each dataset and Stage A is dataset-specific. | Intro |
| T8 | **Label quality, from the datasheet:** LLM-label precision against SAL 0.456; κ per type (0.748 limitation, 0.869 future work, 0.725 research gap). The "116 of 227 confirmed" check goes in **only after** P5 recovers the item-level file and the model used (the datasheet allows a fallback to Opus 5.5, the same model that adjudicated). Describe it as agreement by another model, not as confirmation by people. Also update `data/gold/README.md`: it still has old human-study instructions and says SAL incompleteness is established. | Methods 3.2, Discussion 5.3, datasheet |
| T9 | **Methods details:** labelling model ids, settings and where the guideline is; how the Stage A rules were chosen; versions of PyTorch, Transformers, sentence-transformers, scikit-learn; **time and hardware for Stages A and B** (the cost table covers the LLM stage only today); one sentence that analysis code was written with AI assistance. | Methods |
| T10 | **Related work (keep it short):** two or three sentences on LLM idea-generation systems (e.g. SciMON, ResearchAgent, AI Scientist), because the title says "Towards Automated Idea Generation", and a concise paragraph on the validity of LLM-made annotations. Check each reference against its source. | `02_related_work.tex`, `references.bib` |
| T11 | **SAL's protocol and Lan et al.** Read Lan et al. again and state how SAL's sentences were chosen for annotation (whole paper, or sentences pre-selected from certain sections). If annotation was section-restricted, qualify every "false positive on full papers" statement. 203 of 205 limitations are in Discussion or Conclusion. Also check our sentence "trained with data augmentation, because we did not have its augmentation recipe" (Methods, baselines paragraph): the second review says Lan's *binary detector* was trained on an earlier 2,257-sentence set and augmentation belongs to the *type* classifier. Correct it if so, and say exactly how our PubMedBERT differs from theirs. | Methods 3.2 / 3.5 |
| T12 | Add the zero-shot one-call results that already exist (27B, gold test; see X5) to the format table and text. | Results 4.4 |

### B. Baselines and tables that need no GPU

| # | Change |
|---|---|
| B1 | Add **PubMedBERT alone** to Table 7 from the saved predictions (`sal_test_allmodels.json`): F1 0.495 on full papers, 0.604 after the section rule (funnel 0.744). Extend `paper_baselines_table.py`. Only its dev-tuned decisions are saved, so a "best-F1 threshold" row is not possible; say so. |
| B2 | Treat failed and degenerate baselines the same way: keep the naive Bayes rows with a note, and list the two failed SciBERT/BiomedBERT runs in a table note instead of only in the text. |
| B3 | **Precision and recall for Table 8.** With 27B in high-precision mode on arXiv, whole-paper prompting has higher precision (89.2%) and recall (54.8%) than the funnel (83.4%, 53.3%); the text must say so. Table 8 is already wide: if P and R do not fit at a readable size (≥ 8 pt), put the full P/R table in the supplementary material and keep F1 plus the key P/R numbers in the text. |
| B4 | New small table for the chunked, every-sentence, ids-only, one-call zero-shot and abstract variants with intervals (27B numbers exist; add the 9B/4B rows when ready). |
| B5 | Optional, lowest priority: FWS fine-tuned SciBERT from the Task 5b server job. `import_task5b_baselines.py` has no FWS SciBERT entry yet, so it needs importer support, and the rule "report it whatever it shows" is written down before the result is seen. |
| B6 | Commit `power_arxiv.py` (the projection behind X3) under `scripts/bench/`. |

### D. Data, licences and reproducibility

| # | Change |
|---|---|
| D1 | **Audit every released file that contains paper text,** not only `data/paper_texts.jsonl`: also `arxiv_latex_blocks.jsonl`, the annotation pages and worksheets, the `data/bench*` folders, and `data/gold/gap_gold_final.jsonl` (it holds 3,751 arXiv and **711 SAL** sentences). For each file record the source, the licence or permission, and the attribution. arXiv licences come from arXiv's metadata (OAI-PMH). |
| D2 | **Fix the Data Availability Statement.** It says SAL "is not redistributed", but SAL sentences are in the gold file. Check SAL's licence: either it allows this with attribution (then say so), or those sentences are replaced by ids and rebuilt by script. |
| D3 | Where text must be replaced by a rebuild script: pin source versions, extraction dependencies and hashes, and verify that the rebuilt sentence ids and text are identical before removing anything. |
| D4 | Fix the two DOIs that lack `v1/` (`mishra2026tree`, `thaepprasit2026limits`); add a DOI or URL to `sileo2024tasksource`. |
| D5 | Baseline training code lives on branch `Task_5b`. Cite the **commit** the imported predictions name (`3751532`), not the branch, or merge the needed scripts. |
| D6 | At the end, update both the availability statement and the repository citation to the final tag. |

### A. Admin and submission

| # | Change |
|---|---|
| A1 | Back matter: CRediT text, Funding, Conflicts of Interest (after P6). |
| A2 | GenAI sentence: add **Claude Fable 5.2** (used for the 11 November revision, the baseline tables and Figure 1) and **Codex with its model version** (`gpt-6.1-sol`, also used to review this plan). |
| A3 | **Both** statements for the human study: Institutional Review Board (approval or exemption basis, from P3) and Informed Consent. Thank the raters. |
| A4 | Cover letter (`COVER_LETTER_MAKE.md`): fill in, add the preprint DOI if posted. |
| A5 | Suggested reviewers (3–5, no conflicts) and the fee (CHF 1800 after acceptance; check for an institutional discount). |

### I. Integration (after the experiments)

Put the new results into the tables and text, then update the abstract, the answers to the research
questions, the limitations and the conclusions. See Section 6 for what each outcome changes.

---

## 5. Order and timing (proposal)

The first version of this plan said one week. That was not backed by anything: the runner and scorer
need changes first, and the raters and the ethics answer are outside our control. A realistic target is
**required items in about one week after Day 0 is complete, optional items in a second week**. Re-estimate
after the development smoke runs.

| When | Hamza | Yazan | Osama |
|---|---|---|---|
| Day 0 (Mon 16 Nov) | P1 runner + smoke tests; plan notes for X1, X2 | P3 ethics question; P4 study files + amendment; P5 missing files; plan note for X5 | P2 scorer; start T1–T7 |
| Days 1–2 | X1, then X2 setup C | Recruit and send H1 pages (once P3 is answered); X5 | T8–T12, B1–B3, B6; start D1–D2 |
| Day 3 | Finish X2 C; hand over | H1 analysis when answers are in | B4; integrate X1, X2, X5 |
| Days 4–5 | Check Osama's use of the baseline numbers; X2 setup B if time | X3.0 audit and X3 runs, if done; X4, if frozen on Day 0 | Integrate H1; D3–D6; A1–A5 |
| After all results are in | Read the full PDF | Regenerate Tables 8–10 | Second-machine rebuild; build the package |
| Last step | All three: read the final PDF, approve, tag `submission-v1`, submit | | |

The two H100 GPUs are shared: agree who uses which before starting (e.g. Hamza GPU 0, Yazan GPU 1).

**If only half can be done, keep:** P1–P6; H1 with its amendment; X1; X2 setup C; X5 (mostly existing
results); PubMedBERT-alone rows (B1); all claim corrections (T1–T7, T11, T12); the data and licence
audit (D1–D5); the submission essentials (A1–A5).
**Drop first, in this order:** B5 (FWS SciBERT); X2 setup B (100-sentence chunks); X4; X3 (then the
test-split influence and the 28-paper limit stay as stated limitations).

---

## 6. What each result would change in the paper

All statements below are about the configurations we tested (rule 6).

| Result | If it comes out this way | Then the paper says |
|---|---|---|
| X2 | The every-sentence baseline is clearly below the funnel with 9B/4B | With these models the funnel scores higher than the same LLM on whole papers **and** on short batches of all sentences. Shorter input alone does not account for the gain in these runs |
| X2 | The every-sentence baseline is close to or above the funnel with 9B/4B | The small-model gain is not specific to the funnel; short inputs give it too. Report the measured trade-off in tokens and time (including Stages A and B) instead of an accuracy advantage. Rewrite contribution 2, the RQ2 answer and the abstract |
| X1 (FWS) | The funnel is still ahead with matched prompts | Keep the claim; replace the "broader prompt" caveat with the matched result |
| X1 (FWS) | The difference is no longer significant | Say that the earlier FWS difference was measured against a different prompt and is not found with matched prompts |
| X1 (SAL) | Whole-paper prompting on Stage A sentences is close to the funnel | Most of the SAL difference is also obtained by giving the baseline the section rule; say so. Stages B and C then matter for cost on SAL, not for F1 |
| X5 | Zero-shot one-call results | Report next to the fine-tuned one-call results, so the reader sees the effect of the call format and of fine-tuning separately |
| X3 | Same pattern on 134 papers | Report as a retrospective check on a larger set, with the exposure audit. Not "independent confirmation" unless the audit is clean |
| X3 | Different pattern | Report both. A difference does not by itself show that development shaped the 28-paper result; the two sets also differ in papers |
| H1 | Decoy gate passed and arXiv lower bound above 70% | Report the measured human precision of kept sentences on arXiv, per group. Corpus labels for sentences the funnel did not keep, and recall, remain untested and stay "provisional" |
| H1 | Decoy gate passed and the exact test on S2 is significant (with 40 answers that needs **26 or more** judged limitations, not 21) | In the sampled disagreements, most sentences that SAL did not mark are limitations according to the raters. Limit the claim to that sample |
| H1 | Otherwise | Say what the study found; no claim |
| X4 | Any outcome | Report the share judged real in each group with intervals, as a description of the disagreements |

---

## 7. What the second review changed in this plan

Checked against the repository before accepting. Accepted:
1. Day 0 preparation added: the small-model runner and the scorer cannot run or score the new matrix as they are (P1, P2).
2. Ethics and consent moved to Day 0, before recruiting; both statements in A3 (P3).
3. Zero-shot one-call results for 27B already exist; reuse them, run only the missing cells, and make this a separate required block (X5) instead of part of optional X3.
4. X3 renamed from "held-out set" to "retrospective check", with an exposure audit; no predicted significance.
5. X4 narrowed to a descriptive audit with a design frozen on Day 0.
6. H1: supplementary analysis amendment before answers; outcome wording tied to the exact tests (26 of 40).
7. Section 6 rewritten without causal claims; "equal" replaced by "difference not resolved".
8. Data audit widened to all text-bearing files; the "SAL is not redistributed" statement is inaccurate.
9. Private-machine dependency removed (P5, second-machine rebuild); Task 5b cited by commit.
10. T8 gated on recovering the triple-check file; T11 widened (SAL protocol, Lan et al. sentence).
11. Cost accounting includes Stages A and B and failed calls.
12. Timeline no longer promises one week; priorities reordered (X2 setup C before setup B; X3 and X4 optional).

Kept against the review's advice: the short related-work addition on idea-generation systems (T10),
because the title promises that link; and X3 as a recommended block rather than removed, because 28
papers is the paper's weakest number.

Not yet verified by us (the reviewer's reading of an external paper): the Lan et al. details in T11.
That is why T11 says "check", not "fix".

---

## 8. Final checklist before submitting

- [ ] All new runs have a committed plan note with hashes, dated before the run, and all runs are reported.
- [ ] Every number in the abstract, introduction and conclusions matches a table.
- [ ] No `TODO`, no `% HUMAN STUDY` marker, no "planned human study", no "untouched papers".
- [ ] Abstract ≤ 200 words, with a concluding sentence.
- [ ] Back matter complete: ethics and consent statements, GenAI sentence with every tool and version.
- [ ] Every released text file has a recorded licence or permission; the Data Availability Statement and the repository citation point to the final tag and describe what is actually released.
- [ ] Tables rebuild from a fresh clone on a second machine.
- [ ] `python scripts/publication/make_submission_package.py` builds `manuscript.pdf`, `source.zip`, `figures.zip`; the extracted `source.zip` compiles on its own with all references resolved.
- [ ] Final PDF and exported figures checked by eye: table text ≥ 8 pt, captions match figures.
- [ ] All authors have read the final PDF and approve.
- [ ] Cover letter filled in; suggested reviewers chosen.
