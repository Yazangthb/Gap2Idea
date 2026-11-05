# Gap2Idea: project inventory

Everything done so far on the gap extractor, as of **2026-11-04**. Built by reading the repo, all branches, sibling folders and earlier session transcripts, then recomputing what could be recomputed from saved outputs.

**Evidence grades used below**

| Grade | Meaning |
|---|---|
| **R** | Recomputed on 2026-11-04 from saved row-level or per-paper outputs; matches |
| **V** | Checked offline by Osama's `verify_minimal.py` on 2026-11-04 (not rerun for this inventory) |
| **A** | Saved aggregate only (JSON/TSV), or JSON captured from a run; no row-level check possible |
| **L** | Written in a markdown log or notebook only; no producing output found |
| **-** | Designed or planned; no result exists |

---

## 1. The state of things

**What we can defend today**
- A three-stage funnel, with row-level proof on 40 biomedical papers: **F1 0.604 → 0.658** (R).
- An estimated **3.74× fewer input tokens** than full-article prompting (R).
- A 230-paper LLM-labelled corpus across CS and biomedicine (R), plus a strong negative result: section recovery doesn't raise arXiv recall (L).
- A zero-shot NLI classifier reaching **F1 0.617** on that corpus (A).

**The three biggest holes**
1. **The full A+B+C run on the 230-paper corpus exists only for Qwen (Nov 5): F1 0.457 → 0.665, on LLM gold.** It isn't yet the same Stage C as the SAL headline (Yandex), and ModernBERT NLI has never been run through the funnel.
2. **No human labels exist.** Every cross-domain number rests on LLM labels. The chosen study design isn't built yet.
3. **The draft paper covers none of the cross-domain results.** They aren't on the `minimal` branch it was built from, and the gold data isn't on any branch.

---

## 2. Where everything lives

| Location | What | Pushed? |
|---|---|---|
| `Gap2Idea` `chore/repo-reorg` | Main working tree; all scripts and notes | Yes (`017b1fe`), minus uncommitted files from today |
| `Gap2Idea` `minimal` (`28a6c2d`) | Clean experiment branch; SAL fixtures and result JSONs are tracked here | Yes |
| `origin/paper-writing` (`f0f242f`) | Osama's 41-page draft, `EXPERIMENT_COVERAGE.md` (54 entries), `FINAL_REVIEW.md`, verifier | Yes |
| `manuscript_mdpi/` | MDPI port in progress: title, authors, abstract | Yes |
| `data/`, `artifacts/`, `runs/` | Gold, corpora, caches, downstream runs (~6.2 GB incl. 4.9 GB arXiv snapshot) | **Local only**: `*.json` and `*.jsonl` are gitignored |
| `stage_b_package/v100_results/` | Stage B logs from 6–7 Oct | Yes |
| V100 box (address in the team's private notes), `~/stageb/` | Oct 27 scripts, `results_v2/*.json` | **Box only.** Unreachable today (timeout); may need VPN |
| `docs/experiments/stage_b_v100_oct27_results.json` | The ModernBERT and fine-tuned bge JSONs, recovered today from a session transcript | Not yet committed |
| `scripts/training/stageb_v100.py`, `scripts/dataset/export_stageb_data.py`, `scripts/bench/where_missed.py` | Recovered today from an old temp folder; copied as-is, not rerun | Not yet committed |
| `Gap_Extractor/paper/` | Older course-paper plan: RQs, contributions, structure (AI/DS focus) | Own repo |
| `em.dcs/` | Course-project template and an older paper version | Own repo |
| `Gap2Idea-thesis`, `-thesis-v2`, `docs/thesis_latex/` | Thesis drafts | Mixed |
| `docs/mdpi_latex/`, `docs/paper/mdpi_draft.md` | **Earlier MDPI attempt from Aug 17**, superseded | Yes |
| Google Drive (Hamza's account) | Lit-review sheets: `Gap2Idea_papers` (163 papers), RQ2 additions, Section 2 citation evidence | Drive |

---

## 3. The pipeline and how it evolved

| Stage | Job | Versions |
|---|---|---|
| **A. Localization** | Drop sentences unlikely to hold gaps | Terminal-window rules (v1) → mid-paper anchors (v2) → whole-section union (v3) → DISCUSS+CONCL gate (SAL) → junk filter + recall-first head (funnel v2) |
| **B. Sentence classifier** | Flag candidates | Cue rules + bge-small/logreg → stacking ensemble → fine-tuned PubMedBERT/SciBERT/DistilBERT/bge → **zero-shot NLI** (DeBERTa, ModernBERT) |
| **C. LLM filter** | Drop false positives; keeps source text | `gpt-4o-mini`, then Yandex models; `yandexgpt-5-lite` chosen. Prompts: `validate`, `rct_loose`, SAL strict, SAL keep-biased, GAP-matched |

Three detector setups coexist, so name the one in use: runner bge-small, notebook MiniLM, and cached SAL fine-tuned models.

---

## 4. Datasets

| Dataset | Size | Target | Labels by | Used for | Location |
|---|---|---|---|---|---|
| **SAL** (Lan 2024) | 200 papers, 43,208 sents, 952 limitations; **test 40 papers / 7,968 / 205** | Limitation | Humans (known under-labelling) | Principal evaluation | `data/external/sal_rct` |
| **Our gold** | **230 papers (190 arXiv + 40 SAL test), 68,719 sents, 4,462 gaps**: research_gap 2,234, limitation 1,543, future_work 685; split 162/34/34 | All 3 types | Claude Opus + Sonnet + adjudicator ($31.90 + ~$4; **no humans yet**) | Cross-domain track | `data/gold/` |
| **FWS** (Zhang 2023) | 64,893 sents, 9,013 ACL papers, ~14% positive | Future work | Humans | Breadth, leakage audit | `data/fws_recognition.csv` |
| **Sci-Sentence** (Bolaños 2025) | 700 sents (test 140, 40 positive) | Limitation + research gap, 7-class | Humans | Breadth | `data/external/scisentence_repo` |
| **LimGen** (Faizullah 2024) | 4,068 ACL papers; test 1,856 sents | Limitation sections | Author sections | Training data, Stage B/C | `data/limgen`, `data/acl_limitations.tsv` |
| **BAGELS** (Al Azhar 2025) | 127 ACL papers, 1,212 gold sents | Limitations, positives only | GPT-4o-mini | Coverage check | `artifacts/bagels_cache` |
| **Kilicoglu 2018** | n=1,505 | Limitation | Humans | One notebook cross-check | Not found in `data/` |
| **unarXive subset** | N=10 and N=100 | Section text | Author headings | Early section-parser bench | `data/bench*` |
| Small golds | 19 / 33 / 49 gaps over ~10 papers | Mixed | gpt-4o, gpt-4o-mini, Sonnet | Early development only | `data/bench_gap` |
| Corpus sets | `dataset_v1..v9plus`, `excellent_200`, 5 GB arXiv metadata, 686 MB PDFs, `data/paper_texts.jsonl` | - | - | Source papers | `data/` |

---

## 5. Experiments

### Track 1. Early development (Jan–Jul 2026)

| Experiment | Result | Grade | Status |
|---|---|---|---|
| Per-paper LLM baseline | macro-F1 0.187 on old silver bench | L | Historical |
| Stage A slicer on 19-gap gold | 82% dropped; recall 0.84 at τ=0.80 | L | Too small to publish |
| Poisoned-negatives bug in self-distillation | Recall 0.37 → 0.47 after fix | L | Good anecdote |
| Classifier ablation (logreg/DistilBERT/SetFit) | All tie at 0.37–0.42 | L | Superseded |
| ACL limitation harvest + cap sweep | Limitation recall 0.11 → 0.44; saturates at 1,500 | L | Excluded from draft |
| Encoder sweep (frozen) | All tie at 0.526 | L | Superseded |
| LimGen head-to-head | DistilBERT-FT 0.674 > stack 0.627 > bge 0.610 > NB 0.553 | L | Excluded from draft |
| N=10 / N=100 section parser bench | v2b recovery 0.70, hallucination 0.15; 161 gaps / 90 papers at N=100 | A | Excluded from draft |

### Track 2. Benchmarks against published work (Aug 2026)

| Experiment | Result | Grade | Status |
|---|---|---|---|
| SAL section pool at 20.7% prevalence | Detector 0.762; + Stage C **0.810–0.815** vs Lan's 0.821 | L | Excluded: predictions lost |
| SAL full stream, undersampling variants | 0.561 → 0.637 with Stage C | L | Superseded by Track 3 |
| Stage C model × prompt grid | gpt-oss-120b, yandex-5-pro, qwen3, deepseek all 0.80–0.815 | L | Excluded |
| Frozen encoders on SAL | SPECTER2 0.346 > bge 0.263 > S-PubMedBert 0.250 | L | Excluded |
| **FWS leakage audit** | Zhang's 0.9073 reproduced at 0.9072; chi² inside folds gives **0.8252** | L | Excluded as unproven. **Rerunnable: script and data are local** |
| FWS clean split | bge-small FT 0.879 ≈ PubMedBERT 0.877 > SciBERT 0.864 | L | Excluded |
| FWS held-out + Stage C | Positive F1 ~0.74; Stage C neutral on strong detectors | L | Excluded |
| LimGen + Stage C | 0.645 → 0.632 (P 0.57 → 0.845) | L | Excluded |
| BAGELS coverage | 0.738 → 0.434 after Stage C | L | Excluded |

### Track 3. SAL, saved outputs (Oct 2026): the draft's core

| Experiment | Result | Grade | In draft |
|---|---|---|---|
| Stage A section gate | Keeps 1,885 of 7,968 (76.3% dropped), 203/205 positives | **R** | Main |
| Eight classifiers after Stage A | PubMedBERT-FT **0.604** (P .460, R .878), SciBERT-FT 0.553, DistilBERT-FT 0.523, e5 0.457, bge-base/mpnet 0.437, bge-small 0.436, MiniLM 0.371 | **R** | Main |
| **Stage C, per-detector, `rct_loose`** | 269 kept, 156 TP: P .580, R .761, **F1 0.658** | **R** | Main (headline) |
| **Same run, Qwen3.8-27B (Nov 5)** | 327 kept, 179 TP: P .547, R .873, **F1 0.673** (CI .614–.729); paired difference vs Yandex +.015 (CI −.018 to +.047). See `stage_c_sal40_qwen.md` | **R** | Candidate headline |
| Stage C, SAL keep-biased | 249 kept, 148 TP: P .594, R .722, F1 0.652 | **R** | Main |
| Stage C, strict SAL criteria | F1 0.568 | A | Appendix |
| Union-judged runs | 0.701 (old prompt), 0.554 (keep-biased): batch effects | A/L | Appendix |
| Prevalence sensitivity at 20.7% | .749 detector / .757 rct / .741 SAL | V | Main |
| Neighbour credit W=0/1/2 | F1 .658 / .821 / .853 | V | Main + appendix |
| Batch composition effect | Same sentences: 56/60 kept vs 37/60 kept | L | Appendix |
| Error analysis of false positives | 22 of 101 read by one person: ~45% under-labelled, 36% convention, 18% LLM error | L | Discussion |
| Kilicoglu cross-check | Detector .749; + Stage C .639 | L | Appendix |

### Track 4. Breadth and baselines (Oct 2026)

| Experiment | Result | Grade | In draft |
|---|---|---|---|
| FWS funnel | .667 → **.707** (keeps 134/136 TP, drops 27/52 FP) | A | Main |
| Sci-Sentence, mismatched prompt | .612 → .600 | A | Main |
| Sci-Sentence, GAP-matched prompt | P 1.00, R .575, F1 .730 | L | Main (prompt conflict noted) |
| Sci-Sentence, in-domain + recall-first | **F1 .880** (P .943, R .825) | L | Main (no test held back) |
| Baseline grid | SAL no Stage A: PubMedBERT .495, TF-IDF .443. FWS: TF-IDF .714. Sci-Sentence: bge .814 | A | Main |
| Stage B dev sweeps (Oct 6–7) | Within-corpus SciBERT-FT macro .876 / .900 / .856 | L | Appendix |
| Dataset reuse audit | ~70 citers across 6 datasets; almost none reuse them for evaluation | L | Related work |

### Track 5. Cost (Oct 2026)

| Experiment | Result | Grade | In draft |
|---|---|---|---|
| Token accounting, 40 papers | 282,846 vs 75,622 = **3.7403×** | **R** | Main |
| Full-paper LLM, 10 papers | P .559, R .345, F1 ≈.427; 2/10 calls returned nothing | A | Main |
| Per-stage funnel, same 10 papers | A 498/55, A+B 109/51, A+B+C 77/45: F1 .682 | A | Main |
| Caching, batching, long-paper scenarios | 4.4×, 5.1×, up to 13.2× | L | Excluded: hypothetical |

### Track 6. Cross-domain gold (Oct 17 – Nov 3, 2026): **not in the draft's results**

| Experiment | Result | Grade |
|---|---|---|
| Gold build | 4,462 gaps; 2,960 high confidence + 1,502 adjudicated | **R** |
| Inter-model agreement | Any-gap κ **0.778** (limitation .748, future work .869, research gap .725) | **R** (any-gap); L (typed) |
| SAL overlap of the gold | Limitation recall .927 (190/205); binary .956 (196/205) | **R** |
| Third-model check on extras | 116/227 accepted | L |
| Stage A trained localizer (GBM) | Keep 40% → recall 78%; keep 55% → 87%; rules 47% at keep 16% | L (partial meta) |
| arXiv sections via PDF headings | 40.3% → 37.5% | L |
| **arXiv sections via LaTeX** (175/190) | 55.6% → 55.2%; oracle cap 60.5%; 82% of misses in Methods/Experiments/Results; biomed 84.1% | L |
| Rules + hybrid on gold | Limitation+FW F1 27.1 / 33.7 | L |
| Stage B frozen sweep | bge-small .453, PubMedBERT .442, SPECTER .438, old head .275 | A |
| Stage B fine-tuned bge-small | P .508, R .687, **F1 .584**, type accuracy .80 | A |
| DeBERTa zero-shot | P .419, R .625, **F1 .502** (box run: .501) | L |
| **ModernBERT-large NLI zero-shot** | P .628, R .605, **F1 .617**; biomed .661, arXiv .607 | A |
| Fine-tuned BiomedBERT, SciBERT | **Failed** (torch refused `.bin`); no rerun found | - |
| Fine-tuned ModernBERT | Proposed next step; not run | - |
| Funnel v2 recall-first | 641/664 gaps, **R 96.5%, P 8.8%**, keeps 62.1% | **R** (counts) |
| Funnel v2 balanced | R 72.3%, P 33.4%, keeps 12.2% | **R** (counts) |
| Funnel v2 Stage C (`yandexgpt-5-lite`) | **First 120 of the 1,438 candidates at the balanced threshold, from 3 of 34 papers, 46 gaps; not class-balanced**: P 38.3% → 75.8% (TP 25, FP 8), keeps 54.3% of gaps; batch 40, **no context sent** (`use_context` left at its default of False); verdicts not saved | L |
| Funnel v2 Stage C (Claude Sonnet subagent, keep-biased rubric, with context) | **80 randomly shuffled balanced candidates, 19 gaps**: P 23.8% → 60.0% (TP 18, FP 12), **keeps 94.7% of gaps**. Sonnet also helped make the gold, so this is partly circular; mixed samples are known to inflate recall; verdicts rebuilt from the transcript | L |
| Stage C model vs prompt (`yandexgpt-5-lite` on the same 80) | `validate`, no context: 8/19 gaps, 6 FP. Sonnet's rubric with context, batch 20: **19/19 gaps, 28 FP, P 40.4%**. Sonnet: 18/19, 12 FP. See `stage_c_prompt_vs_model.md` | **R** (verdicts saved) |
| Stage C on self-hosted Qwen3.8-27B (same 80) | Sonnet's rubric, batch 20, no thinking: **19/19 gaps, 16 FP, P 54.3%, F1 0.704**; thinking on gave 18/19, 18 FP at about 15× the time. Not used to make the gold | **R** (verdicts saved) |
| **Full A+B+C, Qwen3.8-27B, all 1,438 candidates, 34 papers** | **P 33.4% → 64.3%, R 72.3% → 68.8%, F1 0.457 → 0.665 (CI 0.608–0.715)**; kept 95.2% of gaps. **Excluding the 196 candidates used to choose the setup: P 63.2%, R 68.8%, F1 0.659 (CI 0.599–0.712).** arXiv (28 papers) 0.641; biomedical (6 papers, a subset of SAL-40) 0.772. See `stage_c_full_qwen.md` | **R** (verdicts saved) |

**NLI variants on validation only (Nov 5):** short per-type hypotheses reach about 79% recall at 39.2% precision, or 87% recall at 27.7%, against 28.2% and 21.4% for the bge-small head at matched recall. Context and negative hypotheses made it worse. See `nli_variants.md`. 

**NLI + Qwen on the test split (Nov 5):** frozen short-hypothesis NLI at τ 0.354 + Qwen gives **P 55.8%, R 86.4%, F1 0.678** (limitation 80.5%, future work 86.6%, research gap 90.2%). F1 is tied with bge-small + Qwen (+0.013, CI −0.034 to +0.069) with 17.6 points more recall, and beats ModernBERT alone (+0.064, CI +0.011 to +0.128). See `stage_c_nli_qwen.md`. On validation, rubric strictness moves P/R from 66.8/82.0% (keep-biased) to 75.7/74.0% (neutral) to 90.4/52.5% (strict); see `stage_c_rubric_val.md`. On test, as fixed settings: keep-biased P 56.3%, R 85.7%, F1 0.679; **neutral P 68.2%, R 79.7%, F1 0.735** (+0.055 vs keep-biased, CI +0.025 to +0.080); strict P 86.6%, R 57.4%, F1 0.690. Rerunning keep-biased changed 1.3% of verdicts. See `stage_c_rubric_test.md`. On SAL-40 human labels, the rubric narrowed to limitations gives high precision P 74.5%, R 78.5%, **F1 0.765**; balanced F1 0.751; high recall F1 0.736. See `STAGE_C_MODES_RESULTS.md`. Chosen funnel (NLI + Qwen, high precision): SAL-40 P 66.2%, R 84.9%, F1 0.744; SAL dev P 64.3%, R 80.0%, F1 0.713; FWS lowercased P 76.5%, R 75.5%, F1 0.760. **FWS casing reveals its label**, so earlier LLM results on raw FWS text are unreliable. Estimated input-token saving vs one whole-paper call: 3.08× SAL-40, 2.04× FWS, 1.25× gold.

**Recall by gap type on the test set (664 gaps):**

| Classifier | Research gap (347) | Limitation (220) | Future work (97) |
|---|---|---|---|
| ModernBERT zero-shot | 74.1% | **46.4%** | **44.3%** |
| bge-small fine-tuned | 77.2% | 48.6% | 83.5% |
| bge-small frozen | 78.7% | 68.2% | 66.0% |

### Track 7. Human validation: nothing labelled yet

| Item | State |
|---|---|
| 210-row worksheet (`.csv`) | Superseded. **Contains labels and is public** (see §8) |
| 362-row blind `annotate.html` + `.xlsx` | Built; 40 biomed papers; no answers returned |
| `gold_human_merge.py` | Built; no input yet |
| **Chosen design** (revised Nov 5): 2 raters + 1 tie-breaker, 136 items across arXiv and SAL-40, on the final funnel's high precision output | Built and sample frozen (`human_study_freeze.md`). **No ratings yet.** Needs raters and about 2.5 person-hours |
| "64-paper human evaluation", "800-paper venue corpus" | Listed as completed in `TEAM_EVIDENCE_REQUEST.md`, but **I found no data for them on disk**. Ask the team |

### Track 8. Downstream, out of scope for this paper

| Work | State |
|---|---|
| Clustering bench (Hamza) | N=100, 161 gaps; silhouette around 0.05 in the rows checked. Branches `experments/clustering_*` |
| Idea generation | Runs on ai/ml/math (58/51/9 gaps). 11 ideas LLM-judged, composite 2.61. **No human evaluation**; the S2 novelty mean of 1.00 looks unreliable (top ideas show `nan`). Saved for the Gap2Idea paper |
| Theme mining, gap graph, semantic search, Streamlit/Docker deploy | Implemented; not evaluated here |
| Thesis RQ1 (graph forecasting) | Separate thrust |

---

## 6. Numbers that conflict, and which to use

| Topic | Values in circulation | Use |
|---|---|---|
| SAL funnel F1, full stream | .658, .652, .701, .637, .568, .554 | **.658** (R). The rest are other runs or prompts |
| SAL vs Lan's condition | .757 (replay) vs .810–.815 (Aug log) vs Lan's .821 | **.757**, labelled as a sensitivity. The .81 has no saved predictions |
| Neighbour credit .821 | Happens to equal Lan's .821 | Never present them together as a match |
| "True precision ≈ 0.87" | From `sal_stage_c_summary.md` | **Don't use.** The stated 45% only gives ≈.777. Human adjudication is needed |
| Full-paper baseline F1 | .429 vs .427 | **.427** |
| Cost reduction | 3.74× vs 5.1×, 13.2×, 128–166× | **3.74×, estimated.** The others are projections |
| Biomed Stage A recall | 99% (203/205) vs 84.1% (427/508) | Both are valid: different labels and slicers. State which |
| arXiv Stage A recall | 55.6% vs 40.3% vs 96.5% | Different targets and settings. **The 40.3 and 55.6 setups need reconciling** before citing both |
| FWS "F1" | .907/.825 macro, .879 macro, .74 positive, .707 positive | Never mix macro and positive-class |
| κ .778 | Docs say it "matches the human ceiling" | Report as **inter-model only** |
| What drives Stage B | Aug: "data, not model". Oct: "architecture beats labels" | Different datasets and regimes. **Needs one honest story** |
| Is `research_gap` in scope? | Benchmark doc: out of scope. Funnel v2 and NLI: counted, and it's half the gold | **Must decide.** It changes every cross-domain number |

---

## 7. What's missing before submission

**Experiments to run**
1. **See the must-fix list in `PAPER_SUMMARY.md`**, starting with a frozen-prompt check on SAL's 40 dev papers. The SAL rerun on Qwen and the balanced 230-paper run are done; the 7,297 recall-first candidates are not.
2. **ModernBERT inside the funnel**, plus fine-tuned ModernBERT and the two failed fine-tunes.
3. **Paired significance tests** between classifiers (needs row-level predictions).
4. **Rerun the FWS leakage audit** with saved output, since it's a strong claim.
5. **Scope-consistent results**: limitation + future work alone, and all three types.

**Human study**
6. Build the sampler and analysis scripts, then run the 2+1 study.

**Evidence to recover or publish**
7. Copy `~/stageb/results_v2/` from the V100 box, including any predictions.
8. Publish the gold (labels and splits) to Zenodo after the human study. `*.jsonl` ignores keep it out of git today.
9. Locate or regenerate fine-tuned checkpoints; the headline SAL predictions can't be regenerated without them.

**Known method weaknesses to disclose or fix**
10. Stage C prompts and thresholds were chosen using test feedback; there's no untouched evaluation.
11. The shared prompt suffix rejects prior-work statements, which conflicts with the GAP-matched prompt.
12. The full-paper baseline is weak: n=10, one cheap model, 2 failed calls.
13. Provider `latest` aliases aren't pinned; model identities need checking for the Methods section.

---

## 8. Open decisions

| Decision | Options |
|---|---|
| **Public label CSV** (`data/gold/human_validation_worksheet.csv`, commit `cedd710`) | Remove from history with a force-push, or leave. 10 of 361 study items are affected |
| **`research_gap` in or out of the main claim** | It's 50% of the gold and scores far higher than the other two types, so it flatters "any gap" numbers |
| **Abstract wording** | "Outperforming fine-tuned and frozen encoders" overstates: only bge-small was fine-tuned on this corpus. The 96.5% recall comes with 8.8% precision, which the abstract omits |
| **Which experiments block submission** | §7 items 1, 3, 5, 6 look essential; the rest are strengthening |
| **Paper shape** | Principal result stays SAL with a cross-domain section, or cross-domain becomes the main result once Stage C runs on it |
| **Authors and metadata** | Rustam's details, ORCIDs, address, Hamza's surname capitalisation |
