# Sentence-level datasets for Stage B

Candidate datasets for training/augmenting the Stage B classifier (per-sentence:
`none` / `limitation` / `future_work`). All validated and downloaded to
`data/external/` (gitignored, local only, ~458 MB total). Samples below are real rows,
trimmed for readability.

**Fit summary**

| Has a gap/limitation/future-work label | Negatives only (role datasets) |
|---|---|
| Sci-Sentence (all three), SAL/RCT (limitation), FWS (future_work) | CSAbstruct, csabstracts, PubMed-RCT |

---

## 1. Sci-Sentence ⭐  `data/external/scisentence_repo/datasets/Based_Version.xlsx`
700 sentences, one `Category` each (7 balanced classes). CC-BY. The only set with
**RESEARCH GAP + LIMITATION + EXTENSION** per-sentence labels *and* negatives.
Columns: `Sentence`, `Category` (+ Venue, Year, Related Work Section, …). An
`Augmented_Version.xlsx` adds 561×4 LLM paraphrases.

```
[RESEARCH GAP] Although a few case studies ... have been reported, ...
[LIMITATION ] ...some models might still be poorly segmented and thus fail to be accurately landmarked
[EXTENSION  ] further, we use the verifier during proof generation rather than merely to rank ...
[RESULT     ] A large multicenter study demonstrated superior success rates ... (81% vs 56%) ...
[DESCRIPTION] archjava is an architecture description language on top of java ...
[OVERALL    ] a related task to playlist recommendation is automatic playlist continuation ...
[OTHER      ] Additionally we give many thanks to Jiang jie and Liew Hongze ...
```
Counts: OVERALL 102, LIMITATION 101, EXTENSION 100, OTHER 100, RESULT 100, RESEARCH GAP 99, DESCRIPTION 98.
→ map: RESEARCH GAP + EXTENSION → future_work-ish, LIMITATION → limitation, rest → none.

## 2. SAL / RCT  `data/external/sal_rct/{train,dev,test}.csv`
Per-sentence limitation detection **with true negatives** (biomedical RCT full text).
train/dev/test = 25,676 / 8,650 / 8,882 sentences. Key cols: `sentences`,
`limitation_annotation` (`[True]`/`[False]`), `coarse_grained_categories` (13 types).

```
[LIMIT Generalization,StudyDuration] ...results ... may not be generalizable to other practices...
[LIMIT Population]                    In this study we did not completely evaluate the effect of the steroid...
[LIMIT Population,UnderpoweredStudy]  Because our study had a small sample size and showed heterogeneity...
[none ] A randomized trial of mail vs. telephone invitation ...   (title)
[none ] Background
```
→ map: `[True]` → limitation (+ sub-type), `[False]` → none.

## 3. FWS / Zhang  `data/external/fws_corpus.csv`
Binary future-work recognition, 64,897 sentences. Cols: `id, year, text, label (0/1), chapter`.

```
[label=1 | Conclusions] ...we hope that other learning algorithms can benefit ... can be generalized ...
[label=1 | Conclusions] we plan to use argumentative zoning as a first step for ir ...
[label=0 | Conclusions] The RRE-based rule sequence learner presented above is able to learn rules ...
[label=0 | Conclusions] These regular-expression based conditions lead to higher accuracy ...
```
→ map: `1` → future_work, `0` → none. Negatives are also conclusion-section sentences → a hard, realistic split.

## 4. CSAbstruct  `data/external/csabstruct_repo/data/CSAbstruct/*.jsonl`
Per-sentence abstract roles (one JSON per abstract: `sentences[]`, `labels[]`, `confs[]`).
train/dev/test = 1,668 / 295 / 226 abstracts.

```
[background] Gamification has the potential to improve the quality of learning ...
[objective ] Our objective in this study is to evaluate a gamified learning activity ...
[method    ] The activity made use of a gamified multiple choice quiz ...
[result    ] Results show that there was some degree of engagement and enjoyment.
```
→ No gap class → use as clean **negatives**.

## 5. csabstracts (sergiog95)  `data/external/csabstracts_repo/train.txt`
Tab-separated `idx ⟶ role ⟶ sentence`, 5 roles, 654 CS abstracts (train = 3,287 sentences).

```
[OBJECTIVE  ] We study the problem of optimally investing in nodes of a social network ...
[METHODS    ] In particular, we consider the possibility of campaigning in multiple phases ...
[CONCLUSIONS] Our simulation study affirms that nodes attributing higher weightage to initial biases ...
```
→ Negatives only (small).

## 6. PubMed-RCT 20k  `data/external/pubmedrct_repo/PubMed_20k_RCT/train.txt`
`role<TAB>sentence`; `###id` starts an abstract, blank line ends it. 5 roles. (Also a 200k set as `train.7z`.)

```
### abstract_id 24293578
[OBJECTIVE  ] To investigate the efficacy of 6 weeks of daily low-dose oral prednisolone ...
[METHODS    ] A total of 125 patients with primary knee OA were randomized 1:1 ...
[RESULTS    ] There was a clinically relevant reduction in the intervention group ...
[CONCLUSIONS] Low-dose oral prednisolone had both a short-term and a longer sustained effect ...
```
→ No gap class → large **negatives** source.

---

## Availability & caveats (verified 2026-10-06)

| Dataset | Status |
|---|---|
| Sci-Sentence | ✅ live — `github.com/fcobolanos/Classifying-the-Components-of-a-Literature-Review` (CC-BY) |
| SAL / RCT | ✅ live, but on the **`checkpoint`** branch, not `main`. Repo's `…/main/data/` path is **404** — our `build_rct_pool.py` / `bench_rct.py` loader URL is stale and needs the branch fixed. |
| FWS / Zhang | ✅ live — `github.com/xiangyi-njust/FWS` (also already in repo as `data/fws_recognition.csv`) |
| CSAbstruct | ✅ live — `github.com/allenai/sequential_sentence_classification` |
| csabstracts | ✅ live — `github.com/sergiog95/csabstracts` |
| PubMed-RCT | ✅ live — `github.com/Franck-Dernoncourt/pubmed-rct` (large; 200k set left compressed) |
| ALTA 2019 `emerald_ascm` | ❌ **dead (404)** — had an explicit "Research limitations" class; needs author contact |
| LREC 2012 | ❌ unreachable — had an explicit "Gap" class; needs author contact |

All downloads live under `data/external/` and are gitignored (local only).
