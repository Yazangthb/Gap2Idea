# SAL and FWS evaluation splits

The exact sentences every SAL and FWS number in the paper is scored on. Use these files instead of
rebuilding the selection. They are written by `scripts/dataset/export_sal_fws_splits.py`.

One sentence per line, in reading order: `{"paper_id", "idx", "sentence", "label"}`; SAL rows also have
`section` (from the PubMed BioC sections; Stage A keeps `DISCUSS` and `CONCL`). `label` is 1 for a
limitation (SAL) or a future-work sentence (FWS).

| File | Papers | Sentences | Positives | Role |
|---|---:|---:|---:|---|
| `sal_test.jsonl` | 40 | 7968 | 205 | SAL-40 test. Same rows as `stage_c_package/sal_criteria/sal_test_allmodels.json` |
| `sal_dev.jsonl` | 40 | 7729 | 180 | SAL dev: thresholds and checks only |
| `fws_test.jsonl` | 209 | 1506 | 220 | FWS test |
| `fws_dev.jsonl` | 418 | 3005 | 418 | FWS dev: thresholds only |

## How they were built

- **SAL** comes from the original split of Lan et al. (`data/external/sal_rct/{train,dev,test}.csv`,
  from github.com/MengfeiLan/SAL_Type_Classification). We drop fragments under three words: 914 rows of
  `test.csv`, none of them a limitation. To train on SAL, use `train.csv`.
- **FWS** comes from `data/external/fws_corpus.csv` (github.com/xiangyi-njust/FWS). We first removed
  sentences that appear in `data/scibert_prep/train.jsonl` (an earlier training set of ours), then shuffled
  the papers with `random.seed(0)`. Test is the first papers up to 1500 sentences; dev is the next papers up
  to 3000 sentences. To train on FWS, use papers that are in neither `fws_test` nor `fws_dev`.
- **FWS text is lowercased.** In the released corpus every positive sentence is all lowercase and almost no
  negative is, so a case-sensitive model can read the label from the casing. Lowercase any FWS text you use,
  including training data.
