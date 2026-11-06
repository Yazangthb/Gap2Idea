# Task 5b: FWS with the current gold training corpus

Prepared 2026-11-06 on `Task_5b`. The teammate's upload commit `85523b4` was
imported as `a030d0d`, containing exactly:

- `data/gold/gap_gold_final.jsonl`: 4462 positive gap sentences, including
  3751 arXiv CS/ML and 711 biomedical SAL annotations.
- `data/gold/splits.json`: 162 train, 34 validation, 34 test papers.
- `data/paper_texts.jsonl`: the original full texts of 190 arXiv papers.

The tracked SAL fixture supplies the other 40 papers. These files are the current
mixed-domain gold corpus, **not the FWS binary-labelled corpus**. FWS still uses
`data/external/fws_corpus.csv`, the public future-work recognition data downloaded
by `fetch-public`. The obsolete `data/scibert_prep/train.jsonl` is unnecessary
under this protocol.

## Explicit protocol revision: `fws_current_gold_v1`

1. Reconstruct every sentence of all current gold papers with the same splitter
   as `export_stageb_data.py`. Check every positive paper/index/text identity;
   unlisted sentences are `none`. For SAL, preserve canonical cached indices,
   but use the current four-class gold labels for this reconstruction.
2. Use **all sentences from the 162 TRAIN papers only** as an exclusion source,
   including unlisted negatives. This is 45777 sentences / 45123 unique stripped,
   lowercase strings. Validation/test gold papers do not contribute exclusions.
   arXiv's share of training is 134 papers / 40276 sentences, not 162 papers.
3. Lowercase **every** FWS sentence before exclusion, training, validation and
   inference, including negatives. The source CSV's 9009 positive sentences are
   already lowercase, whereas only 14 of 55887 negatives are. This source-format
   artifact must not become an NLI/encoder label cue. This matches the casing
   policy in `origin/chore/repo-reorg:scripts/bench/nli_sal_fws.py`.
4. Remove exact stripped/lowercased current-training text overlaps from the FWS
   corpus. Two original CSV rows are removed (indices 32755 and 42550).
5. In stable eligible CSV paper order, shuffle with `random.Random(0)`. Select
   complete eligible paper groups until at least 1500 test sentences. Reserve
   10% of the remaining sorted/seed-shuffled paper IDs for validation, and train
   FWS classifiers on all remaining FWS papers.
6. Use the existing registered model settings: train-only fitting, validation-only
   thresholds (macro F1 for FWS), one test pass, 2000 paper-cluster bootstrap
   replicates. Report both positive-class metrics and macro F1. The primary
   launcher includes both original broad and typed NLI hypotheses as separately
   registered variants; do not choose only the best test result for the paper.

| Split | Papers | Sentences | Positive future-work sentences |
|---|---:|---:|---:|
| Train | 7924 | 57198 | 7985 |
| Validation | 880 | 6190 | 804 |
| Test | 209 | 1506 | 220 |

Paper IDs are disjoint across splits. Two identical sentence strings recur
between train and test in different papers; this is recorded, not silently
removed. Test membership is locked with raw source text hashes and canonical
indices, and lowercase model text has its own hash.

**This is a new exclusion/casing protocol.** Its matching 209/1506 count does not
prove identity with the old breadth-funnel selection. Do not reuse old aggregate
funnel F1 numbers for paired claims. Import complete predictions from a funnel
run using the same papers and lowercase inputs, or verify every sentence identity
before pairing. The legacy reconstruction remains available explicitly with
`--fws-protocol legacy`, and still requires the old file.

## Run now

The full FWS split is already prepared locally at
`data/task5b/prepared-current-gold-v1/fws/`. The completed SAL run is untouched.

```powershell
Set-Location 'E:\vs\Gap2Idea - main code\Gap2Idea'
python scripts/bench/run_task5b_fws.py
```

This runs TF-IDF/LR, binary NB, original ModernBERT NLI, typed ModernBERT NLI,
and the train-only chi2/NB Zhang adaptation. It uses batch size 4 and 2000
bootstrap resamples. No LLM/API credentials are needed. Output will be:

```text
runs/task5b/fws-current-gold-full-01/fws/report.md
```

Equivalent explicit command:

```powershell
python scripts/bench/task5b.py run --datasets fws --data-dir data/task5b/prepared-current-gold-v1 --methods tfidf_lr,binary_nb,modernbert_nli,modernbert_nli_typed,zhang_nb --batch-size 4 --bootstrap 2000 --run-dir runs/task5b/fws-current-gold-full-01
```

The run directory must be new. For another deliberately registered configuration,
use a different `--run-dir`; never overwrite the previous results. Rules intended
for limitations are not run against FWS future-work labels. Optional SciBERT
fine-tuning can be run later in its own directory on the same frozen data.

## Fresh checkout / preparation only

```powershell
python scripts/bench/task5b.py fetch-public --datasets fws
python scripts/bench/task5b.py prepare --datasets fws --fws-protocol current-gold --output-dir data/task5b/prepared-current-gold-v1
# Or use the launcher to prepare if absent, without fitting/scoring:
python scripts/bench/run_task5b_fws.py --prepare-only
```

Preparation refuses to overwrite the existing lock. It records exact input and
prepared-file hashes, all split membership, source row IDs, source/canonical text
hashes, the exclusion-source statistics, and `protocol_id=fws_current_gold_v1`.
It does not substitute positive-only gold sentences for complete training text.

## Validation

See the updated validation entry in [task5b_validation.md](task5b_validation.md).
The implementation smoke uses only FWS train/validation papers. Full official
FWS test scoring is left for the command above; there are no new full FWS results
to add to the paper yet. Prepared data and smoke output remain under ignored
directories. The imported source gold files are tracked by the upload commit.
