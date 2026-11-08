# Evaluation metadata (our annotations only)

The SAL data (Lan et al. 2024) and the FWS corpus (Zhang et al. 2023) have no licence, so this repository
does not redistribute their text or labels. This folder holds only what we added, so the evaluation files
can be rebuilt exactly:

| File | Content |
|---|---|
| `sal_test_meta.json` | One entry per SAL-40 test sentence, in file order: `pmcid`, `sec` (section from PubMed BioC) and `preds` (our eight Stage B detectors). SAL's labels come from the fetched CSV |
| `sal_dev_meta.json` | The same for SAL dev, without `preds` |
| `sha256.json` | SHA-256 of every rebuilt file, as used in the paper's experiments |

To rebuild:

```bash
python scripts/dataset/fetch_external.py     # downloads SAL and FWS at pinned commits, checks git blob hashes
python scripts/dataset/build_eval_files.py   # writes data/splits/*.jsonl and the SAL fixture, checks SHA-256
```

Pinned sources:

- SAL: github.com/MengfeiLan/SAL_Type_Classification, commit `1848913f` (30 July 2023), `data/{train,dev,test}.csv`.
  Later commits changed these files, so the pin matters.
- FWS: github.com/xiangyi-njust/FWS, commit `79985c41` (16 December 2022), `Dataset/Corpus_For_FWS_Recognition.csv`.
