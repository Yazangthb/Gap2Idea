# Stage B package

Self-contained notebooks for **Stage B** — the per-sentence gap classifier that runs on Stage A's output.

## Files

| File | What it is |
|---|---|
| `stage_b_all_experiments.ipynb` | Trains the head **from scratch** and shows the improvement arc: rules-only → +head (no ACL) → +head (ACL 600) → +head (ACL 1500), evaluated end-to-end on the 10 gold papers, with a table + charts. |
| `stage_b_final.ipynb` | The **final Stage B** only — cue rules + bge-small+logreg head trained with 1500 ACL limitation sentences. A clean `run_stage_b(sentences)` step, a demo on a gold paper, and final gold numbers. |

## What Stage B is

- **B.1 cue rules** (`cue_label`): high-precision regex — "future work…", "we did not…", "limitation…" are accepted for free, with a type.
- **B.2 embedding head** (`EmbeddingGapHead`): frozen **bge-small-en-v1.5** encoder + a **logistic regression**, 3 classes (none / limitation / future_work), catches cue-less gaps.
- A sentence is kept as a gap if **rules fire OR the head predicts a gap**.

## How to run

Open either notebook on the `gap2idea` (`.venv`) kernel and **Run All**. No project imports — Stage A and Stage B are implemented inline; only `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `sentence-transformers` are used (all already in the env).

- **Not pre-executed on purpose.** The notebooks train the head when you run them.
- **The one slow cell** embeds ~25,000 training sentences with bge-small. It uses your **GPU automatically if a CUDA PyTorch is installed** (seconds); otherwise CPU (a few minutes). Either way it **caches to `train_emb.npy`** next to the notebook, so later runs are instant.
  - Your current `.venv` has a **CPU-only** PyTorch (`torch.cuda.is_available()` is False). To use the GPU, install a CUDA build of torch into the env; the code will pick it up with no changes.

## Data used (all cached in the repo)

- `data/scibert_prep/train.jsonl` — 25,427 labeled training sentences (self-distilled from the LLM teacher in `runs/*`, merged with ACL limitations).
- `data/acl_limitations.tsv` — harvested ACL limitation sentences, used to vary the training cap (0 / 600 / 1500).
- `data/scibert_prep/gold_papers.jsonl` + `data/bench_gap/gold_sentences_v2.tsv` — 10 papers / 49 gold gaps for evaluation.

## Notes & caveats

- The pre-trained `data/gap_head*.joblib` files are **not used** — they were pickled under an older scikit-learn and fail to predict under the current one. These notebooks retrain the head instead.
- The reconstructed training split matches the shipped head's metadata exactly: 23,748 none, 116 future_work, 63 self-distilled limitation, + up to 1500 ACL limitation.
- **Precision is not directly measurable** on the gold papers (only positive gaps are annotated), so `kept_sents` is a load / precision proxy.
- 10-paper gold set → small-sample numbers, good for the trend, not a large-scale claim.
