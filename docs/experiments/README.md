# Gap2Idea — Experiments index & reproduce guide

Everything behind the cheap gap-extraction funnel, organized for writing it up.

- **[experiment_log.md](experiment_log.md)** — lab notebook (every experiment: why → how → result → conclusion).
- **[results_registry.md](results_registry.md)** — all numbers/tables in one place.
- **[../related_work_analysis.md](../related_work_analysis.md)** — prior-art survey + positioning.
- **[../gap_extraction_architecture.md](../gap_extraction_architecture.md)** — living architecture & status (§8 = the funnel).
- **[../stage_a_explained.md](../stage_a_explained.md)** — Stage A deep-dive with figures.

## Code map (scripts/ — left in place; they cross-import via fixed paths)
| Purpose | Script | Notes |
|---|---|---|
| **Core module** | `src/gap2idea/pipeline/gap_funnel.py` | Stage A slice + Stage B (cue rules + `EmbeddingGapHead`) + corpus runner |
| **CLI** | `gap2idea extract-gaps-funnel --mode hybrid --head data/gap_head.joblib` | drop-in `gaps.tsv` |
| Gold (eval) | `scripts/dataset/build_gap_gold.py` | full paper → gpt-4o extract→verify → 19-gap gold |
| ACL data (train) | `scripts/dataset/harvest_acl_limitations.py` | LimGen Limitations + cue-harvest future-work |
| Train head | `scripts/training/train_gap_head.py` | self-distill + `--acl-cap` + leakage guard + meta |
| Benchmark | `scripts/bench/bench_gap_recall.py` | staged: Stage A recall / Stage B / end-to-end |
| Benchmark (cost) | `scripts/bench/bench_gap_funnel.py` | older bench on `bench_gold` + cost projection |
| LLM baseline | `scripts/archive/eval_gap_extraction.py` | macro-F1 0.187 on old silver bench |
| Ablation: classifier | `scripts/training/test_bert_stageb.py` | logreg vs DistilBERT |
| Ablation: few-shot | `scripts/training/test_setfit_stageb.py` | SetFit contrastive (raw-torch reimpl) |
| Ablation: data cap | `scripts/training/sweep_acl_cap.py` | ACL limitation/future cap sweep |
| Ablation: encoder | `scripts/training/sweep_encoders.py` | frozen encoder sweep |
| Figures | `scripts/bench/plot_stage_a.py` | localization + funnel PNGs |

## Data artifacts
| Path | What |
|---|---|
| `data/bench_gap/gold_sentences.tsv` | **clean eval gold** (19 gaps / 9 papers) |
| `data/bench_gap/gold_sentences_unverified.tsv` | raw pre-verify gold (31, ~25% contaminated) |
| `data/bench_gap/stage_a_audit.tsv` | per-gap Stage-A containment |
| `data/bench_gap/train/gold_sentences.tsv` | partial LLM training gold (41 sents; run died at 10/72 papers — credits) |
| `data/acl_limitations.tsv` | 6,433 ACL mandated-Limitations sentences (LimGen, CC-BY-4.0) |
| `data/acl_futurework.tsv` | 1,268 cue-harvested future-work sentences (net-negative; unused) |
| `data/gap_head*.joblib` | trained heads — see [results_registry.md](results_registry.md) |
| `data/bench/funnel_*.tsv` | older bench outputs (on `bench_gold`) |
| `docs/figures/stage_a_*.{png,svg,mmd}` | Stage A figures |

## Where to read the actual extracted gaps (outputs)
| File | What you'll see |
|---|---|
| **[funnel_demo_output.md](funnel_demo_output.md)** | **Readable** per-paper report on the 10 gold papers: reduction (full→slice→gaps), every emitted gap flagged ✅gold / —extra, and Stage-B-rejected samples |
| `data/demo/funnel_gaps.tsv` | The same 62 gaps, machine-readable (paper_id, gap_type, source, section_type, gold_match, gap_sentence) |
| `data/demo/funnel_corpus_ml.tsv` | Corpus-scale output: 150 gaps from 40 ml papers (the real `gaps.tsv` the pipeline emits) |
| `data/bench/funnel_predictions.tsv` | Earlier rules-vs-hybrid prediction dump on gold papers |

Regenerate the readable report + tsv any time:
```bash
python scripts/bench/demo_funnel.py --head data/gap_head.joblib
# or over any corpus -> gaps.tsv:
python -m gap2idea.cli extract-gaps-funnel --mode hybrid --head data/gap_head.joblib
```

## Reproduce (end-to-end)
```bash
# 1. clean eval gold (needs an LLM API key)          -> data/bench_gap/gold_sentences.tsv
python scripts/dataset/build_gap_gold.py --model openai/gpt-4o

# 2. harvest clean ACL limitation training data (no API; downloads LimGen, CC-BY-4.0)
python scripts/dataset/harvest_acl_limitations.py

# 3. train the shipped head (bge-small + 1500 ACL limitations)
python scripts/training/train_gap_head.py --encoder BAAI/bge-small-en-v1.5 --no-distant --acl-cap 1500 --out data/gap_head.joblib

# 4. benchmark (Stage A recall / Stage B / end-to-end)
python scripts/bench/bench_gap_recall.py --head data/gap_head.joblib

# 5. figures
python scripts/bench/plot_stage_a.py --head data/gap_head.joblib

# --- ablations (reproduce the "model/encoder doesn't matter" finding) ---
python scripts/training/test_bert_stageb.py --model distilbert-base-uncased
python scripts/training/test_setfit_stageb.py
python scripts/training/sweep_acl_cap.py
python scripts/training/sweep_encoders.py

# run the funnel over a corpus (no LLM)
python -m gap2idea.cli extract-gaps-funnel --mode hybrid --head data/gap_head.joblib
```

## Environment notes (reproducibility gotchas)
- Python 3.14, torch 2.10 CPU, transformers 5.5, sentence-transformers 5.5, sklearn 1.9.
- `setfit` package is **broken** here (transformers<5 + a Py3.14 `dill` bug) → SetFit
  reimplemented in raw torch in `test_setfit_stageb.py`.
- `sklearn>=1.7` dropped `LogisticRegression(multi_class=...)`; `transformers` 5.5
  dropped `default_logdir`.
- Set `PYTHONIOENCODING=utf-8` on Windows (ligatures/`ﬁ` in paper text).
- LLM steps (gold build) use OpenRouter; **credits were exhausted mid-project**,
  which capped the training-gold to 10/72 papers and is why the limitation fix
  came from ACL harvesting (no API) instead.
