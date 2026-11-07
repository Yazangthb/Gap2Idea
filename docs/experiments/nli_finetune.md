# Fine-tuned ModernBERT-large-nli for Stage B (2026-11-07)

Can Stage B rank gap sentences better if the NLI model is trained on our gold sentence labels? Threshold tuning
alone cannot help: on the zero-shot scores a single tau only trades recall for precision, and per-type taus are
worse (more candidates at the same recall).

**Setup.** `tasksource/ModernBERT-large-nli` (rev ca476cb9) fine-tuned as NLI pairs: each sentence with the three
Stage B hypotheses ("This is a limitation." / "This is future work." / "This is a research gap."); entailment
when the gold type matches, neutral for another gap type, contradiction for non-gaps. Training data: gold TRAIN,
arXiv papers only (134 papers, 33,404 Stage A survivors, 2,534 gaps, 100,212 pairs). The 28 SAL-40 papers in the
gold train split were left out so SAL-40 stays unseen. Settings fixed before training: lr 1e-5, batch 32, 2 epochs,
6% warmup, weight decay 0.01, fp16, max length 256, seed 0. The epoch with the highest gold-validation precision
at the Stage B rule was kept (epoch 1). One V100, about 35 minutes.

**Rule.** Same as the funnel: Stage A, then the highest tau with recall >= 90% on the set's own development data
(gold validation; SAL dev; FWS dev), counting Stage A losses. Scores are entail / (entail + contradiction).

| Validation (gold) | tau | Candidates | Precision | Recall |
|---|---|---|---|---|
| Zero-shot (epoch 0) | 0.232 | 2,982 | 21.1% | 90.1% |
| Fine-tuned, epoch 1 (kept) | 0.002 | 1,746 | 36.1% | 90.1% |
| Fine-tuned, epoch 2 | 0.000 | 1,900 | 33.2% | 90.1% |

## Test

| Set (labels) | Model | Candidates | Precision | Recall |
|---|---|---|---|---|
| Gold test, 34 papers (LLM) | Zero-shot, frozen tau 0.354 | 2,522 | 24.0% | 91.3% |
| | Zero-shot, tau from the full-validation rule (0.232) | 3,277 | 19.0% | 93.8% |
| | **Fine-tuned** | **1,998** | **30.4%** | 91.4% |
| – arXiv, 28 papers | Zero-shot 0.354 / fine-tuned | 2,137 / 1,670 | 22.2% / 28.7% | 90.1% / 91.1% |
| – biomedical, 6 papers | Zero-shot 0.354 / fine-tuned | 385 / 328 | 34.0% / 38.7% | 95.6% / 92.7% |
| SAL-40 (human) | Zero-shot | **639** | **29.1%** | 90.7% |
| | Fine-tuned | 702 | 26.6% | 91.2% |
| FWS test (human) | Zero-shot | **400** | **48.3%** | 87.7% |
| | Fine-tuned | 411 | 45.7% | 85.5% |

Gold test recall by type, fine-tuned: limitation 87.7%, future work 93.8%, research gap 93.1% (zero-shot at 0.354:
86.4%, 89.7%, 94.8%).

## Reading

- On gold, fine-tuning cuts Stage C's input by about a fifth (1,998 vs 2,522 candidates) and raises candidate
  precision from 24.0% to 30.4% at the same recall.
- On the two human-labelled benchmarks it is slightly worse than zero-shot (SAL-40 precision -2.5 points, FWS -2.6).
  FWS is computer-science text like the training papers, so domain shift alone does not explain it; the model
  more likely learned how the LLM labellers mark gaps. The 6 biomedical gold-test papers improve under LLM labels,
  while SAL-40 (which includes those papers) gets worse under human labels, which points the same way.
- So the gain is tied to the LLM labels. The human study would show whether it is real.

Files: `scripts/training/export_nli_ft_data.py` (data), `scripts/training/finetune_nli.py` (GPU box),
`scripts/training/eval_nli_ft.py` (this table, `docs/experiments/nli_finetune.json`), scores in
`data/gold/nli_ft/`. The checkpoint (`best.pt`, 1.6 GB) is on the V100 box in `~/nli_ft/out/`.
