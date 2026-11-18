# Retrospective arXiv training-set exposure: initial audit

This is an initial source audit, not a freeze plan or evidence that all earlier development has been reconstructed. It uses the files at `submission-make` commit `a8df2f6` and performs no new model inference. The 134-paper check remains retrospective and must not be described as an independent held-out test.

## Confirmed exposure

- `data/gold/splits.json` contains 162 training papers; joining paper IDs to the domains in `gap_gold_final.jsonl` gives **134 arXiv** and 28 SAL papers. The 134 are part of the 230-paper gold corpus that was LLM-labelled before splitting (`gold_dataset_results.md`).
- `scripts/bench/benchmark_arxiv_latex.py` reads the entire gold corpus without selecting a split. Its report `stage_a_latex_sections.md` compares section-recovery strategies over the arXiv corpus and argues against a section gate on arXiv. Among these 134 training IDs, 122 have nonempty LaTeX blocks, ten are recorded as PDF-only, and two as having no sections. The decisions were informed by an aggregate analysis, not a documented separate decision for each of those 122 papers.
- `scripts/training/export_nli_ft_data.py` explicitly selects the arXiv training split and gold labels for fine-tuning; `nli_finetune.md` records 134 papers. The fine-tuned Stage B model must not be evaluated as independent on that same split.
- `scripts/training/train_stage_a.py` and `train_gold_head.py` describe training fitted components using this paper-level train split. This establishes exposure for those fitted components, not that every resulting version is the one used in a proposed retrospective experiment.

## Scope and remaining audit work

Before freezing a retrospective run, inspect the other scripts, experiment logs, prompts, and session records for decisions informed by training-paper text or labels, including Stage A/B rules and Stage C rubric examples. The source review above is not exhaustive and does not establish which individual sentences were read by researchers. Fix the exact pipeline, model revisions, inputs, seed and proposed analysis in a dated plan before inference, and disclose every known exposure. Coordinate any Qwen inference with Hamza. Results are descriptive evidence on previously used papers; they do not establish performance on unseen papers or independently confirm test-set findings.
