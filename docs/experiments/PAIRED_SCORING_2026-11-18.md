# Paired scoring: baseline and funnel

`baseline_report.py` scores each saved baseline against the full evaluation universe, including the SAL AS exclusions, and calls the paired bootstrap in the direction **funnel minus baseline**. Complete recorded failures remain negative predictions with their failure counts; unattempted calls make the row incomplete and suppress paired comparisons. Bootstrap intervals resample papers 2,000 times. An interval including zero means the difference is unresolved, not that methods are equal.

## Original 27B check

After running `fetch_external.py` and `build_eval_files.py`, run `python scripts/bench/verify_published_pairs.py`. This recalculates all nine published Qwen3.8-27B funnel-versus-whole-paper differences and intervals from the saved predictions and verifies the committed result JSON. It requires no model inference or BioC network requests. It was run successfully for all nine dataset/mode pairs on 18 November 2026. Historical FWS whole-paper prompting uses a different rubric from the funnel; reproduce that caveat when reporting it.

## Comparisons for new runs

Use `export_funnel_comparison.py` only with evidence from the actual funnel run. It requires three inputs:

1. A JSON manifest with `name`, `dataset` (`gold`, `sal` or `fws`), `split` (`test`), `mode`, the actual run's complete `model_identity`, `evaluation_data_sha256`, and `candidate_keys_sha256`.
2. A JSON array of every ordered candidate key `[paper_id, sentence_index]`; the manifest's candidate digest is `baseline_evidence.digest(candidate_array)`.
3. A JSONL file containing one `{ "key": [paper_id, sentence_index], "keep": true_or_false }` verdict for every candidate.

The evaluation-data digest is `baseline_evidence.digest(baseline_qwen.load_rows(dataset, "test"))`. The exporter checks it, the candidate digest, key uniqueness and universe membership, Boolean verdicts and complete candidate coverage. It records canonical digests of all three inputs in the output. Select the baseline setup(s) using `--input-types A AS C` as applicable, and save outside any existing run folder:

```bash
python scripts/bench/export_funnel_comparison.py --manifest FUNNEL_MANIFEST.json --candidates CANDIDATES.json --verdicts VERDICTS.jsonl --input-types A AS C --out comparisons.json
python scripts/bench/baseline_report.py RUN_ROOT --comparisons comparisons.json
```

The report matches the full model identity exactly, in addition to dataset, mode and baseline input type. Do not copy a baseline's identity onto a historical funnel run merely to make a comparison appear; the original 27B and smaller-model funnel files do not carry the new complete identity manifests. The published 27B reproduction above is a separate historical comparison, not proof that every later H100 or V100 run has the same runtime identity. Collect or establish valid funnel-run provenance before producing any new cross-run paired claim. Any difference in prompt, type information, context, answer format or execution environment must be reported; a matched model does not by itself establish causality.

These scorer and analysis changes must be used from a separate checkout after an existing inference campaign is frozen. Do not replace the frozen campaign code or prompt files in a running campaign; its manifests hash the code. The original baseline data, verdicts, and reports are never overwritten.
