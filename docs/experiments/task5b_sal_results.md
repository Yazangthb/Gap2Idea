# Task 5b: completed SAL comparison

Full run completed 2026-11-05; packaged on branch `Task_5b` on 2026-11-06.
This is the full SAL test, separate from the implementation smoke checks.

## Results

40 test papers, 7968 sentences, 205 gold limitation sentences. Thresholds were
selected on the original SAL validation split before test scoring. Positive-class
metrics and 95% percentile confidence intervals use 2000 whole-paper bootstrap
resamples (seed 0), conditional on the fitted model and selected threshold.

| Method | Precision | Recall | Positive F1 [95% CI] | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + logistic regression | .473 | .380 | .422 [.355, .488] | .704 |
| Binary ngram + BernoulliNB | .000 | .000 | .000 [.000, .000] | .493 |
| Kilicoglu section proxy | .467 | .478 | .472 [.363, .559] | .729 |
| ModernBERT NLI | .331 | .585 | .423 [.377, .470] | .701 |

The rule baseline substitutes contiguous sections for unavailable paragraphs; it
is an approximation, not a claim of exact reproduction of the original Java
method. The NB model predicted two positives, both false positives; its zero F1
is a completed negative result, not an omitted or failed experiment. These
results do not establish superiority over the funnel: the intended complete
aligned funnel reference is still pending.

## Versioned evidence

The complete original run is tracked at
[`runs/task5b/sal-full-01/sal/`](../../runs/task5b/sal-full-01/sal/), including:

- [Human-readable report](../../runs/task5b/sal-full-01/sal/report.md) and
  [machine-readable results](../../runs/task5b/sal-full-01/sal/results.json).
- Source/input/environment manifests, preregistered methods, frozen validation
  settings, test-start and completion markers.
- Per-method validation and full-test sentence predictions, scores, metrics and
  confidence intervals. The classical joblib models and NLI score arrays are
  included so the saved artifact hashes remain checkable.

Original run metadata intentionally retains the pre-commit base revision, source
hashes, working branch and local input paths. It records how the run actually
executed; it was not rewritten to suggest that the later publication commit
existed during training. Input corpora/prepared datasets and downloaded neural
weights remain outside Git; recover them using the runbook and input hashes.
Git attributes preserve the result files' original bytes and keep experiment
source line endings at LF, so automatic Windows/Linux conversion does not change
their recorded SHA256 hashes.
Only this completed SAL run was explicitly added from the otherwise ignored
`runs/task5b/` directory. Failed smoke runs are not included.

Before publication, all experiment-source, frozen-setting and model-artifact
hashes were verified against the original run. Metrics and all four 2000-resample
paper confidence intervals were recomputed from saved predictions; no model was
rerun on the test set.

## Pending work

- **FWS update:** five methods have completed under the user-approved current-gold
  protocol, which replaces the old training-file dependency. See
  [the current audit](task5b_status.md). The new FWS run still needs publication
  to Git; it does not establish identity with the old breadth-funnel test pool.
- **arXiv update:** gold labels/texts/splits are restored and prepared; baseline
  runs are still pending.
- **Paired funnel comparisons:** await complete aligned reference predictions.
- **Lan EDA/PBA and other secondary methods:** see the
  [runbook](task5b_runbook.md) and [registered protocol](baseline_methods.md).

For a future completed FWS run, review the output and explicitly stage only its
directory (for example `git add -f runs/task5b/fws-full-01/`); avoid force-adding
the entire ignored runs tree.
