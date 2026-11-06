"""Positive-class metrics and paired, paper-cluster percentile bootstrap."""
from __future__ import annotations

import numpy as np


def counts(y, pred):
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    if y.shape != pred.shape or y.ndim != 1:
        raise ValueError("Labels and predictions must be aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError("Only binary labels/predictions are supported")
    return np.array([np.sum((y == 1) & (pred == 1)),
                     np.sum((y == 0) & (pred == 1)),
                     np.sum((y == 1) & (pred == 0)),
                     np.sum((y == 0) & (pred == 0))], dtype=np.int64)


def from_counts(c):
    c = np.asarray(c)
    tp, fp, fn, tn = [c[..., k] for k in range(4)]
    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * tp, 2 * tp + fp + fn)
    negative_f1 = div(2 * tn, 2 * tn + fp + fn)
    return dict(precision=precision, recall=recall, f1=f1,
                macro_f1=(f1 + negative_f1) / 2)


def point(y, pred):
    c = counts(y, pred)
    return {**{k: float(v) for k, v in from_counts(c).items()},
            **dict(zip(("tp", "fp", "fn", "tn"), map(int, c)))}


def select_threshold(y, scores, objective="f1"):
    """Fixed validation grid. Ties prefer proximity to .5, then higher threshold."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or len(scores) != len(y):
        raise ValueError("Invalid/misaligned validation scores")
    if not len(y) or len(set(map(int, y))) != 2:
        raise ValueError("Validation must contain both classes")
    grid = np.round(np.arange(0.05, 1.0, 0.01), 2)
    evaluated = [(float(t), point(y, scores >= t)) for t in grid]
    threshold, metrics = max(evaluated, key=lambda x: (
        x[1][objective], -abs(x[0] - .5), x[0]))
    return threshold, metrics


def bootstrap(rows, pred, reference=None, n=2000, seed=0):
    """Resample whole papers; pool sentence confusion counts in each replicate.

    With reference, returns method-minus-reference differences using identical
    resamples. Does not resample training or threshold selection uncertainty.
    """
    if n < 1:
        raise ValueError("bootstrap replicates must be positive")
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row["paper_id"], []).append(i)
    if len(pred) != len(rows) or (reference is not None and len(reference) != len(rows)):
        raise ValueError("Predictions do not cover the full evaluation set")
    y = np.asarray([r["y"] for r in rows])
    pred = np.asarray(pred)
    c = np.array([counts(y[ix], pred[ix]) for ix in groups.values()])
    ref = None if reference is None else np.asarray(reference)
    rc = None if ref is None else np.array([counts(y[ix], ref[ix]) for ix in groups.values()])
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("precision", "recall", "f1", "macro_f1")}
    # Chunking limits RAM even when bootstrap count / paper count are large.
    for start in range(0, n, 200):
        ix = rng.integers(0, len(c), size=(min(200, n - start), len(c)))
        vals = from_counts(c[ix].sum(axis=1))
        if rc is not None:
            rv = from_counts(rc[ix].sum(axis=1))
            vals = {k: v - rv[k] for k, v in vals.items()}
        for k, v in vals.items():
            samples[k].extend(v.tolist())
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in samples.items()}
