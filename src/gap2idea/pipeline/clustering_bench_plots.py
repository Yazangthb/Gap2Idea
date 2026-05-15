"""Plots for the clustering-quality benchmark.

Inputs:  data/clustering_bench/metrics.tsv  (long format)
Outputs: data/clustering_bench/plots/*.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)


def _heatmap(ax, pivot: pd.DataFrame, title: str, cmap: str = "viridis", fmt: str = "{:.2f}") -> None:
    arr = pivot.values.astype(float)
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(title)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            txt = "n/a" if np.isnan(v) else fmt.format(v)
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if not np.isnan(v) and v > np.nanmean(arr) else "black",
                    fontsize=8)
    return im


def plot_heatmap_grid(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("silhouette", "viridis", "{:.2f}"),
        ("davies_bouldin", "viridis_r", "{:.2f}"),
        ("npmi", "viridis", "{:.2f}"),
        ("bootstrap_mean_ari", "viridis", "{:.2f}"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (metric, cmap, fmt) in zip(axes.flat, panels):
        sub = metrics[metrics["metric"] == metric]
        if sub.empty:
            ax.set_visible(False)
            continue
        pivot = sub.pivot(index="clusterer", columns="embedder", values="value")
        _heatmap(ax, pivot, metric, cmap=cmap, fmt=fmt)
    fig.suptitle("Clustering benchmark — metric heatmaps (rows: clusterer, cols: embedder)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_stability_bars(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    mean = metrics[metrics["metric"] == "bootstrap_mean_ari"].pivot(
        index="clusterer", columns="embedder", values="value")
    std = metrics[metrics["metric"] == "bootstrap_std_ari"].pivot(
        index="clusterer", columns="embedder", values="value")
    if mean.empty:
        log.warning("no bootstrap_mean_ari rows; skipping stability bars")
        return

    clusterers = list(mean.index)
    embedders = list(mean.columns)
    x = np.arange(len(clusterers))
    w = 0.8 / max(1, len(embedders))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, emb in enumerate(embedders):
        m = mean[emb].values
        s = std[emb].values
        ax.bar(x + (i - (len(embedders) - 1) / 2) * w, m, width=w, yerr=s,
               capsize=3, label=emb)
    ax.set_xticks(x)
    ax.set_xticklabels(clusterers)
    ax.set_ylabel("mean Adjusted Rand Index (bootstrap)")
    ax.set_title("Stability — mean ± std ARI over bootstrap resamples")
    ax.axhline(0, color="#888", linewidth=0.7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_silhouette_vs_npmi(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    pivot = metrics.pivot_table(index=["clusterer", "embedder"], columns="metric",
                                values="value").reset_index()
    if "silhouette" not in pivot.columns or "npmi" not in pivot.columns:
        log.warning("missing silhouette or npmi; skipping scatter")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for cls, group in pivot.groupby("clusterer"):
        ax.scatter(group["silhouette"], group["npmi"], label=cls, s=80, alpha=0.8)
        for _, r in group.iterrows():
            ax.annotate(r["embedder"].split("/")[-1][:10], (r["silhouette"], r["npmi"]),
                        fontsize=7, alpha=0.7, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel("silhouette (cosine)")
    ax.set_ylabel("NPMI")
    ax.set_title("Do geometry and topic coherence agree on a winner?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_all_plots(bench_dir: Path) -> Path:
    plots_dir = bench_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(bench_dir / "metrics.tsv", sep="\t")

    plot_heatmap_grid(metrics, plots_dir / "metric_heatmaps.png")
    plot_stability_bars(metrics, plots_dir / "stability_bars.png")
    plot_silhouette_vs_npmi(metrics, plots_dir / "silhouette_vs_npmi.png")
    return plots_dir
