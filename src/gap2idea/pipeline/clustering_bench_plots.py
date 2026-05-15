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
    """Heatmap grid. Per-metric, drop rows that contain any NaN — those
    clusterers don't have meaningful values to compare across embedders
    (e.g. raw HDBSCAN returns 0 clusters → silhouette undefined)."""
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
        # Drop rows where every cell is NaN, then rows where any cell is NaN
        pivot = pivot.dropna(how="any")
        if pivot.empty:
            ax.set_title(f"{metric} (all rows NaN)")
            ax.axis("off")
            continue
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


def _top_words_per_cluster(texts: list[str], labels: np.ndarray, n_top: int = 3) -> dict[int, list[str]]:
    """Top tokens by frequency per cluster (lightweight, no gensim)."""
    import re
    from collections import Counter
    TOK = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
    STOP = frozenset(["that", "with", "this", "from", "have", "into", "such",
                      "their", "these", "than", "when", "which", "would", "where",
                      "while", "been", "more", "most", "some", "what", "thus",
                      "they", "them", "also", "since", "between", "based", "using",
                      "however", "further", "future", "work", "should"])
    by_cluster: dict[int, Counter] = {}
    for t, lab in zip(texts, labels):
        if lab == -1:
            continue
        cnt = by_cluster.setdefault(int(lab), Counter())
        for tok in TOK.findall(t.lower()):
            if tok in STOP:
                continue
            cnt[tok] += 1
    return {c: [w for w, _ in cnt.most_common(n_top)] for c, cnt in by_cluster.items()}


def plot_clusters_2d(
    bench_dir: Path,
    gaps_tsv: Path,
    clusterer: str,
    embedder: str,
    out_path: Path,
) -> None:
    """2D UMAP projection of gap embeddings, coloured by cluster label.

    Re-fits both the embedding and the chosen clusterer (cheap on a few-hundred
    statement corpus) so the plot reflects the exact same labels metrics.tsv
    used. Top-3 words per cluster are annotated at the cluster centroid.
    """
    import matplotlib.pyplot as plt
    from umap import UMAP

    from gap2idea.pipeline.clustering_bench import (
        embed_texts, fit_clusters, load_gaps,
    )

    texts = load_gaps(gaps_tsv)
    X = embed_texts(texts, embedder, bench_dir / "embeddings")
    labels, _ = fit_clusters(X, texts, clusterer)

    # 2D projection — UMAP cosine, small n_neighbors for tiny corpora
    n = X.shape[0]
    reducer = UMAP(n_components=2, n_neighbors=max(2, min(15, n - 1)),
                   metric="cosine", random_state=0)
    X2 = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(9, 7))
    unique = sorted(np.unique(labels).tolist())
    # Stable colour palette: noise (-1) → grey
    cmap = plt.cm.get_cmap("tab10", max(10, len(unique)))
    for i, lab in enumerate(unique):
        mask = labels == lab
        color = "#cccccc" if lab == -1 else cmap(i % cmap.N)
        marker_label = "noise" if lab == -1 else f"c{lab}"
        ax.scatter(X2[mask, 0], X2[mask, 1], s=40, alpha=0.7,
                   color=color, label=marker_label, edgecolors="none")

    # Annotate cluster centroids with top-3 words
    top_words = _top_words_per_cluster(texts, labels, n_top=3)
    for lab, words in top_words.items():
        mask = labels == lab
        if mask.sum() == 0:
            continue
        cx, cy = X2[mask, 0].mean(), X2[mask, 1].mean()
        ax.annotate(", ".join(words), (cx, cy), ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="black", lw=0.5, alpha=0.85))

    ax.set_title(f"{clusterer} × {embedder}  (n_gaps={n}, n_clusters={len([u for u in unique if u != -1])})")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_clusters_grid(
    bench_dir: Path,
    gaps_tsv: Path,
    embedder: str,
    clusterers: list[str],
    out_path: Path,
) -> None:
    """Side-by-side 2D projections for a single embedder across all clusterers."""
    import matplotlib.pyplot as plt
    from umap import UMAP

    from gap2idea.pipeline.clustering_bench import (
        embed_texts, fit_clusters, load_gaps,
    )

    texts = load_gaps(gaps_tsv)
    X = embed_texts(texts, embedder, bench_dir / "embeddings")
    n = X.shape[0]
    reducer = UMAP(n_components=2, n_neighbors=max(2, min(15, n - 1)),
                   metric="cosine", random_state=0)
    X2 = reducer.fit_transform(X)

    n_panels = len(clusterers)
    n_cols = 3 if n_panels > 2 else n_panels
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, cls in zip(axes, clusterers):
        labels, _ = fit_clusters(X, texts, cls)
        unique = sorted(np.unique(labels).tolist())
        cmap = plt.cm.get_cmap("tab10", max(10, len(unique)))
        for i, lab in enumerate(unique):
            mask = labels == lab
            color = "#cccccc" if lab == -1 else cmap(i % cmap.N)
            ax.scatter(X2[mask, 0], X2[mask, 1], s=22, alpha=0.7,
                       color=color, edgecolors="none")
        n_cls = len([u for u in unique if u != -1])
        n_noise = int((labels == -1).sum())
        ax.set_title(f"{cls}  (k={n_cls}, noise={n_noise})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    for ax in axes[len(clusterers):]:
        ax.set_visible(False)

    fig.suptitle(f"Cluster shapes on 2-D UMAP — embedder: {embedder}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_all_plots(bench_dir: Path,
                   gaps_tsv: Path | None = None,
                   showcase: tuple[str, str] | None = None,
                   grid_embedder: str | None = None,
                   ) -> Path:
    """Generate metric plots; if `gaps_tsv` is provided also draw 2-D cluster
    scatters for one (clusterer, embedder) showcase combo and one all-clusterers
    grid on a chosen embedder."""
    plots_dir = bench_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(bench_dir / "metrics.tsv", sep="\t")

    plot_heatmap_grid(metrics, plots_dir / "metric_heatmaps.png")
    plot_stability_bars(metrics, plots_dir / "stability_bars.png")
    plot_silhouette_vs_npmi(metrics, plots_dir / "silhouette_vs_npmi.png")

    if gaps_tsv is not None:
        showcase = showcase or ("agglomerative", "BAAI/bge-small-en-v1.5")
        grid_embedder = grid_embedder or showcase[1]
        try:
            plot_clusters_2d(bench_dir, gaps_tsv, showcase[0], showcase[1],
                             plots_dir / f"clusters_{showcase[0]}_{showcase[1].replace('/', '_')}.png")
        except Exception as e:  # noqa: BLE001
            log.warning("plot_clusters_2d failed: %s", e)
        # All available clusterers from metrics.tsv
        all_clusterers = sorted(metrics["clusterer"].unique().tolist())
        try:
            plot_clusters_grid(bench_dir, gaps_tsv, grid_embedder, all_clusterers,
                               plots_dir / f"clusters_grid_{grid_embedder.replace('/', '_')}.png")
        except Exception as e:  # noqa: BLE001
            log.warning("plot_clusters_grid failed: %s", e)

    return plots_dir
