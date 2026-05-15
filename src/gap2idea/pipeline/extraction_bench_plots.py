"""Plots for the extraction-quality benchmark.

Reads `data/bench/metrics.tsv` (long format produced by extraction_bench.py)
and writes PNGs to `data/bench/plots/`.

Plots produced:
  1. regex_rouge_per_paper.png      ROUGE-1/2/L bars per paper + mean line
  2. regex_section_types.png         Which fallback the regex landed in
  3. llm_recovery_vs_halluc.png      Recovery vs hallucination at 3 thresholds
  4. llm_sim_per_paper.png           Mean cosine sim to gold and to full per paper
  5. summary_bars.png                One bar per aggregate metric (mean ± std)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)


def _load(metrics_tsv: Path, sections_jsonl: Path | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    m = pd.read_csv(metrics_tsv, sep="\t", dtype={"id": str})
    s = None
    if sections_jsonl is not None and sections_jsonl.exists():
        s = pd.read_json(sections_jsonl, lines=True, dtype=False)
        s["id"] = s["id"].astype(str)
    return m, s


def _short_id(pid: str, n: int = 14) -> str:
    return pid if len(pid) <= n else pid[: n - 1] + "…"


def plot_regex_rouge_per_paper(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = metrics[(metrics["stage"] == "regex_section") &
                  metrics["metric"].isin(["rouge1_f", "rouge2_f", "rougeL_f"])]
    pivot = sub.pivot(index="id", columns="metric", values="value").sort_index()
    pivot = pivot[["rouge1_f", "rouge2_f", "rougeL_f"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    w = 0.27
    colors = ["#3b78c2", "#7fb069", "#d97757"]
    for i, col in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * w, pivot[col].values, width=w, label=col, color=colors[i])
    means = pivot.mean()
    for i, col in enumerate(pivot.columns):
        ax.axhline(means[col], color=colors[i], linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_id(p) for p in pivot.index], rotation=40, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("ROUGE F1")
    ax.set_title("Stage 1 (regex) — predicted-section vs unarXive gold")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_regex_section_types(sections: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    counts = sections["section_type"].value_counts()
    # canonical ordering for readability
    order = ["limitations", "future_work", "discussion", "fallback", "tail"]
    counts = counts.reindex([k for k in order if k in counts.index])
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values,
                  color=["#3b78c2", "#7fb069", "#a86fb8", "#d97757", "#7d7d7d"][: len(counts)])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(int(v)), ha="center", va="bottom")
    ax.set_ylabel("# sections picked")
    ax.set_title("Stage 1 — which branch the regex landed in")
    ax.set_ylim(0, max(counts.values) + 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_llm_recovery_vs_halluc(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    taus = [0.5, 0.6, 0.7]
    rec_means, rec_stds, hal_means, hal_stds = [], [], [], []
    for tau in taus:
        r = metrics[(metrics["stage"] == "llm_gap") &
                    (metrics["metric"] == f"recovery_at_{tau}")]["value"]
        h = metrics[(metrics["stage"] == "llm_gap") &
                    (metrics["metric"] == f"hallucination_at_{tau}")]["value"]
        rec_means.append(r.mean()); rec_stds.append(r.std())
        hal_means.append(h.mean()); hal_stds.append(h.std())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(taus))
    w = 0.36
    ax.bar(x - w / 2, rec_means, width=w, yerr=rec_stds, capsize=4,
           color="#7fb069", label="recovery (↑ better)")
    ax.bar(x + w / 2, hal_means, width=w, yerr=hal_stds, capsize=4,
           color="#d97757", label="hallucination (↓ better)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ = {t}" for t in taus])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of gap sentences")
    ax.set_title("Stage 2 (LLM gaps) — recovery vs hallucination by threshold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_llm_sim_per_paper(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = metrics[(metrics["stage"] == "llm_gap") &
                  metrics["metric"].isin(["mean_sim_to_gold", "mean_sim_to_full", "n_gaps"])]
    pivot = sub.pivot(index="id", columns="metric", values="value").sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(pivot))
    w = 0.4
    ax.bar(x - w / 2, pivot["mean_sim_to_gold"].values, width=w,
           color="#3b78c2", label="mean cosine → gold section")
    ax.bar(x + w / 2, pivot["mean_sim_to_full"].values, width=w,
           color="#a86fb8", label="mean cosine → full paper")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_id(p) for p in pivot.index], rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean cosine similarity")
    ax.set_title("Stage 2 — per-paper LLM gap-sentence similarity")
    ax.legend(loc="lower right", frameon=False)
    # Annotate n_gaps above each pair
    for xi, pid in zip(x, pivot.index):
        n = int(pivot.loc[pid, "n_gaps"])
        ax.text(xi, 0.98, f"n={n}", ha="center", va="top", fontsize=8, color="#444")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def plot_summary_bars(metrics: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    keep = [
        ("regex_section", "rouge1_f"),
        ("regex_section", "rouge2_f"),
        ("regex_section", "rougeL_f"),
        ("llm_gap", "mean_sim_to_gold"),
        ("llm_gap", "mean_sim_to_full"),
        ("llm_gap", "recovery_at_0.6"),
        ("llm_gap", "hallucination_at_0.6"),
    ]
    rows = []
    for stage, metric in keep:
        v = metrics[(metrics["stage"] == stage) & (metrics["metric"] == metric)]["value"]
        rows.append({"label": f"{stage}\n{metric}", "mean": v.mean(), "std": v.std()})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    colors = ["#3b78c2"] * 3 + ["#7fb069", "#a86fb8", "#7fb069", "#d97757"]
    ax.bar(df["label"], df["mean"], yerr=df["std"], capsize=4, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("value")
    ax.set_title("Benchmark summary (mean ± std over 10 papers)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


def make_all_plots(bench_dir: Path) -> Path:
    plots_dir = bench_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics, sections = _load(bench_dir / "metrics.tsv", bench_dir / "sections_extracted.jsonl")

    plot_regex_rouge_per_paper(metrics, plots_dir / "regex_rouge_per_paper.png")
    if sections is not None and not sections.empty:
        plot_regex_section_types(sections, plots_dir / "regex_section_types.png")
    if (metrics["stage"] == "llm_gap").any():
        plot_llm_recovery_vs_halluc(metrics, plots_dir / "llm_recovery_vs_halluc.png")
        plot_llm_sim_per_paper(metrics, plots_dir / "llm_sim_per_paper.png")
    plot_summary_bars(metrics, plots_dir / "summary_bars.png")
    return plots_dir
