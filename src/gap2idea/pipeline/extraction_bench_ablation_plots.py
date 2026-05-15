"""Ablation plots across bench variants and the oracle condition.

Reads metrics.tsv from each variant directory and produces:

  ablation_stage1.png        Stage 1 ROUGE-1/2/L bars (v1 / v2a / v2b)
  ablation_stage2.png        Stage 2 LLM-gap metrics across variants
  pipeline_vs_oracle.png     recovery vs coverage at tau in {0.5, 0.6, 0.7}
  per_paper_pipe_vs_oracle.png  per-paper pipeline gap count vs oracle gap count
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)

DEFAULT_VARIANTS = [
    ("v2a", Path("data/bench_n100_v2a")),
    ("v2b", Path("data/bench_n100")),
]


def _load(variants: list[tuple[str, Path]]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for label, p in variants:
        m = p / "metrics.tsv"
        if not m.exists():
            log.warning("missing %s — skipping %s", m, label)
            continue
        out[label] = pd.read_csv(m, sep="\t", dtype={"id": str})
    return out


def _agg(metrics: dict[str, pd.DataFrame], stage: str, metric: str) -> dict[str, float]:
    """Mean of a metric per variant. NaN if missing."""
    out: dict[str, float] = {}
    for label, df in metrics.items():
        sub = df[(df.stage == stage) & (df.metric == metric)]["value"]
        out[label] = float(sub.mean()) if not sub.empty else float("nan")
    return out


# ----------------------------------------------------------------------
# Stage 1 ablation
# ----------------------------------------------------------------------

def plot_ablation_stage1(metrics: dict[str, pd.DataFrame], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(metrics.keys())
    rs = ["rouge1_f", "rouge2_f", "rougeL_f"]
    data = np.array([[_agg(metrics, "regex_section", r)[lbl] for r in rs] for lbl in labels])
    # data shape: (n_variants, 3)

    x = np.arange(len(rs))
    w = 0.8 / max(1, len(labels))
    colors = ["#7d7d7d", "#a86fb8", "#3b78c2"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, lbl in enumerate(labels):
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, data[i], width=w,
               color=colors[i % len(colors)], label=lbl)
        for j, v in enumerate(data[i]):
            if np.isnan(v):
                continue
            ax.text(x[j] + (i - (len(labels) - 1) / 2) * w, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(rs)
    ax.set_ylim(0, max(0.6, np.nanmax(data) * 1.15))
    ax.set_ylabel("F1 score (higher better)")
    ax.set_title("Stage 1 — predicted section vs unarXive gold (ROUGE)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ----------------------------------------------------------------------
# Stage 2 ablation
# ----------------------------------------------------------------------

def plot_ablation_stage2(metrics: dict[str, pd.DataFrame], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = list(metrics.keys())
    keep = [
        ("mean_sim_to_gold", "mean sim → gold"),
        ("recovery_at_0.6",   "recovery @ τ=0.6"),
        ("hallucination_at_0.6", "hallucination @ τ=0.6"),
    ]
    data = np.array([[_agg(metrics, "llm_gap", m)[lbl] for m, _ in keep] for lbl in labels])
    n_gaps = [_agg(metrics, "llm_gap", "n_gaps")[lbl] for lbl in labels]

    x = np.arange(len(keep))
    w = 0.8 / max(1, len(labels))
    colors = ["#7d7d7d", "#a86fb8", "#3b78c2"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, lbl in enumerate(labels):
        # halluc is lower-better; we just plot raw value, label clarifies
        ax.bar(x + (i - (len(labels) - 1) / 2) * w, data[i], width=w,
               color=colors[i % len(colors)], label=f"{lbl}  (n_gaps={n_gaps[i]:.1f})")
        for j, v in enumerate(data[i]):
            if np.isnan(v):
                continue
            ax.text(x[j] + (i - (len(labels) - 1) / 2) * w, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([disp for _, disp in keep])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("value")
    ax.set_title("Stage 2 — LLM gap-extraction metrics across variants")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ----------------------------------------------------------------------
# Pipeline vs Oracle
# ----------------------------------------------------------------------

def plot_pipeline_vs_oracle(metrics_v2b: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = metrics_v2b[metrics_v2b.stage == "pipeline_vs_oracle"]
    if sub.empty:
        log.warning("no pipeline_vs_oracle rows in v2b — skipping")
        return

    taus = [0.5, 0.6, 0.7]
    rec_means, rec_stds, cov_means, cov_stds = [], [], [], []
    for t in taus:
        r = sub[sub.metric == f"recovery_at_{t}"]["value"]
        c = sub[sub.metric == f"coverage_at_{t}"]["value"]
        rec_means.append(r.mean()); rec_stds.append(r.std())
        cov_means.append(c.mean()); cov_stds.append(c.std())

    x = np.arange(len(taus))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - w / 2, rec_means, yerr=rec_stds, width=w, capsize=4,
           color="#3b78c2", label="recovery (pipeline gap → oracle)")
    ax.bar(x + w / 2, cov_means, yerr=cov_stds, width=w, capsize=4,
           color="#d97757", label="coverage (oracle gap ← pipeline)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"τ = {t}" for t in taus])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction of gaps matched")
    ax.set_title("v2b pipeline vs Oracle (LLM fed gold section)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ----------------------------------------------------------------------
# Per-paper agreement
# ----------------------------------------------------------------------

def plot_per_paper_pipe_vs_oracle(metrics_v2b: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    sub = metrics_v2b[metrics_v2b.stage == "pipeline_vs_oracle"]
    if sub.empty:
        log.warning("no pipeline_vs_oracle rows in v2b — skipping")
        return
    pivot = sub.pivot(index="id", columns="metric", values="value")
    pivot = pivot.sort_index()
    keep_cols = ["recovery_at_0.6", "coverage_at_0.6"]
    if not all(c in pivot.columns for c in keep_cols):
        log.warning("missing per-paper recovery/coverage columns — skipping")
        return

    x = np.arange(len(pivot))
    w = 0.4
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - w / 2, pivot["recovery_at_0.6"].values, width=w,
           color="#3b78c2", label="recovery @ 0.6")
    ax.bar(x + w / 2, pivot["coverage_at_0.6"].values, width=w,
           color="#d97757", label="coverage @ 0.6")
    ax.set_xticks(x)
    ax.set_xticklabels([p if len(p) <= 14 else p[:13] + "…" for p in pivot.index],
                       rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction at τ = 0.6")
    ax.set_title("Per-paper agreement: v2b pipeline vs Oracle gap extraction")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    log.info("Wrote %s", out_path)


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

def make_all_plots(
    out_dir: Path = Path("data/bench_ablation_plots"),
    variants: list[tuple[str, Path]] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = variants or DEFAULT_VARIANTS
    metrics = _load(variants)
    if not metrics:
        raise RuntimeError("No metrics found for any variant")

    plot_ablation_stage1(metrics, out_dir / "ablation_stage1.png")
    plot_ablation_stage2(metrics, out_dir / "ablation_stage2.png")

    if "v2b" in metrics:
        plot_pipeline_vs_oracle(metrics["v2b"], out_dir / "pipeline_vs_oracle.png")
        plot_per_paper_pipe_vs_oracle(metrics["v2b"], out_dir / "per_paper_pipe_vs_oracle.png")
    return out_dir
