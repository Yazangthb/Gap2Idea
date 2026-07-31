"""Reproducible plots for the Stage-A explainer (docs/stage_a_explained.md).

Runs Stage A on the clean gap-sentence gold and renders two documentation
figures from REAL numbers (not hand-typed):

    docs/figures/stage_a_localization.png   localization recall vs containment tau
    docs/figures/stage_a_funnel.png         sentences/paper: full -> sliced -> emitted

Usage:
    python scripts/bench/plot_stage_a.py --head data/gap_head.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    extract_gaps, slice_terminal_regions, token_containment,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs" / "figures"
TYPES = ["limitation", "future_work"]
TAUS = [0.90, 0.80, 0.70]


def load():
    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences.tsv", sep="\t", dtype={"paper_id": str})
    mani = pd.read_csv(ROOT / "data/bench_gold/papers_manifest.tsv", sep="\t", dtype=str)
    texts = {}
    for src in {str(s) for s in mani["source"]}:
        for line in (ROOT / src).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("id")) in set(mani["id"]):
                    texts[str(r.get("id"))] = {"text": str(r.get("text", "")),
                                               "blocks": r.get("blocks") if isinstance(r.get("blocks"), list) else None}
    return gold, texts


def compute(gold, texts, head):
    cont, full, sliced, emitted, npapers = [], 0, 0, 0, len(texts)
    for pid, rec in texts.items():
        regions = slice_terminal_regions(rec["text"], blocks=rec["blocks"])
        slice_text = " ".join(s for r in regions for s in r.sentences)
        sliced += sum(len(r.sentences) for r in regions)
        full += len(split_sentences(_cut_before_references(rec["text"])))
        emitted += len(extract_gaps(pid, rec["text"], blocks=rec["blocks"], head=head, mode="hybrid"))
        for _, g in gold[gold["paper_id"] == pid].iterrows():
            cont.append((g["gap_type"], token_containment(g["gap_sentence"], slice_text)))
    cdf = pd.DataFrame(cont, columns=["gap_type", "containment"])
    rec_by_tau = {}
    for tau in TAUS:
        hit = cdf["containment"] >= tau
        rec_by_tau[tau] = {"all": hit.mean(),
                           **{t: hit[cdf.gap_type == t].mean() for t in TYPES}}
    return rec_by_tau, dict(full=full/npapers, sliced=sliced/npapers, emitted=emitted/npapers)


def plot_localization(rec_by_tau):
    cats = ["all", "future_work", "limitation"]
    colors = {"all": "#374151", "future_work": "#2563eb", "limitation": "#db2777"}
    x = np.arange(len(TAUS)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for k, c in enumerate(cats):
        vals = [rec_by_tau[t][c] for t in TAUS]
        bars = ax.bar(x + (k-1)*w, vals, w, label=c, color=colors[c])
        for b, v in zip(bars, vals):
            ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"τ = {t:.2f}" for t in TAUS])
    ax.set_ylabel("localization recall\n(gold gap sentence is inside the slice)")
    ax.set_ylim(0, 1.08); ax.axhline(0.9, ls="--", lw=0.8, color="#9ca3af")
    ax.set_title("Stage A — localization recall vs containment threshold\n(clean gold: 19 gaps / 9 papers)")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "stage_a_localization.png", dpi=150); plt.close(fig)


def plot_funnel(funnel):
    stages = ["Full paper\n(pre-refs)", "Stage A slice\n(kept)", "Emitted gaps\n(Stage B, hybrid)"]
    vals = [funnel["full"], funnel["sliced"], funnel["emitted"]]
    colors = ["#cbd5e1", "#60a5fa", "#16a34a"]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    bars = ax.bar(stages, vals, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("sentences per paper (log scale)")
    ax.set_title("Stage A is the free funnel: ~82% of sentences dropped before any model")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.08, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    drop_a = 1 - vals[1]/vals[0]
    ax.annotate(f"−{drop_a*100:.0f}%  (free, regex/CPU)", xy=(0.5, (vals[0]*vals[1])**0.5),
                ha="center", color="#1d4ed8", fontsize=10)
    ax.annotate("Stage B\nclassifies", xy=(1.5, (vals[1]*vals[2])**0.5), ha="center",
                color="#15803d", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG / "stage_a_funnel.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=Path("data/gap_head.joblib"))
    a = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    gold, texts = load()
    head = None
    if a.head.exists():
        from gap2idea.pipeline.gap_funnel import EmbeddingGapHead
        head = EmbeddingGapHead.load(a.head)
    rec_by_tau, funnel = compute(gold, texts, head)
    plot_localization(rec_by_tau); plot_funnel(funnel)
    print("localization:", {f"{t:.2f}": {k: round(v, 3) for k, v in d.items()} for t, d in rec_by_tau.items()})
    print("funnel/paper:", {k: round(v, 1) for k, v in funnel.items()})
    print("wrote docs/figures/stage_a_localization.png, stage_a_funnel.png")


if __name__ == "__main__":
    main()
