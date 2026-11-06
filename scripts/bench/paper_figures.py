# -*- coding: utf-8 -*-
"""Draw the manuscript's result figures straight from the saved result files, so no number is retyped.

Reads docs/experiments/{stage_c_modes,nli_sal_fws,stage_c_cost,stage_c_nli_qwen}.json (gitignored, local)
and writes manuscript_mdpi/figures/{modes,funnel,types,cost}.pdf.

The only hand-entered values are the gold Stage A counts (9604 sentences, 644 gaps), which no JSON file
stores; they come from docs/experiments/stage_c_nli_qwen.md and match Table 1 of the manuscript.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "docs/experiments"
OUT = ROOT / "manuscript_mdpi/figures"

GOLD_STAGE_A = {"kept": 9604, "tp": 644}  # docs/experiments/stage_c_nli_qwen.md, stage table

# First three categorical slots of the dataviz reference palette (validated all-pairs, light mode).
# Markers carry identity too, so the figures still read in grayscale print.
MODES = [("high_recall", "High recall", "#eb6834", "^"),
         ("balanced", "Balanced", "#1baf7a", "s"),
         ("high_precision", "High precision", "#2a78d6", "o")]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino Linotype", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(name):
    return json.loads((EXP / name).read_text(encoding="utf-8"))


def grid(ax, axis="both"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def fig_modes(m, n, q):
    """Precision against recall for the three modes, with 95% CIs, one panel per evaluation set."""
    before_gold = dict(zip("PR", q["r90"]["before"]["PRF"][:2]))
    sets = [("SAL-40 test", lambda k: n["sal_test"][k], n["sal_test"]["before_stage_c"]),
            ("SAL dev", lambda k: n["sal_dev"][k], n["sal_dev"]["before_stage_c"]),
            ("FWS test", lambda k: n["fws_test"][k], n["fws_test"]["before_stage_c"]),
            ("Gold test (LLM labels)", lambda k: m[k]["all"], {"P": {"value": before_gold["P"]},
                                                                "R": {"value": before_gold["R"]}})]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.15), sharex=True, sharey=True)
    r = np.linspace(0.01, 1, 300)
    for ax, (title, get, before) in zip(axes, sets):
        grid(ax)
        for f in (0.5, 0.6, 0.7, 0.8):  # iso-F1 curves: P = fR / (2R - f)
            p = f * r / (2 * r - f)
            ok = (2 * r > f) & (p <= 1.0)
            ax.plot(r[ok], p[ok], color=MUTED, linewidth=0.5, linestyle=(0, (2, 2)), zorder=1)
            if title.startswith("Gold"):  # label where the curve meets the right edge of the last panel
                ax.text(0.993, f * 0.99 / (1.98 - f) - 0.012, f"$F_1$={f}", fontsize=6, color=MUTED,
                        ha="right", va="top")
        pts = [get(k) for k, *_ in MODES]
        ax.plot([s["R"]["value"] for s in pts], [s["P"]["value"] for s in pts], color=MUTED, linewidth=0.8, zorder=2)
        ax.plot(before["R"]["value"], before["P"]["value"], marker="D", markersize=5, color="white",
                markeredgecolor=INK2, markeredgewidth=0.9, linestyle="none", zorder=3)
        for (k, label, color, marker), s in zip(MODES, pts):
            (rl, rh), (pl, ph) = s["R"]["ci95"], s["P"]["ci95"]
            rv, pv = s["R"]["value"], s["P"]["value"]
            ax.errorbar(rv, pv, xerr=[[rv - rl], [rh - rv]], yerr=[[pv - pl], [ph - pv]], fmt=marker, color=color,
                        markersize=5.5, markeredgecolor="white", markeredgewidth=0.7, elinewidth=0.9, capsize=0,
                        zorder=4)
        ax.set_title(title, color=INK, pad=4)
        ax.set_xlim(0.45, 1.0)
        ax.set_ylim(0.2, 1.0)
        ax.set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    handles = [plt.Line2D([], [], marker=mk, color=c, markeredgecolor="white", linestyle="none", markersize=6)
               for _, _, c, mk in MODES]
    handles.append(plt.Line2D([], [], marker="D", color="white", markeredgecolor=INK2, linestyle="none", markersize=5))
    fig.legend(handles, [lab for _, lab, *_ in MODES] + ["Before Stage C"], loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.08), handletextpad=0.3, columnspacing=1.4)
    fig.subplots_adjust(wspace=0.12)
    fig.savefig(OUT / "modes.pdf")
    plt.close(fig)


def fig_funnel(m, n, c, q):
    """Precision and recall after each stage, high precision mode, with the number of sentences kept."""
    def pr(kept, tp, gaps):
        return tp / kept, tp / gaps

    sal, fws = n["sal_test"], n["fws_test"]
    g_hp, g_gaps = m["high_precision"]["all"], m["high_precision"]["all"]["gaps"]
    g_b = q["r90"]["before"]
    sets = [
        ("SAL-40 test", [("Input", sal["sentences"], sal["positives"]),
                         ("A", sal["stage_a"]["kept"], sal["stage_a"]["tp"]),
                         ("B", sal["before_stage_c"]["kept"], sal["before_stage_c"]["tp"]),
                         ("C", sal["high_precision"]["kept"], sal["high_precision"]["tp"])], sal["positives"]),
        ("FWS test", [("Input", fws["sentences"], fws["positives"]),
                      ("B", fws["before_stage_c"]["kept"], fws["before_stage_c"]["tp"]),
                      ("C", fws["high_precision"]["kept"], fws["high_precision"]["tp"])], fws["positives"]),
        ("Gold test (LLM labels)", [("Input", c["gold_all"]["sentences"], g_gaps),
                                    ("A", GOLD_STAGE_A["kept"], GOLD_STAGE_A["tp"]),
                                    ("B", g_b["kept"], g_b["tp"]),
                                    ("C", g_hp["kept"], g_hp["tp"])], g_gaps),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3), sharey=True)
    for ax, (title, stages, gaps) in zip(axes, sets):
        grid(ax, "y")
        x = np.arange(len(stages))
        P, R = zip(*(pr(k, t, gaps) for _, k, t in stages))
        ax.plot(x, np.array(R) * 100, color="#eb6834", marker="^", markersize=5, linewidth=1.6,
                markeredgecolor="white", markeredgewidth=0.6, label="Recall")
        ax.plot(x, np.array(P) * 100, color="#2a78d6", marker="o", markersize=5, linewidth=1.6,
                markeredgecolor="white", markeredgewidth=0.6, label="Precision")
        for xi, p, rr in zip(x, P, R):  # upper series labelled above its point, lower one below (or above if near 0)
            for v, upper in ((100 * p, p >= rr), (100 * rr, rr > p)):
                above = upper or v < 12
                ax.text(xi + (0.12 if (above and not upper) else 0), v + 4 if above else v - 5, f"{v:.1f}",
                        ha="left" if (above and not upper) else "center", va="bottom" if above else "top",
                        fontsize=6.3, color=INK2)
        ax.set_xticks(x, [f"{s}\n{k:,}" for s, k, _ in stages], fontsize=6.8)
        ax.set_xlabel("Stage (sentences kept)", fontsize=7)
        ax.set_xlim(-0.35, len(stages) - 0.65)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, color=INK, pad=4)
    axes[0].set_ylabel("%")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h[::-1], lab[::-1], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(OUT / "funnel.pdf")
    plt.close(fig)


def fig_types(m):
    """Recall by gap type on the gold test split, per mode."""
    types = [("limitation", "Limitation"), ("future_work", "Future work"), ("research_gap", "Research gap")]
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    grid(ax, "y")
    w, gap = 0.25, 0.02
    x = np.arange(len(types))
    for i, (k, label, color, _) in enumerate(MODES):
        vals = [100 * m[k]["recall_by_type"][t] for t, _ in types]
        pos = x + (i - 1) * (w + gap)
        ax.bar(pos, vals, width=w, color=color, label=label, edgecolor="white", linewidth=0.8)
        for p, v in zip(pos, vals):
            ax.text(p, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xticks(x, [lab for _, lab in types])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Recall (%)")
    ax.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.16), handlelength=1.0)
    fig.savefig(OUT / "types.pdf")
    plt.close(fig)


def fig_cost(c):
    """Input tokens per paper: funnel (high precision mode) against one whole-paper call and every sentence."""
    sets = [("sal40_nli", "SAL-40"), ("fws_nli", "FWS"), ("gold_arxiv", "Gold, arXiv"), ("gold_all", "Gold, all")]
    series = [("every_sentence", "Every sentence", "#eb6834", "^"), ("whole_paper", "Whole paper", "#1baf7a", "s"),
              ("high_precision", "Funnel", "#2a78d6", "o")]
    fig, ax = plt.subplots(figsize=(4.6, 1.9))
    grid(ax, "x")
    y = np.arange(len(sets))[::-1]
    for yi, (k, _) in zip(y, sets):
        d = c[k]
        vals = [d[s]["tokens"] / d["papers"] for s, *_ in series]
        ax.plot([min(vals), max(vals)], [yi, yi], color=GRID, linewidth=2.2, zorder=1, solid_capstyle="round")
        for (s, lab, color, mk), v in zip(series, vals):
            ax.plot(v, yi, marker=mk, color=color, markersize=6.5, markeredgecolor="white", markeredgewidth=0.7,
                    linestyle="none", zorder=3)
        ax.text(vals[1] * 1.12 if vals[1] > vals[2] else vals[2] * 1.12, yi + 0.2,
                f"{d['high_precision']['whole_paper_times']:.2f}× fewer than whole paper", fontsize=6.3,
                color=INK2, va="bottom")
    ax.set_xscale("log")
    ax.set_yticks(y, [lab for _, lab in sets])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(sets) - 0.2)
    ax.set_xlabel("Estimated LLM input tokens per paper (log scale)")
    handles = [plt.Line2D([], [], marker=mk, color=col, markeredgecolor="white", linestyle="none", markersize=6.5)
               for _, _, col, mk in series]
    ax.legend(handles[::-1], [lab for _, lab, *_ in series][::-1], loc="upper center", ncol=3, frameon=False,
              bbox_to_anchor=(0.45, 1.2))
    fig.savefig(OUT / "cost.pdf")
    plt.close(fig)


def main():
    m, n, c, q = (load(f) for f in ("stage_c_modes.json", "nli_sal_fws.json", "stage_c_cost.json",
                                    "stage_c_nli_qwen.json"))
    assert q["r90"]["before"]["kept"] == m["candidates"], "gold Stage B count differs between result files"
    OUT.mkdir(parents=True, exist_ok=True)
    fig_modes(m, n, q)
    fig_funnel(m, n, c, q)
    fig_types(m)
    fig_cost(c)
    print("wrote", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
