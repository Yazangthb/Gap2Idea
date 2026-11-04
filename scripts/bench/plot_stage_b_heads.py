# -*- coding: utf-8 -*-
"""Stage B head comparison on the gap-gold dataset — grouped TEST P/R/F1 bars.

Sources (TEST split, gap = any type, binary threshold tuned on VAL):
- docs/experiments/gold_detector_results.md  (frozen-encoder sweep, scripts/training/train_gold_head.py)
- docs/experiments/stage_b_zeroshot.md       (zero-shot NLI cross-encoder, scripts/bench/zeroshot_stage_b.py)

Writes docs/figures/stage_b_head_comparison.{png,pdf}.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# (label, TEST P, TEST R, TEST F1) — see sources above, ordered by TEST F1.
# NOTE: zero-shot NLI row pending (scripts/bench/zeroshot_stage_b.py — needs ~2.5 GB
# free VRAM or ~3 h on CPU); add ("zero-shot NLI (DeBERTa)\nno training", P, R, F)
# from docs/experiments/stage_b_zeroshot.md when it lands.
HEADS = [
    ("bge-small + logreg",    0.328, 0.733, 0.453),
    ("PubMedBERT + logreg",   0.327, 0.681, 0.442),
    ("SPECTER + logreg",      0.342, 0.608, 0.438),
    ("old pipeline head",     0.174, 0.658, 0.275),
]

# dataviz reference palette, slots 1-3 (validated, light surface)
C_P, C_R, C_F = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, BASE, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

x = np.arange(len(HEADS))
w = 0.26
p = [h[1] for h in HEADS]; r = [h[2] for h in HEADS]; f = [h[3] for h in HEADS]

b1 = ax.bar(x - w, p, w, color=C_P, label="Precision", zorder=3)
b2 = ax.bar(x,     r, w, color=C_R, label="Recall",    zorder=3)
b3 = ax.bar(x + w, f, w, color=C_F, label="F1",        zorder=3)

# direct value labels on every bar (contrast relief for the aqua F1)
for bars in (b1, b2, b3):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center",
                    fontsize=8, color=INK2)

ax.set_xticks(x, [h[0] for h in HEADS], fontsize=9)
ax.set_ylabel("TEST metric (binary gap = any type)", fontsize=9.5, color=INK2)
ax.set_ylim(0, max(max(r), max(f)) * 1.18)
ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(BASE)
ax.tick_params(colors=INK2, length=0)
ax.legend(frameon=False, fontsize=9, loc="upper right", ncols=3,
          handlelength=1.4, columnspacing=1.2)

fig.text(0.085, 0.985, "Stage B heads on the gap-gold dataset (test split, 34 papers)",
         fontsize=12.5, color=INK, va="top", fontweight="bold")
fig.text(0.085, 0.935,
         "Paper-disjoint train/val/test (162/34/34); binary threshold tuned on val; all-sentence test at "
         "realistic prevalence; LLM-ensemble gold",
         fontsize=8.5, color=INK2, va="top")
fig.text(0.085, 0.905,
         "All heads = frozen encoder + multinomial logreg (per-sentence, 4-class). "
         "Zero-shot NLI baseline pending — see docs/experiments/stage_b_zeroshot.md.",
         fontsize=8.5, color=INK2, va="top")

fig.tight_layout(rect=[0, 0, 1, 0.885])
out = ROOT / "docs" / "figures" / "stage_b_head_comparison"
fig.savefig(out.with_suffix(".png"), facecolor=SURF)
fig.savefig(out.with_suffix(".pdf"), facecolor=SURF)
print("wrote", out.with_suffix(".png"), "and .pdf")
