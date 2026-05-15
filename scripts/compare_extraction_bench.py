"""Compare metrics.tsv across bench variants. Writes data/bench_ablation.md."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LABELS = [
    ("data/bench",      "v1 (text + old regex)"),
    ("data/bench_v2a",  "v2a (text + new vocab)"),
    ("data/bench_v2b",  "v2b (PDF + style + new vocab)"),
]
HEADLINE = ["rouge1_f", "rouge2_f", "rougeL_f"]


def main(out_md: Path) -> None:
    cells: dict[str, dict[str, float]] = {}
    for path, label in LABELS:
        p = Path(path) / "metrics.tsv"
        if not p.exists():
            cells[label] = {m: float("nan") for m in HEADLINE}
            continue
        df = pd.read_csv(p, sep="\t")
        sub = df[df["stage"] == "regex_section"]
        cells[label] = {m: float(sub[sub["metric"] == m]["value"].mean()) for m in HEADLINE}

    lines: list[str] = []
    lines.append("# Extraction-bench ablation (N=10 papers, unarXive gold)\n")
    lines.append("Regex stage only. Lower-is-better → none here (all higher-is-better).\n")
    lines.append("\n| variant | rouge1_f | rouge2_f | rougeL_f |")
    lines.append("|---|---:|---:|---:|")
    for _, label in LABELS:
        c = cells[label]
        lines.append(f"| {label} | {c['rouge1_f']:.3f} | {c['rouge2_f']:.3f} | {c['rougeL_f']:.3f} |")

    # Per-paper diff between v1 and v2b
    if (Path("data/bench/metrics.tsv").exists()
            and Path("data/bench_v2b/metrics.tsv").exists()):
        v1 = pd.read_csv("data/bench/metrics.tsv", sep="\t")
        v2 = pd.read_csv("data/bench_v2b/metrics.tsv", sep="\t")
        v1p = v1[(v1.stage == "regex_section") & (v1.metric == "rouge1_f")].set_index("id")["value"]
        v2p = v2[(v2.stage == "regex_section") & (v2.metric == "rouge1_f")].set_index("id")["value"]
        lines.append("\n## Per-paper rouge1_f, v1 → v2b\n")
        lines.append("| paper_id | gold title (truncated) | v1 | v2b | Δ |")
        lines.append("|---|---|---:|---:|---:|")
        # gold titles
        gold = pd.read_json("data/bench_v2b/bench_papers.jsonl", lines=True)
        gold_titles = {row["id"]: ", ".join(row["gold_section_titles"])[:50]
                       for _, row in gold.iterrows()}
        for pid in sorted(set(v1p.index) | set(v2p.index)):
            a, b = v1p.get(pid, float("nan")), v2p.get(pid, float("nan"))
            title = gold_titles.get(pid, "")
            lines.append(f"| `{pid}` | {title} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/bench_ablation.md")
    main(out)
