"""Compare metrics.tsv across bench variants. Writes data/bench_ablation.md."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

LABELS = [
    ("data/bench",            "v1 (text + old regex)  [N=10]"),
    ("data/bench_v2a",        "v2a (text + new vocab) [N=10]"),
    ("data/bench_v2b",        "v2b (PDF + style)      [N=10]"),
    ("data/bench_n100_v2a",   "v2a (text + new vocab) [N=100]"),
    ("data/bench_n100",       "v2b (PDF + style)      [N=100]"),
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
    lines.append("\n## Stage 1 — regex section parser (higher is better)\n")
    lines.append("| variant | rouge1_f | rouge2_f | rougeL_f |")
    lines.append("|---|---:|---:|---:|")
    for _, label in LABELS:
        c = cells[label]
        lines.append(f"| {label} | {c['rouge1_f']:.3f} | {c['rouge2_f']:.3f} | {c['rougeL_f']:.3f} |")

    # Stage 2 — LLM gap extraction. Pull from same metrics.tsv files if rows exist.
    llm_metrics = ["n_gaps", "mean_sim_to_gold", "recovery_at_0.6", "hallucination_at_0.6"]
    llm_cells: dict[str, dict[str, float]] = {}
    for path, label in LABELS:
        p = Path(path) / "metrics.tsv"
        if not p.exists():
            llm_cells[label] = {m: float("nan") for m in llm_metrics}
            continue
        df = pd.read_csv(p, sep="\t")
        sub = df[df["stage"] == "llm_gap"]
        llm_cells[label] = {m: float(sub[sub["metric"] == m]["value"].mean()) for m in llm_metrics}

    lines.append("\n## Stage 2 — LLM gap extraction (recovery higher better, hallucination lower better)\n")
    lines.append("| variant | n_gaps | mean_sim_to_gold | recovery@0.6 | hallucination@0.6 |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, label in LABELS:
        c = llm_cells[label]
        lines.append(f"| {label} | {c['n_gaps']:.2f} | {c['mean_sim_to_gold']:.3f} | "
                     f"{c['recovery_at_0.6']:.3f} | {c['hallucination_at_0.6']:.3f} |")

    # Stage 2 oracle comparison — pull from the N=100 v2b run (oracle was rerun there)
    p = Path("data/bench_n100/metrics.tsv")
    if p.exists():
        df = pd.read_csv(p, sep="\t")
        sub = df[df["stage"] == "pipeline_vs_oracle"]
        if not sub.empty:
            lines.append("\n## Pipeline gaps vs Oracle gaps (gold section fed straight to LLM)\n")
            lines.append("Two systems, same Stage-2 LLM. Oracle skips Stage 1.\n")
            lines.append("| metric | mean | meaning |")
            lines.append("|---|---:|---|")
            for m, meaning in [
                ("n_oracle_gaps",              "gaps the LLM produces on the gold section"),
                ("mean_sim_pipe_to_oracle",    "avg cosine: each pipeline gap → closest oracle gap"),
                ("mean_sim_oracle_to_pipe",    "avg cosine: each oracle gap → closest pipeline gap"),
                ("recovery_at_0.6",            "fraction of pipeline gaps that match an oracle gap"),
                ("coverage_at_0.6",            "fraction of oracle gaps the pipeline reproduced"),
            ]:
                v = sub[sub["metric"] == m]["value"].mean()
                lines.append(f"| {m} | {v:.3f} | {meaning} |")

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
