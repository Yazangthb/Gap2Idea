"""
Analyse evaluator responses from the Gap2Idea evaluator Google Form.

Workflow
--------
1. In the form's "Responses" tab, click the green Sheets icon to create a
   linked spreadsheet.
2. In that spreadsheet: File -> Download -> Comma Separated Values (.csv).
3. Save locally and run:

       python scripts/eval/analyze_eval_responses.py path/to/responses.csv

   Output:
   - Markdown report -> artifacts/eval_analysis.md
   - Per-idea TSV    -> artifacts/eval_analysis.tsv (mean/std/n per axis)
   - Console summary

Dependencies
------------
- pandas (already a project dep)
- krippendorff (optional, for inter-rater alpha): pip install krippendorff
  If not installed, the script still runs but skips the alpha column.

The script is robust to:
- Subset of evaluators (some skipping ideas — though scale items are required)
- Reordered columns
- Extra columns from form edits (timestamp / email / name / expertise / comments)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import krippendorff  # type: ignore
    HAS_KRIPP = True
except ImportError:
    HAS_KRIPP = False


AXES = ["Novelty", "Feasibility", "Potential impact"]
IDEA_COL_RE = re.compile(r"^Idea\s+(\d+)\s*[—-]\s*(.+?)\s*$")


def parse_csv(csv_path: Path) -> pd.DataFrame:
    """Load the responses CSV and return it untouched (column names preserved)."""
    df = pd.read_csv(csv_path)
    # Strip whitespace from column names (Google sometimes adds trailing spaces).
    df.columns = [c.strip() for c in df.columns]
    return df


def discover_idea_columns(df: pd.DataFrame) -> dict[int, dict[str, str]]:
    """Map idea_number -> {axis: column_name}."""
    mapping: dict[int, dict[str, str]] = {}
    for col in df.columns:
        m = IDEA_COL_RE.match(col)
        if not m:
            continue
        idea_n = int(m.group(1))
        axis = m.group(2).strip()
        # Normalise common spellings
        axis = axis.replace("Pot.", "Potential")
        if axis not in AXES:
            # Tolerate "Impact" / "Potential impact" / etc.
            for canonical in AXES:
                if axis.lower() in canonical.lower() or canonical.lower() in axis.lower():
                    axis = canonical
                    break
        mapping.setdefault(idea_n, {})[axis] = col
    return mapping


# Domain assignment matches the build_evaluator_form_gs.py output order:
# ideas 1-5 = AI, 6-10 = Classical ML, 11-15 = Mathematics.
def domain_of(idea_n: int) -> str:
    if 1 <= idea_n <= 5:
        return "AI"
    if 6 <= idea_n <= 10:
        return "Classical ML"
    if 11 <= idea_n <= 15:
        return "Mathematics"
    return "Unknown"


def krippendorff_alpha(matrix: np.ndarray) -> Optional[float]:
    """
    matrix shape: (n_raters, n_items). NaN = missing.
    Ordinal Krippendorff's alpha. Returns None if not installed or undefined.
    """
    if not HAS_KRIPP:
        return None
    # Need at least 2 raters and 2 items with data
    if matrix.shape[0] < 2 or np.isnan(matrix).all():
        return None
    try:
        return float(
            krippendorff.alpha(
                reliability_data=matrix,
                level_of_measurement="ordinal",
            )
        )
    except Exception:
        return None


def fmt_score(mean: float, std: float, n: int) -> str:
    if n == 0 or np.isnan(mean):
        return "—"
    return f"{mean:.2f} ± {std:.2f} (n={n})"


def fmt_alpha(a: Optional[float]) -> str:
    if a is None:
        return "—"
    return f"{a:.2f}"


def analyse(df: pd.DataFrame, idea_cols: dict[int, dict[str, str]]) -> tuple[pd.DataFrame, dict]:
    """
    Returns (per-idea table, summary dict with alpha per axis).
    """
    rows = []
    # For Krippendorff: one matrix per axis, shape (n_raters, n_items)
    n_raters = len(df)
    alpha_matrices: dict[str, np.ndarray] = {
        axis: np.full((n_raters, len(idea_cols)), np.nan)
        for axis in AXES
    }
    sorted_ideas = sorted(idea_cols.keys())

    for j, idea_n in enumerate(sorted_ideas):
        row = {"idea": idea_n, "domain": domain_of(idea_n)}
        for axis in AXES:
            col = idea_cols[idea_n].get(axis)
            if col is None or col not in df.columns:
                row[f"{axis}_mean"] = np.nan
                row[f"{axis}_std"] = np.nan
                row[f"{axis}_n"] = 0
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
            row[f"{axis}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{axis}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0
            row[f"{axis}_n"] = int(len(vals))
            # Fill alpha matrix
            series = pd.to_numeric(df[col], errors="coerce").to_numpy()
            alpha_matrices[axis][: len(series), j] = series
        rows.append(row)

    per_idea = pd.DataFrame(rows)

    alphas = {axis: krippendorff_alpha(mat) for axis, mat in alpha_matrices.items()}

    return per_idea, alphas


def render_markdown(per_idea: pd.DataFrame, alphas: dict, csv_path: Path, n_raters: int) -> str:
    lines: list[str] = []
    lines.append("# Gap2Idea evaluator response analysis")
    lines.append("")
    lines.append(f"Source: `{csv_path.name}` — **{n_raters} evaluator(s)**.")
    lines.append("")
    lines.append("## Inter-rater agreement (Krippendorff's α, ordinal)")
    lines.append("")
    if not HAS_KRIPP:
        lines.append("> `krippendorff` package not installed. "
                     "Run `pip install krippendorff` and re-execute to populate this section.")
        lines.append("")
    else:
        lines.append("| Axis | α |")
        lines.append("|---|---|")
        for axis in AXES:
            lines.append(f"| {axis} | {fmt_alpha(alphas.get(axis))} |")
        lines.append("")
        lines.append("Interpretation: α ≥ 0.80 = strong agreement, "
                     "0.67-0.80 = tentative, <0.67 = unreliable. Likert / 5-point "
                     "with few raters often lands in the unreliable range — that's "
                     "useful information about the rubric itself.")
        lines.append("")

    # Per-domain table
    for domain in ["AI", "Classical ML", "Mathematics"]:
        sub = per_idea[per_idea["domain"] == domain].copy()
        if sub.empty:
            continue
        lines.append(f"## {domain}")
        lines.append("")
        lines.append("| Idea | Novelty | Feasibility | Potential impact |")
        lines.append("|---:|---|---|---|")
        for _, r in sub.iterrows():
            cells = [
                f"#{int(r['idea'])}",
                fmt_score(r["Novelty_mean"], r["Novelty_std"], r["Novelty_n"]),
                fmt_score(r["Feasibility_mean"], r["Feasibility_std"], r["Feasibility_n"]),
                fmt_score(r["Potential impact_mean"], r["Potential impact_std"], r["Potential impact_n"]),
            ]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    # Composite rank (mean of three axes, equal weight)
    per_idea = per_idea.copy()
    per_idea["composite"] = per_idea[[f"{a}_mean" for a in AXES]].mean(axis=1)
    ranked = per_idea.sort_values("composite", ascending=False)
    lines.append("## Composite ranking (mean across all three axes)")
    lines.append("")
    lines.append("| Rank | Idea | Domain | Composite | Novelty | Feasibility | Impact |")
    lines.append("|---:|---:|---|---:|---:|---:|---:|")
    for rank, (_, r) in enumerate(ranked.iterrows(), 1):
        lines.append(
            f"| {rank} | #{int(r['idea'])} | {r['domain']} | "
            f"{r['composite']:.2f} | "
            f"{r['Novelty_mean']:.2f} | "
            f"{r['Feasibility_mean']:.2f} | "
            f"{r['Potential impact_mean']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    csv_path = Path(sys.argv[1]).resolve()
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found.", file=sys.stderr)
        sys.exit(2)

    df = parse_csv(csv_path)
    idea_cols = discover_idea_columns(df)
    if not idea_cols:
        print("ERROR: no Idea N — Axis columns found. "
              "Did you export the right sheet?", file=sys.stderr)
        sys.exit(3)
    print(f"Found {len(idea_cols)} ideas, {len(df)} responses.")

    per_idea, alphas = analyse(df, idea_cols)

    artifacts = Path(__file__).resolve().parents[2] / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # Save TSV
    tsv_out = artifacts / "eval_analysis.tsv"
    per_idea.to_csv(tsv_out, sep="\t", index=False)
    print(f"Wrote {tsv_out}")

    # Save Markdown
    md_out = artifacts / "eval_analysis.md"
    md_out.write_text(
        render_markdown(per_idea, alphas, csv_path, n_raters=len(df)),
        encoding="utf-8",
    )
    print(f"Wrote {md_out}")

    # Console summary (plain ASCII so Windows cp1252 console doesn't choke;
    # the markdown report uses the proper alpha char).
    print()
    print("=" * 60)
    print("Inter-rater agreement (Krippendorff's alpha, ordinal):")
    for axis in AXES:
        print(f"  {axis:<20} {fmt_alpha(alphas.get(axis))}")
    print()
    print("Top 5 by composite score:")
    per_idea["composite"] = per_idea[[f"{a}_mean" for a in AXES]].mean(axis=1)
    top = per_idea.sort_values("composite", ascending=False).head(5)
    for _, r in top.iterrows():
        print(f"  #{int(r['idea']):>2}  {r['domain']:<14}  composite={r['composite']:.2f}")


if __name__ == "__main__":
    main()
