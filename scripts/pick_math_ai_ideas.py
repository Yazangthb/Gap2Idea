"""Filter the generated ideas into 3 math + 3 AI, ranked by panel score.

Topic is assigned heuristically from the cluster_id the idea came from,
based on the cluster contents we inspected before running:

  Math clusters (pure or math-dominant):
    4  — Polynomial Freiman-Ruzsa + random matrix + combinatorial designs
    10 — Bounds with limited derivative access (optimisation theory)
    14 — Wall-crossing on non-projective Calabi-Yau threefolds
    12 — Efficient scalable model merging (mathematical aspects)

  Mixed cluster — assign topic by inspecting the generated idea's title/method:
    5  — OT + PDE  AND  KV-cache + speculative decoding

  Everything else is AI.

Use:
  PYTHONIOENCODING=utf-8 python scripts/pick_math_ai_ideas.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Negative cluster IDs come from the targeted math-only driver
# (gen_math_ideas.py). They are always math by construction.
MATH_KEYWORDS = (
    "freiman", "ruzsa", "polynomial freiman", "random matrix",
    "spectral universal", "erdős", "calabi", "moduli", "wall-crossing",
    "optimal transport", "brenier", "wasserstein", "helmholtz",
    "combinatorial design", "additive combinatorics",
    "k-resilient", "singleton bound", "neural discretisation",
    "neural discretization", "convergence rate", "entropic",
    "log-concav", "bulk eigenvalue", "sparse symmetric",
    "non-projective", "semistable", "regularity", "lipschitz domain",
)
AI_OVERRIDE_KEYWORDS = (
    "deep hedging", "reinforcement learning", "rlhf", "rlaif",
    "speculative decoding", "kv-cache", "kv cache", "diffusion model",
    "transformer attention", "domain generalization", "formant tracking",
    "volume rendering", "style-gan", "stylegan", "nerf",
    "language model", "llm ", "cybersecurity", "drug",
    "federated", "code agent", "constitutional",
)


def classify(idea_row: pd.Series) -> str:
    """Return 'math' or 'ai' for one idea."""
    cid = int(idea_row.get("cluster_a", 0))
    if cid < 0:                           # synthetic IDs from gen_math_ideas.py
        return "math"
    haystack = " ".join(
        str(idea_row.get(c, "")) for c in (
            "title", "research_question", "method_sketch",
            "evaluation_plan", "expected_contribution",
        )
    ).lower()
    ai_hits   = sum(1 for k in AI_OVERRIDE_KEYWORDS if k in haystack)
    math_hits = sum(1 for k in MATH_KEYWORDS         if k in haystack)
    # Helmholtz-on-its-own is a numerical-PDE / math signal even if the
    # method invokes adversarial PGD (which would otherwise look AI).
    if "helmholtz" in haystack and "stability" in haystack:
        return "math"
    if math_hits > ai_hits:
        return "math"
    return "ai"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    root = Path(__file__).resolve().parent.parent
    ideas_path = root / "artifacts" / "ideas.tsv"
    if not ideas_path.exists():
        raise SystemExit(f"missing {ideas_path}; run generate-ideas first")
    df = pd.read_csv(ideas_path, sep="\t")
    if df.empty:
        raise SystemExit("ideas.tsv is empty")

    df["topic"] = df.apply(classify, axis=1)
    df["panel_composite"] = pd.to_numeric(df.get("panel_composite", 0), errors="coerce").fillna(0)
    df["panel_agreement"] = pd.to_numeric(df.get("panel_agreement", 0), errors="coerce").fillna(0)
    df = df.sort_values("panel_composite", ascending=False).reset_index(drop=True)

    print("=" * 80)
    print(f"All {len(df)} generated ideas, sorted by panel_composite:")
    print("=" * 80)
    for _, r in df.iterrows():
        print(f"[{r['topic']:4s}]  cluster={r['cluster_a']:>2}  "
              f"composite={r['panel_composite']:.2f}  α={r['panel_agreement']:.2f}  "
              f"{str(r['title'])[:90]}")

    for topic in ("math", "ai"):
        print()
        print("=" * 80)
        print(f"  Top 3 {topic.upper()} ideas")
        print("=" * 80)
        top = df[df["topic"] == topic].head(3)
        if top.empty:
            print(f"  (no {topic} ideas found in this batch)")
            continue
        for i, (_, r) in enumerate(top.iterrows(), 1):
            print(f"\n  #{i}  composite={r['panel_composite']:.2f}  "
                  f"α={r['panel_agreement']:.2f}  cluster={r['cluster_a']}")
            print(f"  TITLE:  {r['title']}")
            print(f"  RQ:     {r['research_question']}")
            ms = str(r.get("method_sketch", ""))
            print(f"  METHOD: {ms[:400]}{'...' if len(ms) > 400 else ''}")
            print(f"  BASELINE: {r.get('named_baseline', '(none)')}")
            print(f"  FALSIFIABLE: {r.get('falsifiable_prediction', '(none)')}")


if __name__ == "__main__":
    main()
