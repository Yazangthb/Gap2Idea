"""Generate 3 math ideas by feeding math-only evidence directly into the
orchestrated synthesiser → critic → judge panel.

The existing pipeline's clustering mixed math content (PFR, random matrices,
optimal transport, PDE, combinatorial designs, Calabi-Yau) with non-math
gaps in the same clusters, so within-mode kept producing AI ideas. This
driver constructs three math-only evidence batches and calls the
orchestration pieces directly, then appends the results to ideas.tsv.

Each batch combines two related math sub-themes (bridge-style), giving the
synthesiser a clear cross-pollination target.

Usage:
  PYTHONIOENCODING=utf-8 python scripts/gen/gen_math_ideas.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


from gap2idea.pipeline.agents import synthesise_with_critic
from gap2idea.pipeline.evaluation import (
    DEFAULT_JUDGE_PANEL, _aggregate_panel, _call_judge,
    _normalise_judge_scores, _falsifiability_gate,
)
from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client


# ---------- math evidence batches ----------

BATCHES = [
    {
        "label_a": "Additive Combinatorics: quantitative bounds",
        "label_b": "Random matrix universality",
        "fed_a": [
            {"paper_id": "2311.05762", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "An open problem is to derive a quantitative version of the Polynomial "
                "Freiman-Ruzsa theorem over the integers with the same asymptotic exponent "
                "as the characteristic-2 case.",
             "paragraph_text":
                "Our results establish the Polynomial Freiman-Ruzsa conjecture for vector "
                "spaces over GF(2) with explicit polynomial bounds. The corresponding "
                "integer statement is qualitatively known but the best quantitative "
                "exponents remain far apart."},
            {"paper_id": "2311.05762", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will extend the entropy-compression argument to the non-abelian "
                "setting, in particular to small-doubling subsets of solvable groups.",
             "paragraph_text":
                "We outline two natural directions: the non-abelian extension and a "
                "quantitative Bogolyubov-Ruzsa lemma whose convexity parameters match the "
                "recent polynomial bound."},
        ],
        "fed_b": [
            {"paper_id": "2406.22421", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Whether universality of local spectral statistics holds for Erdős-Rényi "
                "adjacency matrices with mean degree d growing slower than log n is an "
                "open problem.",
             "paragraph_text":
                "Existing universality results require d ≥ (log n)^c for some c > 1. The "
                "obstruction in the sparser regime is the lack of concentration of the "
                "local resolvent at the spectral edge."},
            {"paper_id": "2406.22421", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "An important direction is the universality of bulk eigenvalue statistics "
                "for sparse symmetric matrices with heavy-tailed entries beyond the "
                "second-moment regime.",
             "paragraph_text":
                "We focused on Gaussian-like entry distributions to keep the second-moment "
                "estimates clean. The relevant heuristic suggests a transition at a "
                "degree-dependent threshold under heavy-tailed entries."},
        ],
    },
    {
        "label_a": "Entropic optimal transport",
        "label_b": "Numerical PDEs at high frequency",
        "fed_a": [
            {"paper_id": "2503.09214", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "The sample complexity of entropic Brenier maps in dimensions d ≥ 100 "
                "beyond log-concave source measures is an open problem we leave to future "
                "work.",
             "paragraph_text":
                "Our convergence rate is dimension-free under log-concavity but degrades "
                "exponentially in d for general sub-Gaussian source measures."},
            {"paper_id": "2503.09214", "gap_type": "limitation", "confidence": 0.95,
             "gap_sentence":
                "Our convergence rate is dimension-free only under the log-concavity "
                "assumption, which excludes the heavy-tailed targets common in applied "
                "optimal-transport problems.",
             "paragraph_text":
                "We use the log-concavity assumption in two places: the Brascamp-Lieb step "
                "and the variance bound on the dual potential."},
        ],
        "fed_b": [
            {"paper_id": "2502.04419", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Provably stable neural-network discretisations for high-frequency Helmholtz "
                "operators on non-convex Lipschitz domains remain an open problem.",
             "paragraph_text":
                "Standard finite-element methods suffer from the pollution effect at high "
                "wavenumber k, and existing neural alternatives lack a-priori stability "
                "bounds independent of k."},
            {"paper_id": "2502.04419", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "We leave to future work the analysis of stability and convergence rates "
                "when the network is trained with stochastic gradient methods rather than "
                "the convex relaxation considered here.",
             "paragraph_text":
                "Our theoretical guarantees assume the network reaches a global optimum of "
                "the residual loss. Preliminary experiments suggest stability degrades by "
                "a factor logarithmic in the wavenumber."},
        ],
    },
    {
        "label_a": "Combinatorial designs at the Singleton bound",
        "label_b": "Algebraic geometry: moduli wall-crossing",
        "fed_a": [
            {"paper_id": "2409.11763", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Constructing K-resilient combinatorial designs with block size 2K + 1 and "
                "density approaching the Singleton bound for K ≥ 6 is an open problem.",
             "paragraph_text":
                "We achieve the Singleton bound for K ≤ 5 via an explicit polynomial "
                "construction. The natural extension breaks at K = 6 because the auxiliary "
                "covering code stops existing."},
        ],
        "fed_b": [
            {"paper_id": "2410.21908", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "An important direction is extending the wall-crossing formula for moduli "
                "of semistable sheaves to non-projective Calabi-Yau threefolds where "
                "current numerical invariants are undefined.",
             "paragraph_text":
                "Our derivation of the wall-crossing formula relies crucially on "
                "projectivity to define the stability function. A starting point would be "
                "the local Calabi-Yau case where Bridgeland stability is well-developed."},
            {"paper_id": "2311.05762", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will extend the entropy-compression argument to the non-abelian "
                "setting, in particular to small-doubling subsets of solvable groups.",
             "paragraph_text":
                "We outline two natural directions for entropy-compression arguments under "
                "weaker group hypotheses."},
        ],
    },
]


async def judge_one(idea: dict, judge_models: list[str]) -> dict:
    client = get_llm_client()
    results: list[tuple[str, dict]] = []
    for m in judge_models:
        try:
            raw = _call_judge(client, idea, model=m)
            results.append((m, _normalise_judge_scores(raw)))
        except Exception as e:
            print(f"  judge {m} failed: {e}", file=sys.stderr)
    if not results:
        return {"composite": 0.0, "agreement": 0.0, "n_judges": 0}
    return _aggregate_panel(results)


async def one_batch(batch: dict, idx: int) -> dict:
    print(f"\n[{idx}] synthesising: {batch['label_a']} × {batch['label_b']}")
    result = await synthesise_with_critic(
        mode="bridge",
        cluster_a=-100 - idx,             # synthetic marker so it doesn't collide
        cluster_b=-200 - idx,
        label_a=batch["label_a"],
        label_b=batch["label_b"],
        gaps_df=pd.DataFrame(),           # unused in bridge mode
        fed_evidence_a=batch["fed_a"],
        fed_evidence_b=batch["fed_b"],
        max_iterations=1,
        accept_score=4.0,
    )
    idea = result["idea"]
    print(f"    critic verdict={result['_critique_history'][-1]['verdict']} "
          f"score={result['_critique_history'][-1]['score']:.2f}")
    consensus = await judge_one(idea, DEFAULT_JUDGE_PANEL)
    print(f"    panel composite={consensus.get('composite', 0):.2f} "
          f"agreement={consensus.get('agreement', 0):.2f}")
    return {
        "mode": "math-targeted",
        "cluster_a": -100 - idx,
        "cluster_b": -200 - idx,
        "label_a": batch["label_a"],
        "label_b": batch["label_b"],
        "title": idea["title"],
        "research_question": idea["research_question"],
        "method_sketch": idea["method_sketch"],
        "evaluation_plan": idea["evaluation_plan"],
        "expected_contribution": idea["expected_contribution"],
        "assumptions_and_risks": idea["assumptions_and_risks"],
        "falsifiable_prediction": idea.get("falsifiable_prediction", ""),
        "named_baseline": idea.get("named_baseline", ""),
        "idea_confidence": float(idea["confidence"]),
        "evidence_used_json": json.dumps(idea.get("evidence_used", []),
                                          ensure_ascii=False),
        "novelty_score": None,
        "max_similarity_to_prior": None,
        "closest_paper_title": "",
        "closest_paper_year": "",
        "closest_paper_id": "",
        "n_critic_iterations": result.get("_n_iterations", 0),
        "panel_composite": consensus.get("composite"),
        "panel_agreement": consensus.get("agreement"),
        "panel_n_judges": consensus.get("n_judges"),
        "falsifiability_gate_passed": _falsifiability_gate(idea),
    }


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict] = []
    for i, batch in enumerate(BATCHES, 1):
        try:
            row = await one_batch(batch, i)
            rows.append(row)
        except Exception as e:
            print(f"  batch {i} failed: {e}", file=sys.stderr)
    if not rows:
        raise SystemExit("no math ideas generated")

    out_path = REPO / "artifacts" / "ideas.tsv"
    existing = pd.read_csv(out_path, sep="\t")
    new_df = pd.DataFrame(rows)
    # Reindex new_df to match existing columns where possible
    for col in existing.columns:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[existing.columns.tolist()]
    out = pd.concat([existing, new_df], ignore_index=True)
    out.to_csv(out_path, sep="\t", index=False)
    print(f"\nappended {len(rows)} math ideas → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
