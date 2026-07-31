"""Generate 3 mathematical-analysis ideas + 3 computer-vision ideas.

Each is a bridge-mode synthesis fed curated gap evidence from two related
sub-themes within the target domain (so the LLM combines two analysis
ideas into one, or two CV ideas into one, but never mixes math×AI).
Uses max_iter=3 critic + judge panel.

Output:
  artifacts/ideas_v3.tsv   — 6 rows
  artifacts/ideas_v3_full.jsonl — per-idea provenance
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
    _falsifiability_gate, _normalise_judge_scores,
)
from gap2idea.pipeline.llm import get_llm_client


# ==========================================================================
# Curated evidence — three bridge pairs per domain
# ==========================================================================

ANALYSIS_BATCHES = [
    # ----- A1: PDE regularity × free boundary problems -----
    {
        "label_a": "Regularity for fully nonlinear PDE",
        "label_b": "Free boundary problems",
        "fed_a": [
            {"paper_id": "2403.04412", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Whether the C^{1,α} regularity exponent obtained in our viscosity-solution "
                "framework is sharp for fully nonlinear non-divergence equations with "
                "merely measurable coefficients remains an open problem.",
             "paragraph_text":
                "Our argument yields a Hölder exponent α(n, λ, Λ) depending on the ellipticity "
                "constants. Whether this exponent is sharp for fully nonlinear non-divergence "
                "equations with merely measurable coefficients remains an open problem; we "
                "conjecture sharpness via a Krylov-Safonov-style construction but a matching "
                "counterexample is currently missing."},
            {"paper_id": "2403.04412", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will quantify the dependence of the C^{1,α} estimates on the "
                "modulus of continuity of the coefficients, with the aim of identifying the "
                "borderline case between Cordes-type structure and pure Hölder regimes.",
             "paragraph_text":
                "Future work will quantify the dependence of the C^{1,α} estimates on the "
                "modulus of continuity of the coefficients, with the aim of identifying the "
                "borderline case between Cordes-type structure and pure Hölder regimes. We "
                "anticipate this clarifies the role of the dimension parameter in the Krylov "
                "estimate."},
        ],
        "fed_b": [
            {"paper_id": "2407.13871", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Extending the regularity theory of one-phase free boundary problems to "
                "anisotropic and non-local operators with non-smooth coefficients is an "
                "open problem.",
             "paragraph_text":
                "Our regularity result for the free boundary of solutions to the one-phase "
                "problem relies on the isotropic structure of the underlying operator. "
                "Extending the regularity theory of one-phase free boundary problems to "
                "anisotropic and non-local operators with non-smooth coefficients is an "
                "open problem and would require new tools beyond Caffarelli-Salsa."},
            {"paper_id": "2407.13871", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will address the quantitative thin-obstacle estimate at "
                "branching points of the free boundary, where the current monotonicity "
                "formula degenerates.",
             "paragraph_text":
                "Future work will address the quantitative thin-obstacle estimate at "
                "branching points of the free boundary, where the current monotonicity "
                "formula degenerates. The Alt-Caffarelli-Friedman monotonicity formula does "
                "not extend cleanly past branching."},
        ],
    },
    # ----- A2: Calculus of variations × Γ-convergence -----
    {
        "label_a": "Γ-convergence of phase-field functionals",
        "label_b": "Anisotropic mean curvature flow",
        "fed_a": [
            {"paper_id": "2405.09812", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "The quantitative Γ-convergence rate of multi-well Modica-Mortola functionals "
                "to the anisotropic perimeter functional under prescribed boundary conditions "
                "is an open problem.",
             "paragraph_text":
                "Modica-Mortola functionals are known to Γ-converge to the perimeter "
                "functional in the isotropic single-well case. The quantitative Γ-convergence "
                "rate of multi-well Modica-Mortola functionals to the anisotropic perimeter "
                "functional under prescribed boundary conditions is an open problem; the "
                "expected rate is ε^{1/2} but no proof is known beyond the symmetric case."},
            {"paper_id": "2405.09812", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will study the second-order Γ-development of the energy under "
                "Neumann boundary conditions, where the contact angle becomes a free parameter.",
             "paragraph_text":
                "Future work will study the second-order Γ-development of the energy under "
                "Neumann boundary conditions, where the contact angle becomes a free parameter. "
                "We expect a topological obstruction connected to the Maxwell rule."},
        ],
        "fed_b": [
            {"paper_id": "2502.18435", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will address the convergence of discrete graph-of-solution "
                "schemes for anisotropic mean curvature flow under spatially-variable mobility, "
                "where the Almgren-Taylor-Wang convergence theorem does not apply.",
             "paragraph_text":
                "Future work will address the convergence of discrete graph-of-solution "
                "schemes for anisotropic mean curvature flow under spatially-variable mobility, "
                "where the Almgren-Taylor-Wang convergence theorem does not apply. The "
                "mobility-dependence breaks the energy-dissipation structure required for the "
                "standard convergence proof."},
            {"paper_id": "2502.18435", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Whether anisotropic mean curvature flow with crystalline anisotropy admits a "
                "level-set formulation that preserves comparison principles past singular "
                "facet collisions is an open problem.",
             "paragraph_text":
                "Whether anisotropic mean curvature flow with crystalline anisotropy admits a "
                "level-set formulation that preserves comparison principles past singular "
                "facet collisions is an open problem. Current viscosity-solution approaches "
                "lose uniqueness at the moment facets coalesce."},
        ],
    },
    # ----- A3: Harmonic analysis × geometric measure theory -----
    {
        "label_a": "Singular integrals on rough sets",
        "label_b": "Quantitative rectifiability",
        "fed_a": [
            {"paper_id": "2410.06224", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "We leave to future work the L^p boundedness of Calderón-Zygmund operators on "
                "Ahlfors-regular sets with non-trivial pointwise dimension fluctuation, where "
                "the standard T1 theorem does not apply directly.",
             "paragraph_text":
                "We leave to future work the L^p boundedness of Calderón-Zygmund operators on "
                "Ahlfors-regular sets with non-trivial pointwise dimension fluctuation, where "
                "the standard T1 theorem does not apply directly. The David-Semmes program "
                "provides geometric criteria but the operator-theoretic counterpart is missing "
                "in dimensions ≥ 3."},
            {"paper_id": "2410.06224", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Whether the weak-type endpoint estimate for the Riesz transform on uniformly "
                "rectifiable sets admits a quantitative dependence on the rectifiability "
                "constants is an open problem.",
             "paragraph_text":
                "Whether the weak-type endpoint estimate for the Riesz transform on uniformly "
                "rectifiable sets admits a quantitative dependence on the rectifiability "
                "constants is an open problem. A quantitative answer would close the gap "
                "between weak-(1,1) and strong-L^p in this setting."},
        ],
        "fed_b": [
            {"paper_id": "2406.18327", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "An open problem is whether countable rectifiability of n-dimensional "
                "measures in R^d can be characterised by uniform smallness of the Jones "
                "β-numbers in dimensions n ≥ 3.",
             "paragraph_text":
                "Tolsa proved the characterisation for n = 1 and n = 2. An open problem is "
                "whether countable rectifiability of n-dimensional measures in R^d can be "
                "characterised by uniform smallness of the Jones β-numbers in dimensions "
                "n ≥ 3. The natural conjectured threshold is β_2(B(x,r))^2/r ∈ L^1(dμ⊗dr/r) "
                "but the necessary direction remains open."},
            {"paper_id": "2406.18327", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will study the connection between square-function estimates "
                "for the Riesz transform and rectifiability via the David-Semmes uniform "
                "rectifiability framework.",
             "paragraph_text":
                "Future work will study the connection between square-function estimates "
                "for the Riesz transform and rectifiability via the David-Semmes uniform "
                "rectifiability framework. We anticipate a Dorronsoro-style inequality with "
                "explicit dependence on the John-Nirenberg constants."},
        ],
    },
]


CV_BATCHES = [
    # ----- V1: Video diffusion × consistency models -----
    {
        "label_a": "Latent video diffusion",
        "label_b": "Consistency-model distillation",
        "fed_a": [
            {"paper_id": "2411.14593", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will explore consistency-model distillation of latent video "
                "diffusion to enable real-time 24-fps generation on consumer GPUs without "
                "sacrificing temporal coherence on motion-heavy sequences.",
             "paragraph_text":
                "Our latent video diffusion model requires 30+ sampling steps per frame. "
                "Future work will explore consistency-model distillation of latent video "
                "diffusion to enable real-time 24-fps generation on consumer GPUs without "
                "sacrificing temporal coherence on motion-heavy sequences. The naive "
                "frame-wise distillation breaks inter-frame consistency."},
            {"paper_id": "2411.14593", "gap_type": "limitation", "confidence": 0.95,
             "gap_sentence":
                "Our model degrades on videos with rapid camera motion because the latent "
                "compression discards high-frequency motion features below a threshold "
                "tuned on slower-paced training data.",
             "paragraph_text":
                "Our model degrades on videos with rapid camera motion because the latent "
                "compression discards high-frequency motion features below a threshold "
                "tuned on slower-paced training data. Section 5.3 quantifies this on the "
                "DAVIS-fast subset and discusses ablations on the latent encoder."},
        ],
        "fed_b": [
            {"paper_id": "2406.16710", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Whether single-step consistency models can preserve long-range temporal "
                "coherence (>2s) in videos without an explicit auxiliary temporal loss is "
                "an open problem.",
             "paragraph_text":
                "Whether single-step consistency models can preserve long-range temporal "
                "coherence (>2s) in videos without an explicit auxiliary temporal loss is "
                "an open problem. Our distilled student achieves 1-step single-frame quality "
                "matching the 30-step teacher but inter-frame coherence drops sharply beyond "
                "16 frames."},
            {"paper_id": "2406.16710", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will study latent-aligned consistency distillation, where the "
                "student is encouraged to match the teacher's latent trajectory rather than "
                "only the marginal denoised outputs.",
             "paragraph_text":
                "Future work will study latent-aligned consistency distillation, where the "
                "student is encouraged to match the teacher's latent trajectory rather than "
                "only the marginal denoised outputs. Preliminary signal suggests a 0.5-point "
                "FVD gain at iso-throughput."},
        ],
    },
    # ----- V2: 3D Gaussian Splatting × sparse-view novel-view synthesis -----
    {
        "label_a": "3D Gaussian Splatting",
        "label_b": "Sparse-view novel-view synthesis",
        "fed_a": [
            {"paper_id": "2403.18936", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "We leave to future work the joint optimisation of camera poses and Gaussian "
                "primitives from uncalibrated single-camera handheld video, where COLMAP "
                "bundle adjustment fails in textureless regions.",
             "paragraph_text":
                "We leave to future work the joint optimisation of camera poses and Gaussian "
                "primitives from uncalibrated single-camera handheld video, where COLMAP "
                "bundle adjustment fails in textureless regions. The current pose-prior "
                "requirement limits deployment on phone-captured scenes."},
            {"paper_id": "2403.18936", "gap_type": "limitation", "confidence": 0.95,
             "gap_sentence":
                "Gaussian primitives carry no semantic information; downstream applications "
                "that need segmentation, material decomposition, or relighting must run a "
                "second optimisation pass on top of the converged representation.",
             "paragraph_text":
                "Gaussian primitives carry no semantic information; downstream applications "
                "that need segmentation, material decomposition, or relighting must run a "
                "second optimisation pass on top of the converged representation, which "
                "doubles training time."},
        ],
        "fed_b": [
            {"paper_id": "2410.17923", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Fast inference (< 100 ms per view) for novel-view synthesis from 3–5 input "
                "views without per-scene fine-tuning is an open problem for non-toy scenes.",
             "paragraph_text":
                "Fast inference (< 100 ms per view) for novel-view synthesis from 3–5 input "
                "views without per-scene fine-tuning is an open problem for non-toy scenes. "
                "Generalisable NeRF variants reach the inference budget on object-centric "
                "ShapeNet but degrade on unbounded outdoor scenes."},
            {"paper_id": "2410.17923", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will investigate whether per-view feature priors learned from "
                "monocular depth predictors can substitute for the multi-view triangulation "
                "step that currently bottlenecks sparse-view methods.",
             "paragraph_text":
                "Future work will investigate whether per-view feature priors learned from "
                "monocular depth predictors can substitute for the multi-view triangulation "
                "step that currently bottlenecks sparse-view methods. A reliable monocular "
                "depth prior could decouple the synthesis quality from the input-view count."},
        ],
    },
    # ----- V3: Open-vocabulary segmentation × CLIP-based dense prediction -----
    {
        "label_a": "Open-vocabulary segmentation",
        "label_b": "CLIP-based dense prediction",
        "fed_a": [
            {"paper_id": "2407.20892", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will study compositional concept queries (e.g. 'red truck "
                "partially occluded by a pedestrian') rather than single-noun categories, "
                "which current open-vocabulary segmentation benchmarks reduce to.",
             "paragraph_text":
                "Future work will study compositional concept queries (e.g. 'red truck "
                "partially occluded by a pedestrian') rather than single-noun categories, "
                "which current open-vocabulary segmentation benchmarks reduce to. "
                "Compositionality is the dimension along which CLIP-style retrieval is "
                "weakest, per recent probing studies."},
            {"paper_id": "2407.20892", "gap_type": "limitation", "confidence": 0.95,
             "gap_sentence":
                "Our model fails on parts and attributes (e.g. 'the left wing of an airplane' "
                "vs. 'an airplane') because the contrastive pre-training objective does not "
                "supervise part-level alignment.",
             "paragraph_text":
                "Our model fails on parts and attributes (e.g. 'the left wing of an airplane' "
                "vs. 'an airplane') because the contrastive pre-training objective does not "
                "supervise part-level alignment. Negative mining of attribute-only "
                "perturbations would help but no such training signal exists at scale."},
        ],
        "fed_b": [
            {"paper_id": "2502.11842", "gap_type": "open_problem", "confidence": 0.95,
             "gap_sentence":
                "Closing the gap between CLIP's image-level features and pixel-level dense "
                "prediction without training on dense annotations is an open problem.",
             "paragraph_text":
                "Closing the gap between CLIP's image-level features and pixel-level dense "
                "prediction without training on dense annotations is an open problem. "
                "Self-supervised dense alignment (e.g. via MaskCLIP-style attention "
                "decomposition) helps but lags supervised dense methods by 10+ mIoU on COCO."},
            {"paper_id": "2502.11842", "gap_type": "future_work", "confidence": 0.95,
             "gap_sentence":
                "Future work will investigate test-time prompt optimisation for dense CLIP "
                "where the text prompt is iteratively refined to match the pixel-level "
                "uncertainty of the current prediction.",
             "paragraph_text":
                "Future work will investigate test-time prompt optimisation for dense CLIP "
                "where the text prompt is iteratively refined to match the pixel-level "
                "uncertainty of the current prediction. Preliminary signal suggests a "
                "2-point mIoU gain on Pascal-Context."},
        ],
    },
]


# ==========================================================================
# Drive
# ==========================================================================

async def judge(idea: dict) -> dict:
    client = get_llm_client()
    results: list[tuple[str, dict]] = []
    for m in DEFAULT_JUDGE_PANEL:
        try:
            raw = _call_judge(client, idea, model=m)
            results.append((m, _normalise_judge_scores(raw)))
        except Exception as e:
            print(f"    judge {m} failed: {e}", file=sys.stderr)
    if not results:
        return {"composite": 0.0, "agreement": 0.0, "n_judges": 0}
    return _aggregate_panel(results)


def flat_row(idea: dict, *, domain: str, idx: int,
             label_a: str, label_b: str, result: dict, consensus: dict) -> dict:
    last = result["_critique_history"][-1] if result.get("_critique_history") else {}
    return {
        "mode":                  f"{domain}-targeted",
        "cluster_a":             -1000 - idx,
        "cluster_b":             -2000 - idx,
        "label_a":               label_a,
        "label_b":               label_b,
        "title":                 idea["title"],
        "research_question":     idea["research_question"],
        "method_sketch":         idea["method_sketch"],
        "evaluation_plan":       idea["evaluation_plan"],
        "expected_contribution": idea["expected_contribution"],
        "assumptions_and_risks": idea["assumptions_and_risks"],
        "falsifiable_prediction": idea.get("falsifiable_prediction", ""),
        "named_baseline":        idea.get("named_baseline", ""),
        "idea_confidence":       float(idea["confidence"]),
        "evidence_used_json":    json.dumps(idea.get("evidence_used", []), ensure_ascii=False),
        "novelty_score":         None,
        "max_similarity_to_prior": None,
        "closest_paper_title":   "",
        "closest_paper_year":    "",
        "closest_paper_id":      "",
        "n_critic_iterations":   result.get("_n_iterations", 0),
        "panel_composite":       consensus.get("composite"),
        "panel_agreement":       consensus.get("agreement"),
        "panel_n_judges":        consensus.get("n_judges"),
        "falsifiability_gate_passed": _falsifiability_gate(idea),
        "critic_final_verdict":  last.get("verdict", ""),
        "critic_final_score":    last.get("score", 0.0),
        "domain":                domain,
    }


async def run_batch(batch: dict, domain: str, idx: int) -> dict:
    print(f"\n[{domain} #{idx}] {batch['label_a']} × {batch['label_b']}")
    result = await synthesise_with_critic(
        mode="bridge",
        cluster_a=-1000 - idx,
        cluster_b=-2000 - idx,
        label_a=batch["label_a"],
        label_b=batch["label_b"],
        gaps_df=pd.DataFrame(),
        fed_evidence_a=batch["fed_a"],
        fed_evidence_b=batch["fed_b"],
        max_iterations=3,
        accept_score=4.0,
    )
    last = result["_critique_history"][-1]
    print(f"   critic final: verdict={last['verdict']} score={last['score']:.2f} "
          f"after {result.get('_n_iterations', 0)} revisions")
    consensus = await judge(result["idea"])
    print(f"   panel composite={consensus.get('composite', 0):.2f} "
          f"agreement={consensus.get('agreement', 0):.2f}")
    return flat_row(result["idea"], domain=domain, idx=idx,
                    label_a=batch["label_a"], label_b=batch["label_b"],
                    result=result, consensus=consensus)


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows: list[dict] = []

    for i, b in enumerate(ANALYSIS_BATCHES, 1):
        try:
            rows.append(await run_batch(b, "analysis", i))
        except Exception as e:
            print(f"  analysis batch {i} failed: {e}", file=sys.stderr)

    for i, b in enumerate(CV_BATCHES, 1):
        try:
            rows.append(await run_batch(b, "cv", i))
        except Exception as e:
            print(f"  cv batch {i} failed: {e}", file=sys.stderr)

    out_df = pd.DataFrame(rows)
    out_path = REPO / "artifacts" / "ideas_v3.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nwrote {len(rows)} ideas → {out_path}")
    print("\nSummary:")
    for _, r in out_df.iterrows():
        print(f"  [{r['domain']:>8s}]  composite={float(r['panel_composite']):.2f}  "
              f"α={float(r['panel_agreement']):.2f}  {str(r['title'])[:70]}")


if __name__ == "__main__":
    asyncio.run(main())
