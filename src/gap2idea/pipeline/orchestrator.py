"""Phase 4: end-to-end multi-agent idea pipeline.

Wires together every agent class:
  Synthesiser -> Critic -> (revise loop) -> JudgePanel -> persist

Per cluster (or pair, or method-gap match) we produce:
  • a final idea
  • a critique history (every revision step)
  • a panel of judge scores with inter-judge agreement
  • the standard S2 novelty payload

All artifacts go to:
  artifacts/ideas.tsv          (flat, one row per idea, has `mode` column)
  artifacts/ideas_full.jsonl   (rich provenance: prompt, critiques, panel)
  artifacts/idea_eval.tsv      (panel-judged scores)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gap2idea.pipeline.agents import (
    run_critic_pipeline_for_pair,
    run_critic_pipeline_method_gap,
    run_critic_pipeline_within,
)
from gap2idea.pipeline.evaluation import (
    DEFAULT_JUDGE_PANEL,
    _aggregate_panel,
    _call_judge,
    _normalise_judge_scores,
)
from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client
from gap2idea.pipeline.openai_ideas import (
    _diverse_evidence,
    _evidence_payload,
    _retrieve_methods_for_cluster,
    novelty_check,
)
from gap2idea.utils import get_logger

log = get_logger(__name__)


def _idea_to_flat_row(
    result: dict,
    *,
    mode: str,
    cluster_a: int,
    cluster_b: int | None,
    label_a: str,
    label_b: str,
    novelty_payload: dict,
    panel_consensus: dict,
) -> dict:
    """Flatten the multi-agent result into the unified `ideas.tsv` schema."""
    idea = result["idea"]
    closest = (novelty_payload.get("closest_paper") or {})
    return {
        "mode": mode,
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "label_a": label_a,
        "label_b": label_b,
        "title": idea["title"],
        "research_question": idea["research_question"],
        "method_sketch": idea["method_sketch"],
        "evaluation_plan": idea["evaluation_plan"],
        "expected_contribution": idea["expected_contribution"],
        "assumptions_and_risks": idea["assumptions_and_risks"],
        "falsifiable_prediction": idea.get("falsifiable_prediction", ""),
        "named_baseline": idea.get("named_baseline", ""),
        "idea_confidence": float(idea["confidence"]),
        "evidence_used_json": json.dumps(idea["evidence_used"], ensure_ascii=False),
        "novelty_score": novelty_payload.get("novelty_score"),
        "max_similarity_to_prior": novelty_payload.get("max_similarity"),
        "closest_paper_title": closest.get("title", ""),
        "closest_paper_year": closest.get("year", ""),
        "closest_paper_id": closest.get("paperId", ""),
        # Multi-agent extras
        "n_critic_iterations": result.get("_n_iterations", 0),
        "panel_composite": panel_consensus.get("composite"),
        "panel_agreement": panel_consensus.get("agreement"),
        "panel_n_judges": panel_consensus.get("n_judges"),
    }


async def _judge_idea_with_panel(idea: dict, judge_models: list[str]) -> tuple[dict, list[tuple[str, dict]]]:
    """Run a panel of judges on one idea. Returns (consensus, [(model, scores), ...])."""
    client = get_llm_client()
    results: list[tuple[str, dict]] = []
    for judge in judge_models:
        try:
            raw = _call_judge(client, idea, model=judge)
            scores = _normalise_judge_scores(raw)
            results.append((judge, scores))
        except Exception as e:
            log.warning("Judge %s failed: %s", judge, e)
    if not results:
        return {"composite": 0.0, "agreement": 0.0, "n_judges": 0}, []
    return _aggregate_panel(results), results


async def orchestrate_one(
    *,
    mode: str,
    gaps: pd.DataFrame,
    cluster_a: int,
    cluster_b: int | None,
    label_a: str,
    label_b: str,
    gap_embeddings: np.ndarray | None = None,
    methods: pd.DataFrame | None = None,
    method_embeddings: np.ndarray | None = None,
    model: str = DEFAULT_MODEL,
    critic_model: str = "anthropic/claude-sonnet-4",
    judge_panel: list[str] | None = None,
    max_critic_iterations: int = 2,
    k_evidence: int = 4,
    k_methods: int = 5,
    sim_low: float = 0.30,
    sim_high: float = 0.70,
    check_novelty: bool = True,
) -> dict:
    """End-to-end orchestration for ONE idea. Returns {flat, full} dicts."""
    judge_panel = judge_panel or DEFAULT_JUDGE_PANEL

    # ---- Synthesise with critic loop ----
    if mode == "bridge":
        assert cluster_b is not None, "bridge mode needs cluster_b"
        result = await run_critic_pipeline_for_pair(
            gaps, cluster_a, cluster_b, label_a, label_b,
            k_evidence=k_evidence, model=model, critic_model=critic_model,
            max_iterations=max_critic_iterations,
        )
    elif mode == "within":
        result = await run_critic_pipeline_within(
            gaps, cluster_a, label_a,
            k_evidence=max(k_evidence, 6), model=model, critic_model=critic_model,
            max_iterations=max_critic_iterations,
        )
    elif mode == "method-gap":
        assert gap_embeddings is not None and methods is not None and method_embeddings is not None
        result = await run_critic_pipeline_method_gap(
            gaps, cluster_a, label_a, gap_embeddings, methods, method_embeddings,
            k_evidence=k_evidence, k_methods=k_methods,
            sim_low=sim_low, sim_high=sim_high,
            model=model, critic_model=critic_model,
            max_iterations=max_critic_iterations,
        )
    else:
        raise ValueError(f"unknown mode: {mode}")

    idea = result["idea"]

    # ---- Novelty check ----
    nov_payload: dict = {}
    if check_novelty:
        try:
            from gap2idea.pipeline.semantic_scholar import S2Client
            from sentence_transformers import SentenceTransformer
            emb = SentenceTransformer("all-MiniLM-L6-v2")
            nov_payload = novelty_check(idea, S2Client(), emb, top_k=10)
        except Exception as e:
            log.warning("Novelty check failed: %s", e)

    # ---- Judge panel ----
    consensus, panel = await _judge_idea_with_panel(idea, judge_panel)

    flat = _idea_to_flat_row(
        result, mode=mode, cluster_a=cluster_a, cluster_b=cluster_b,
        label_a=label_a, label_b=label_b,
        novelty_payload=nov_payload, panel_consensus=consensus,
    )

    full = {
        "mode": mode,
        "pair": {
            "cluster_a": cluster_a, "cluster_b": cluster_b,
            "label_a": label_a, "label_b": label_b,
        },
        "idea": idea,
        "novelty": nov_payload,
        "critique_history": result.get("_critique_history", []),
        "n_critic_iterations": result.get("_n_iterations", 0),
        "panel_consensus": consensus,
        "panel_per_judge": [
            {"model": m, "scores": s} for m, s in panel
        ],
        "model": model,
        "critic_model": critic_model,
        "judge_panel": judge_panel,
    }
    return {"flat": flat, "full": full}


async def orchestrate_batch(
    *,
    gaps: pd.DataFrame,
    cluster_labels: pd.DataFrame,
    out_tsv: Path,
    out_jsonl: Path,
    mode: str = "within",                   # bridge | within | method-gap
    pairs: pd.DataFrame | None = None,      # required for bridge mode
    gap_embeddings: np.ndarray | None = None,
    methods: pd.DataFrame | None = None,
    method_embeddings: np.ndarray | None = None,
    n: int | None = 5,
    model: str = DEFAULT_MODEL,
    critic_model: str = "anthropic/claude-sonnet-4",
    judge_panel: list[str] | None = None,
    max_critic_iterations: int = 2,
    k_evidence: int = 4,
    k_methods: int = 5,
    sim_low: float = 0.30,
    sim_high: float = 0.70,
    check_novelty: bool = True,
) -> pd.DataFrame:
    """Run orchestrate_one over the top-N candidates of the chosen mode."""
    label_map = dict(zip(cluster_labels["cluster_id"].astype(int), cluster_labels["theme_label"]))

    if mode == "bridge":
        assert pairs is not None and not pairs.empty, "bridge mode requires non-empty pairs"
        ordered = pairs.sort_values("bridge_score", ascending=False).head(n or len(pairs))
        candidates = [
            (int(r.cluster_a), int(r.cluster_b),
             label_map.get(int(r.cluster_a), ""), label_map.get(int(r.cluster_b), ""))
            for r in ordered.itertuples(index=False)
        ]
    else:
        cids = sorted(c for c in gaps["cluster_id"].unique() if c != -1)
        sizes = gaps[gaps["cluster_id"].isin(cids)].groupby("cluster_id").size().sort_values(ascending=False)
        cids = sizes.head(n or len(cids)).index.tolist()
        candidates = [(int(c), None, label_map.get(int(c), ""), "") for c in cids]

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    flat_rows: list[dict] = []
    with open(out_jsonl, "w", encoding="utf-8") as jf:
        for i, (ca, cb, la, lb) in enumerate(candidates, 1):
            log.info("[%d/%d] orchestrating mode=%s cluster_a=%s cluster_b=%s",
                     i, len(candidates), mode, ca, cb)
            try:
                out = await orchestrate_one(
                    mode=mode, gaps=gaps, cluster_a=ca, cluster_b=cb,
                    label_a=la, label_b=lb,
                    gap_embeddings=gap_embeddings, methods=methods,
                    method_embeddings=method_embeddings,
                    model=model, critic_model=critic_model, judge_panel=judge_panel,
                    max_critic_iterations=max_critic_iterations,
                    k_evidence=k_evidence, k_methods=k_methods,
                    sim_low=sim_low, sim_high=sim_high,
                    check_novelty=check_novelty,
                )
            except Exception as e:
                log.error("Orchestration failed for cluster %s: %s", ca, e)
                continue
            flat_rows.append(out["flat"])
            jf.write(json.dumps(out["full"], ensure_ascii=False, default=str) + "\n")
            log.info(
                "  -> '%s'  iters=%d  panel=%.2f  agreement=%.2f  nov=%s",
                out["flat"]["title"][:50],
                out["flat"]["n_critic_iterations"],
                out["flat"].get("panel_composite") or 0.0,
                out["flat"].get("panel_agreement") or 0.0,
                f"{out['flat'].get('novelty_score'):.2f}" if out['flat'].get('novelty_score') is not None else "n/a",
            )

    df = pd.DataFrame(flat_rows)
    df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote %d orchestrated ideas to %s", len(df), out_tsv)
    return df
