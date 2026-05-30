"""Multi-agent layer.

Three agent classes, all LLM-driven via OpenRouter, all sharing the tool
surface defined in `gap2idea.tools`:

  • Synthesiser   drafts an idea from theme evidence (+ optional methods)
  • Critic        critiques a draft using tools (novelty + evidence overlap)
                  and proposes specific revisions
  • JudgePanel    scores the final idea using N independent models

The agents do not know whether their tool calls go via the MCP server or
in-process — they always use `tools.dispatch`. This keeps unit tests
hermetic (no MCP server needed) while leaving the door open for an
externally-orchestrated agent (e.g. Claude Desktop driving the same tools).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import numpy as np
import pandas as pd
from openai import OpenAI

from gap2idea import tools as T
from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client, parse_json_response
from gap2idea.pipeline.openai_ideas import (
    IDEA_SCHEMA,
    SYSTEM_IDEA,
    SYSTEM_METHOD_GAP,
    SYSTEM_WITHIN,
    _build_method_gap_prompt,
    _build_user_prompt,
    _build_within_prompt,
    _diverse_evidence,
    _evidence_payload,
)
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)


# ===========================================================================
# Critic
# ===========================================================================

CRITIC_SYSTEM = (
    "You are a strict but constructive research-idea critic. "
    "Given a draft idea, its source evidence, and tool-derived diagnostics "
    "(novelty score, evidence-overlap), identify the SPECIFIC weaknesses "
    "and propose concrete revisions a synthesiser agent should make. "
    "Be honest: if the idea is already solid, say so and request no revision.\n\n"
    "Treat the following as SPECIFICITY issues that require revision: "
    "(a) `falsifiable_prediction` missing, vague, or lacking a quantitative "
    "threshold; (b) `named_baseline` missing, 'TBD', 'none', 'no baseline', "
    "or otherwise unnamed."
)

CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["accept", "revise", "reject"]},
        "score": {"type": "number", "minimum": 1, "maximum": 5},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "axis": {
                        "type": "string",
                        "enum": ["novelty", "specificity", "feasibility",
                                 "evidence_grounding", "scope"],
                    },
                    "problem": {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["axis", "problem", "suggested_fix"],
                "additionalProperties": False,
            },
        },
        "revision_directive": {
            "type": "string",
            "description": "One- or two-sentence instruction for the synthesiser. "
                           "Empty string if verdict is 'accept' or 'reject'.",
        },
    },
    "required": ["verdict", "score", "issues", "revision_directive"],
    "additionalProperties": False,
}


@retry(tries=3, base_delay=2.0)
def _call_critic(client: OpenAI, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "idea_critique", "schema": CRITIC_SCHEMA, "strict": True},
        },
        temperature=0.0,
    )
    return parse_json_response(resp.choices[0].message.content)


async def critique_idea(
    idea: dict,
    fed_evidence: list[dict],
    *,
    model: str = "anthropic/claude-sonnet-4",
    novelty_payload: dict | None = None,
    overlap_payload: dict | None = None,
) -> dict:
    """Critique a drafted idea. Calls the novelty + overlap tools if not provided.

    Cross-provider default (Claude) — the critic should disagree productively
    with a generator that defaults to GPT-4.1-mini.
    """
    if novelty_payload is None:
        novelty_payload = await T.score_novelty(
            title=idea.get("title", ""),
            research_question=idea.get("research_question", ""),
            method_sketch=idea.get("method_sketch", ""),
            top_k=8,
        )
    if overlap_payload is None:
        overlap_payload = await T.check_evidence_overlap(
            evidence_used=idea.get("evidence_used", []),
            fed_evidence=fed_evidence,
        )

    payload = {
        "idea": idea,
        "diagnostics": {
            "novelty_score": novelty_payload.get("novelty_score"),
            "max_similarity_to_prior": novelty_payload.get("max_similarity"),
            "closest_prior_work": novelty_payload.get("closest_paper"),
            "evidence_overlap": overlap_payload.get("overlap"),
            "n_hallucinated_evidence": len(overlap_payload.get("hallucinated", [])),
            "hallucinated_evidence": overlap_payload.get("hallucinated", [])[:3],
        },
    }
    prompt = (
        "Review the proposed research idea using the included diagnostics.\n"
        "If `evidence_overlap` is below 0.8, treat that as evidence_grounding "
        "issue and require revision. If `novelty_score` is below 0.4, treat "
        "as novelty issue. If the method_sketch lacks a named method or the "
        "evaluation_plan lacks a metric + baseline, treat as specificity issue.\n\n"
        "Return JSON matching the schema.\n\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False, default=str)
    )
    client = get_llm_client()
    result = _call_critic(client, prompt, model=model)
    # Defensive normalisation: non-OpenAI critics (Claude/Gemini via OpenRouter)
    # don't always honour `strict=True` on `response_format=json_schema`, so the
    # returned dict can be missing required CRITIC_SCHEMA fields. Default
    # conservatively rather than crashing the orchestrated run.
    if not isinstance(result, dict):
        result = {}
    verdict = str(result.get("verdict", "")).strip().lower()
    if verdict not in ("accept", "revise", "reject"):
        verdict = "revise"  # safe default: triggers another revision pass
    result["verdict"] = verdict
    try:
        result["score"] = float(result.get("score", 3.0))
    except (TypeError, ValueError):
        result["score"] = 3.0
    result["issues"] = list(result.get("issues") or [])
    result["revision_directive"] = str(result.get("revision_directive") or "")
    result["_diagnostics"] = payload["diagnostics"]  # surface for caller logging
    return result


# ===========================================================================
# Synthesiser with revision support
# ===========================================================================

REVISION_SYSTEM = (
    "You are a research planning assistant revising a draft idea based on "
    "a critic's feedback. Address each numbered issue. Stay strictly within "
    "the provided evidence. Return the revised idea as JSON matching the schema.\n\n"
    "Preserve a quantitative `falsifiable_prediction` (number/%/inequality) "
    "and a specific `named_baseline` (a real prior method or paper). If the "
    "prior draft had these and you weaken them in revision, you have failed. "
    "Never replace a named baseline with 'TBD', 'none', or 'no baseline'."
)


@retry(tries=3, base_delay=2.0)
def _call_revisor(client: OpenAI, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVISION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "idea_revision", "schema": IDEA_SCHEMA, "strict": True},
        },
        temperature=0.5,
    )
    return parse_json_response(resp.choices[0].message.content)


async def revise_idea(
    idea: dict,
    critique: dict,
    fed_evidence_a: list[dict],
    fed_evidence_b: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Apply a critique to produce a revised idea. Returns the new flat idea dict."""
    payload = {
        "prior_draft": idea,
        "fed_evidence_a": fed_evidence_a,
        "fed_evidence_b": fed_evidence_b or [],
        "critique": {
            "verdict": critique["verdict"],
            "score": critique["score"],
            "issues": critique["issues"],
            "revision_directive": critique["revision_directive"],
        },
    }
    prompt = (
        "Revise the prior draft to address every item in `critique.issues`.\n"
        "Constraints:\n"
        "- Use ONLY content from fed_evidence_a and fed_evidence_b.\n"
        "- evidence_used must subset the fed evidence (paper_id + verbatim sentence).\n"
        "- Increase specificity where the critic flagged it.\n"
        "- Do NOT silently weaken existing strengths.\n\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False, default=str)
    )
    client = get_llm_client()
    data = _call_revisor(client, prompt, model=model)
    revised = data["idea"] if "idea" in data else data

    # Defensive field copy-through: non-OpenAI providers (Claude/Gemini)
    # sometimes silently drop schema fields they didn't choose to populate.
    # If the prior draft had a value and the revised one doesn't, restore
    # the prior value rather than losing it. This is critical for the
    # PR-2 falsifiable_prediction + named_baseline fields, which the
    # critic depends on.
    if isinstance(revised, dict) and isinstance(idea, dict):
        for key, prior_val in idea.items():
            if key not in revised and prior_val not in (None, "", [], {}):
                revised[key] = prior_val
    return revised


# ===========================================================================
# Critique-revise loop (Phase 2)
# ===========================================================================

async def synthesise_with_critic(
    *,
    mode: str,                       # "bridge" | "within" | "method-gap"
    cluster_a: int,
    cluster_b: int | None = None,
    label_a: str = "",
    label_b: str = "",
    gaps_df: pd.DataFrame,
    fed_evidence_a: list[dict],
    fed_evidence_b: list[dict] | None = None,
    candidate_methods: list[dict] | None = None,
    model: str = DEFAULT_MODEL,
    critic_model: str = "anthropic/claude-sonnet-4",
    max_iterations: int = 2,
    accept_score: float = 4.0,
) -> dict:
    """Generate idea, critique, revise up to `max_iterations` times.

    Returns the final flat idea dict with two extra fields:
      _critique_history : list of every critic verdict + score
      _n_iterations     : how many revisions were applied (0 = first draft kept)
    """
    client = get_llm_client()

    # ---- initial draft ----
    if mode == "bridge":
        prompt = _build_user_prompt(
            cluster_a, cluster_b, label_a, label_b, fed_evidence_a, fed_evidence_b or [],
        )
        system = SYSTEM_IDEA
    elif mode == "within":
        prompt = _build_within_prompt(cluster_a, label_a, fed_evidence_a)
        system = SYSTEM_WITHIN
    elif mode == "method-gap":
        prompt = _build_method_gap_prompt(
            cluster_a, label_a, fed_evidence_a, candidate_methods or [],
        )
        system = SYSTEM_METHOD_GAP
    else:
        raise ValueError(f"unknown mode: {mode}")

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "idea_synthesis", "schema": IDEA_SCHEMA, "strict": True},
        },
        temperature=0.7,
    )
    data = parse_json_response(resp.choices[0].message.content)
    idea = data["idea"]

    # ---- critique / revise loop ----
    history = []
    fed_all = fed_evidence_a + (fed_evidence_b or [])
    n_iter = 0
    for i in range(max_iterations):
        critique = await critique_idea(
            idea, fed_all, model=critic_model,
        )
        history.append({
            "iteration": i,
            "verdict": critique["verdict"],
            "score": critique["score"],
            "issues_count": len(critique["issues"]),
            "directive": critique["revision_directive"],
            "diagnostics": critique["_diagnostics"],
        })
        log.info(
            "  critic [iter %d/%d] verdict=%s score=%.2f issues=%d",
            i + 1, max_iterations, critique["verdict"], critique["score"], len(critique["issues"]),
        )

        if critique["verdict"] == "accept" or critique["score"] >= accept_score:
            break
        if critique["verdict"] == "reject" and i == max_iterations - 1:
            break

        idea = await revise_idea(idea, critique, fed_evidence_a, fed_evidence_b, model=model)
        n_iter += 1

    return {
        "idea": idea,
        "_critique_history": history,
        "_n_iterations": n_iter,
    }


# ===========================================================================
# Helper: build evidence + invoke synthesise_with_critic from a corpus
# ===========================================================================

async def run_critic_pipeline_for_pair(
    gaps: pd.DataFrame,
    cluster_a: int,
    cluster_b: int,
    label_a: str,
    label_b: str,
    *,
    k_evidence: int = 4,
    model: str = DEFAULT_MODEL,
    critic_model: str = "anthropic/claude-sonnet-4",
    max_iterations: int = 2,
) -> dict:
    ev_a = _evidence_payload(_diverse_evidence(gaps, cluster_a, k=k_evidence))
    ev_b = _evidence_payload(_diverse_evidence(gaps, cluster_b, k=k_evidence))
    return await synthesise_with_critic(
        mode="bridge", cluster_a=cluster_a, cluster_b=cluster_b,
        label_a=label_a, label_b=label_b,
        gaps_df=gaps, fed_evidence_a=ev_a, fed_evidence_b=ev_b,
        model=model, critic_model=critic_model, max_iterations=max_iterations,
    )


async def run_critic_pipeline_within(
    gaps: pd.DataFrame,
    cluster_id: int,
    label: str,
    *,
    k_evidence: int = 8,
    model: str = DEFAULT_MODEL,
    critic_model: str = "anthropic/claude-sonnet-4",
    max_iterations: int = 2,
) -> dict:
    ev = _evidence_payload(_diverse_evidence(gaps, cluster_id, k=k_evidence))
    return await synthesise_with_critic(
        mode="within", cluster_a=cluster_id, label_a=label,
        gaps_df=gaps, fed_evidence_a=ev,
        model=model, critic_model=critic_model, max_iterations=max_iterations,
    )


async def run_critic_pipeline_frontier(
    gaps: pd.DataFrame,
    embeddings: np.ndarray,
    seed_idx: int,
    cluster_id: int,
    label: str,
    *,
    k_evidence: int = 5,
    model: str = DEFAULT_MODEL,
    critic_model: str = "anthropic/claude-sonnet-4",
    max_iterations: int = 2,
) -> dict:
    """Frontier-mode wrapper: pick the seed frontier gap plus its k-1 nearest
    neighbours by cosine, then run the within-mode critic loop on the joined
    evidence. Cross-community structure falls out for free because the seed
    sits at a community boundary."""
    from gap2idea.pipeline.openai_ideas import _frontier_evidence

    ev_df = _frontier_evidence(gaps, embeddings, seed_idx, k_evidence=k_evidence)
    if ev_df.empty:
        raise ValueError(f"No evidence for frontier gap {seed_idx}")
    ev = _evidence_payload(ev_df)
    return await synthesise_with_critic(
        mode="within", cluster_a=cluster_id, label_a=label or "frontier gap",
        gaps_df=gaps, fed_evidence_a=ev,
        model=model, critic_model=critic_model, max_iterations=max_iterations,
    )


async def run_critic_pipeline_method_gap(
    gaps: pd.DataFrame,
    cluster_id: int,
    label: str,
    gap_embeddings: np.ndarray,
    methods: pd.DataFrame,
    method_embeddings: np.ndarray,
    *,
    k_evidence: int = 4,
    k_methods: int = 5,
    sim_low: float = 0.30,
    sim_high: float = 0.70,
    model: str = DEFAULT_MODEL,
    critic_model: str = "anthropic/claude-sonnet-4",
    max_iterations: int = 2,
) -> dict:
    from gap2idea.pipeline.openai_ideas import _retrieve_methods_for_cluster

    ev = _evidence_payload(_diverse_evidence(gaps, cluster_id, k=k_evidence))
    mdf = _retrieve_methods_for_cluster(
        cluster_id, gap_embeddings, gaps, methods, method_embeddings,
        top_k=k_methods, sim_low=sim_low, sim_high=sim_high,
    )
    if mdf.empty:
        raise ValueError(f"No methods in sweet spot [{sim_low}, {sim_high}] for cluster {cluster_id}")
    candidate_methods = [
        {
            "paper_id": str(r["id"]),
            "method_type": str(r.get("method_type", "")),
            "method_sentence": str(r["method_sentence"]),
            "paragraph_text": str(r.get("paragraph_text", "")),
            "similarity_to_gap_cluster": float(r["_sim_to_gap_cluster"]),
        }
        for _, r in mdf.iterrows()
    ]
    return await synthesise_with_critic(
        mode="method-gap", cluster_a=cluster_id, label_a=label,
        gaps_df=gaps, fed_evidence_a=ev, candidate_methods=candidate_methods,
        model=model, critic_model=critic_model, max_iterations=max_iterations,
    )
