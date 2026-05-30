"""Unit tests for the agentic layer (critic, revisor, multi-agent loop).

All LLM calls are mocked. We're testing orchestration logic, not model
output quality.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, AsyncMock

import numpy as np
import pandas as pd
import pytest

from gap2idea.pipeline import agents


# ---------- fixtures ----------

@pytest.fixture
def fake_gaps():
    rows = []
    for i, pid in enumerate(["p1", "p2", "p3", "p4"]):
        rows.append({"id": pid, "gap_sentence": f"gap on topic A {i} " * 3,
                     "paragraph_text": "para",  "confidence": 0.9 - i * 0.02,
                     "gap_type": "future_work", "cluster_id": 0})
    for i, pid in enumerate(["q1", "q2", "q3"]):
        rows.append({"id": pid, "gap_sentence": f"gap on topic B {i} " * 3,
                     "paragraph_text": "para", "confidence": 0.95 - i * 0.02,
                     "gap_type": "limitation", "cluster_id": 1})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_idea_response():
    """A minimal idea-schema response from the synthesiser."""
    return {
        "pair": {"cluster_a": 0, "cluster_b": 1},
        "idea": {
            "title": "Bridging A and B",
            "research_question": "Can A inform B?",
            "method_sketch": "Train on A; transfer to B; eval metric M vs baseline R.",
            "evaluation_plan": "Compare against baseline R on M.",
            "expected_contribution": "Cross-domain insight.",
            "assumptions_and_risks": "Risk: domain shift.",
            "evidence_used": [
                {"paper_id": "p1", "gap_sentence": "gap on topic A 0 gap on topic A 0 gap on topic A 0 "},
                {"paper_id": "q1", "gap_sentence": "gap on topic B 0 gap on topic B 0 gap on topic B 0 "},
            ],
            "confidence": 0.75,
        },
    }


@pytest.fixture
def fake_critique_accept():
    return {
        "verdict": "accept",
        "score": 4.5,
        "issues": [],
        "revision_directive": "",
    }


@pytest.fixture
def fake_critique_revise():
    return {
        "verdict": "revise",
        "score": 2.5,
        "issues": [
            {"axis": "specificity", "problem": "metric not named", "suggested_fix": "name a metric"},
        ],
        "revision_directive": "Name a specific evaluation metric and a comparison baseline.",
    }


# ---------- critique_idea: calls the tool surface for diagnostics ----------

@pytest.mark.asyncio
async def test_critique_idea_uses_diagnostics(fake_critique_accept):
    idea = {
        "title": "Test", "research_question": "?",
        "method_sketch": "...", "evidence_used": [],
    }
    fed = [{"paper_id": "p1", "gap_sentence": "abc"}]

    fake_nov = {"novelty_score": 0.8, "max_similarity": 0.2, "closest_paper": None}
    fake_overlap = {"overlap": 1.0, "n_used": 0, "n_grounded": 0, "hallucinated": []}

    with patch.object(agents, "_call_critic", return_value=fake_critique_accept), \
         patch.object(agents, "get_llm_client", return_value=object()):
        out = await agents.critique_idea(
            idea, fed,
            novelty_payload=fake_nov,
            overlap_payload=fake_overlap,
        )
    assert out["verdict"] == "accept"
    assert out["score"] == 4.5
    # Diagnostics are surfaced for caller logging
    assert "_diagnostics" in out
    assert out["_diagnostics"]["novelty_score"] == 0.8


@pytest.mark.asyncio
async def test_critique_idea_calls_tools_when_payloads_missing(fake_critique_revise):
    """If no payloads provided, the function should call tools.score_novelty
    and tools.check_evidence_overlap."""
    from gap2idea import tools as T

    nov_stub = AsyncMock(return_value={"novelty_score": 0.6, "max_similarity": 0.4, "closest_paper": None})
    overlap_stub = AsyncMock(return_value={"overlap": 0.5, "hallucinated": [{"paper_id": "x", "gap_sentence": "y"}]})

    with patch.object(T, "score_novelty", nov_stub), \
         patch.object(T, "check_evidence_overlap", overlap_stub), \
         patch.object(agents, "_call_critic", return_value=fake_critique_revise), \
         patch.object(agents, "get_llm_client", return_value=object()):
        await agents.critique_idea(
            {"title": "t", "research_question": "?", "method_sketch": "...", "evidence_used": []},
            [],
        )

    nov_stub.assert_called_once()
    overlap_stub.assert_called_once()


# ---------- revise_idea ----------

@pytest.mark.asyncio
async def test_revise_idea_returns_idea_payload(fake_idea_response, fake_critique_revise):
    with patch.object(agents, "_call_revisor", return_value=fake_idea_response), \
         patch.object(agents, "get_llm_client", return_value=object()):
        revised = await agents.revise_idea(
            fake_idea_response["idea"], fake_critique_revise,
            [{"paper_id": "p1", "gap_sentence": "g"}],
        )
    assert revised["title"] == "Bridging A and B"


@pytest.mark.asyncio
async def test_revise_idea_defensive_field_copy_through(fake_critique_revise):
    """PR-2 regression: non-OpenAI providers (Claude/Gemini) sometimes drop
    schema fields silently. If the prior draft had falsifiable_prediction +
    named_baseline and the revisor's response lacks them, revise_idea must
    restore the prior values rather than losing them."""
    prior = {
        "title": "Prior title",
        "research_question": "RQ?",
        "method_sketch": "method",
        "evaluation_plan": "evaluate vs B",
        "expected_contribution": "C",
        "assumptions_and_risks": "R",
        "falsifiable_prediction": "We expect 5pp gain over baseline on dev.",
        "named_baseline": "RoBERTa-base",
        "evidence_used": [],
        "confidence": 0.7,
    }
    # The revisor drops the two new fields silently (simulating Claude/Gemini).
    stripped = {
        "pair": {"cluster_a": 0, "cluster_b": 1},
        "idea": {
            "title": "Revised title",
            "research_question": "RQ-revised?",
            "method_sketch": "method-revised",
            "evaluation_plan": "evaluate vs B (revised)",
            "expected_contribution": "C-revised",
            "assumptions_and_risks": "R-revised",
            # NOTE: falsifiable_prediction and named_baseline INTENTIONALLY missing
            "evidence_used": [],
            "confidence": 0.7,
        },
    }
    with patch.object(agents, "_call_revisor", return_value=stripped), \
         patch.object(agents, "get_llm_client", return_value=object()):
        revised = await agents.revise_idea(
            prior, fake_critique_revise,
            [{"paper_id": "p1", "gap_sentence": "g"}],
        )
    assert revised["title"] == "Revised title"  # revision did apply
    # Defensive copy-through restored the dropped fields:
    assert revised["falsifiable_prediction"] == "We expect 5pp gain over baseline on dev."
    assert revised["named_baseline"] == "RoBERTa-base"


# ---------- critique-revise loop ----------

@pytest.mark.asyncio
async def test_synthesise_with_critic_short_circuits_on_accept(
    fake_gaps, fake_idea_response, fake_critique_accept,
):
    """Critic returns 'accept' on first iteration -> no revision applied."""
    from gap2idea.pipeline.openai_ideas import _evidence_payload, _diverse_evidence

    ev_a = _evidence_payload(_diverse_evidence(fake_gaps, 0, k=2))
    ev_b = _evidence_payload(_diverse_evidence(fake_gaps, 1, k=2))

    mock_client = type("C", (), {})()
    # Outer call (initial draft) goes through client.chat.completions.create
    class FakeChat:
        class completions:
            @staticmethod
            def create(**kwargs):
                class R:
                    class _Choice:
                        class message:
                            content = '{"pair": {"cluster_a": 0, "cluster_b": 1}, "idea": ' + \
                                pd_to_json(fake_idea_response["idea"]) + '}'
                    choices = [_Choice()]
                return R()
    mock_client.chat = FakeChat

    async def fake_critique(*a, **kw):
        return {**fake_critique_accept, "_diagnostics": {"novelty_score": 0.8}}

    with patch.object(agents, "get_llm_client", return_value=mock_client), \
         patch.object(agents, "critique_idea", new=fake_critique):
        result = await agents.synthesise_with_critic(
            mode="bridge", cluster_a=0, cluster_b=1,
            label_a="A", label_b="B",
            gaps_df=fake_gaps, fed_evidence_a=ev_a, fed_evidence_b=ev_b,
            max_iterations=2, accept_score=4.0,
        )
    assert result["_n_iterations"] == 0  # short-circuit, no revision applied
    assert len(result["_critique_history"]) == 1
    assert result["_critique_history"][0]["verdict"] == "accept"


def pd_to_json(d):
    """Tiny helper because we can't use json.dumps inside the FakeChat closure
    without making the test file uglier. Inlined here for clarity."""
    import json as _json
    return _json.dumps(d, ensure_ascii=False)


@pytest.mark.asyncio
async def test_synthesise_with_critic_iterates_when_revise(
    fake_gaps, fake_idea_response, fake_critique_revise, fake_critique_accept,
):
    """First critique says 'revise', second says 'accept' -> one revision applied."""
    from gap2idea.pipeline.openai_ideas import _evidence_payload, _diverse_evidence

    ev_a = _evidence_payload(_diverse_evidence(fake_gaps, 0, k=2))
    ev_b = _evidence_payload(_diverse_evidence(fake_gaps, 1, k=2))

    mock_client = type("C", (), {})()
    class FakeChat:
        class completions:
            @staticmethod
            def create(**kwargs):
                class R:
                    class _Choice:
                        class message:
                            content = '{"pair": {"cluster_a": 0, "cluster_b": 1}, "idea": ' + \
                                pd_to_json(fake_idea_response["idea"]) + '}'
                    choices = [_Choice()]
                return R()
    mock_client.chat = FakeChat

    critiques_iter = iter([
        {**fake_critique_revise, "_diagnostics": {"novelty_score": 0.4}},
        {**fake_critique_accept,  "_diagnostics": {"novelty_score": 0.8}},
    ])
    async def fake_critique(*a, **kw):
        return next(critiques_iter)

    async def fake_revise(*a, **kw):
        return fake_idea_response["idea"]  # pretend revision returns same shape

    with patch.object(agents, "get_llm_client", return_value=mock_client), \
         patch.object(agents, "critique_idea", new=fake_critique), \
         patch.object(agents, "revise_idea", new=fake_revise):
        result = await agents.synthesise_with_critic(
            mode="bridge", cluster_a=0, cluster_b=1,
            label_a="A", label_b="B",
            gaps_df=fake_gaps, fed_evidence_a=ev_a, fed_evidence_b=ev_b,
            max_iterations=3, accept_score=4.0,
        )
    assert result["_n_iterations"] == 1
    assert len(result["_critique_history"]) == 2
    assert result["_critique_history"][0]["verdict"] == "revise"
    assert result["_critique_history"][1]["verdict"] == "accept"


# ---------- multi-judge aggregation ----------

def test_aggregate_panel_consensus_is_mean():
    from gap2idea.pipeline.evaluation import _aggregate_panel

    panel = [
        ("model_a", {"novelty": 4, "specificity": 3, "feasibility": 4, "evidence_grounding": 3,
                     "overall_critique": "good"}),
        ("model_b", {"novelty": 3, "specificity": 3, "feasibility": 4, "evidence_grounding": 3,
                     "overall_critique": "ok"}),
        ("model_c", {"novelty": 5, "specificity": 4, "feasibility": 5, "evidence_grounding": 4,
                     "overall_critique": "excellent and detailed reasoning here"}),
    ]
    out = _aggregate_panel(panel)
    assert out["novelty"] == pytest.approx(4.0)
    assert out["specificity"] == pytest.approx(10/3)
    assert out["n_judges"] == 3
    # Agreement is in [0, 1]
    assert 0.0 <= out["agreement"] <= 1.0
    # The longest critique wins
    assert "excellent" in out["consensus_critique"]


def test_aggregate_panel_handles_partial_failures():
    """One judge crashes -> still aggregate over the others."""
    from gap2idea.pipeline.evaluation import _aggregate_panel

    panel = [
        ("model_a", {"novelty": 4, "specificity": 3, "feasibility": 4, "evidence_grounding": 3}),
        ("model_b", {"novelty": 2, "specificity": 2, "feasibility": 2, "evidence_grounding": 2}),
    ]
    out = _aggregate_panel(panel)
    assert out["n_judges"] == 2
    assert out["novelty"] == 3.0


def test_aggregate_panel_perfect_agreement_is_one():
    """When all judges return the same scores, agreement = 1.0."""
    from gap2idea.pipeline.evaluation import _aggregate_panel

    panel = [(f"m{i}", {"novelty": 3, "specificity": 3, "feasibility": 3, "evidence_grounding": 3}) for i in range(3)]
    out = _aggregate_panel(panel)
    assert out["agreement"] == 1.0
