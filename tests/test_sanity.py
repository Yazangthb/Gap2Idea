"""Unit tests for the multi-agent experimental sanity stage.

LLM calls are mocked. The sandbox is REAL — we test the actual subprocess
runner against synthetic scripts to verify timeout and network blocking.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from gap2idea.pipeline import sanity


# ===========================================================================
# Sandbox: real subprocess execution
# ===========================================================================

def test_sandbox_parses_result_and_seed_lines():
    code = (
        "print('RESULT: f1=0.83')\n"
        "for s in range(3):\n"
        "    print(f'SEED: {s} METRIC: f1={0.80 + s * 0.01:.2f}')\n"
    )
    out = sanity.run_sandbox(code, tier=1, wall_budget_s=10)
    assert out["ran_to_completion"] is True
    assert out["exit_code"] == 0
    # 1 RESULT + 3 SEED lines = 4 parsed entries
    assert len(out["parsed_results"]) == 4
    kinds = {p["kind"] for p in out["parsed_results"]}
    assert kinds == {"result", "seed_metric"}


def test_sandbox_tier_zero_short_circuits():
    """Tier 0 must NOT invoke a subprocess."""
    out = sanity.run_sandbox("import os; os.system('echo should-not-run')",
                             tier=0, wall_budget_s=10)
    assert out["ran_to_completion"] is False
    assert out["exit_code"] is None
    assert out["skipped_reason"] == "tier=0 (untestable)"


def test_sandbox_blocks_network():
    """The injected network stub must make urllib.request.urlopen raise."""
    code = (
        "try:\n"
        "    import urllib.request\n"
        "    urllib.request.urlopen('http://example.com')\n"
        "    print('UNREACHABLE')\n"
        "except OSError as e:\n"
        "    print(f'BLOCKED: {e}')\n"
        "    print('RESULT: blocked=1')\n"
    )
    out = sanity.run_sandbox(code, tier=1, wall_budget_s=10)
    assert out["ran_to_completion"] is True
    assert "BLOCKED" in out["stdout_tail"]
    assert "UNREACHABLE" not in out["stdout_tail"]


def test_sandbox_enforces_wall_budget():
    """A 30-second infinite loop must be killed under a 5-second budget."""
    code = "while True:\n    pass\n"
    out = sanity.run_sandbox(code, tier=1, wall_budget_s=3)
    assert out["ran_to_completion"] is False
    assert out["wall_seconds"] < 30  # killed long before naturally exiting


# ===========================================================================
# Result parsing helpers
# ===========================================================================

def test_parse_results_handles_negative_and_scientific():
    stdout = (
        "RESULT: loss=-0.42\n"
        "RESULT: ratio=1.5e-3\n"
        "RESULT: pi=3.14159\n"
    )
    parsed = sanity._parse_results(stdout)
    metrics = {p["metric"]: p["value"] for p in parsed if p["kind"] == "result"}
    assert metrics["loss"] == -0.42
    assert metrics["ratio"] == pytest.approx(1.5e-3)
    assert metrics["pi"] == pytest.approx(3.14159)


def test_parse_results_ignores_non_matching_lines():
    stdout = "noise\nRESULT: f1=0.7\nmore noise\n"
    parsed = sanity._parse_results(stdout)
    assert len(parsed) == 1
    assert parsed[0]["metric"] == "f1"


def test_tail_keeps_end_bytes():
    long = "x" * 20000
    tail = sanity._tail(long, max_bytes=100)
    assert len(tail) <= 200  # multi-byte safety margin
    assert tail.endswith("x")


# ===========================================================================
# Gate logic — short-circuit before any LLM call
# ===========================================================================

@pytest.mark.asyncio
async def test_run_sanity_short_circuits_on_low_confidence():
    idea = {"title": "X", "confidence": 0.3,
            "falsifiable_prediction": "5% gain", "named_baseline": "BERT"}
    # No mocks needed — the gate must reject before any LLM call.
    out = await sanity.run_sanity_check(idea)
    assert out["sanity_ran"] is False
    assert "skipped" in out["sanity_notes"]
    assert out["_trace"]["skipped"] is True


@pytest.mark.asyncio
async def test_run_sanity_short_circuits_on_critic_rejection():
    idea = {"title": "X", "confidence": 0.9,
            "falsifiable_prediction": "5% gain", "named_baseline": "BERT"}
    history = [{"verdict": "reject", "score": 1.5}]
    out = await sanity.run_sanity_check(idea, critique_history=history)
    assert out["sanity_ran"] is False
    assert "skipped" in out["sanity_notes"]


# ===========================================================================
# E1 deliberation — Scale-Estimator clamps tier to budget
# ===========================================================================

@pytest.mark.asyncio
async def test_scale_estimator_clamps_tier_to_max():
    """The Scale-Estimator must hard-clamp tier to the budget cap, even if the
    LLM returns a higher value."""
    overshooting = AsyncMock(return_value={
        "tier": 3,
        "justification": "needs tier 3",
        "must_change": [],
    })
    with patch.object(sanity, "_call_agent", overshooting):
        out = await sanity.agent_scale_estimator(
            idea={}, draft={}, attacks={"attacks": [], "severity": 0},
            max_tier=1, client=object(), model="dummy",
        )
    assert out["tier"] == 1  # clamped from 3 to budget cap


# ===========================================================================
# E2 loop — Reviewer rejection forces a second Coder round
# ===========================================================================

@pytest.mark.asyncio
async def test_implement_protocol_loops_until_reviewer_accepts():
    """Reviewer rejects round 1, accepts round 2 -> final code is the second
    Coder pass, trace records both rounds."""
    protocol = {"tier": 1, "wall_budget_seconds": 10}
    coder = AsyncMock(side_effect=[
        {"code": "v1", "notes": "first draft"},
        {"code": "v2", "notes": "fixes applied"},
    ])
    reviewer = AsyncMock(side_effect=[
        {"accepts": False, "issues": [{"severity": "high",
                                         "problem": "no RESULT line",
                                         "fix": "print one"}]},
        {"accepts": True, "issues": []},
    ])
    with patch.object(sanity, "agent_coder", coder), \
         patch.object(sanity, "agent_reviewer", reviewer), \
         patch.object(sanity, "get_llm_client", return_value=object()):
        result = await sanity.implement_protocol(protocol, max_rounds=3)
    assert result["accepted"] is True
    assert result["code"] == "v2"
    assert len(result["rounds"]) == 2


@pytest.mark.asyncio
async def test_implement_protocol_returns_unaccepted_when_max_rounds_hit():
    protocol = {"tier": 1, "wall_budget_seconds": 10}
    coder = AsyncMock(return_value={"code": "vN", "notes": "n"})
    reviewer = AsyncMock(return_value={"accepts": False,
                                        "issues": [{"severity": "high",
                                                     "problem": "x",
                                                     "fix": "y"}]})
    with patch.object(sanity, "agent_coder", coder), \
         patch.object(sanity, "agent_reviewer", reviewer), \
         patch.object(sanity, "get_llm_client", return_value=object()):
        result = await sanity.implement_protocol(protocol, max_rounds=2)
    assert result["accepted"] is False
    assert len(result["rounds"]) == 2


# ===========================================================================
# E4 hard constraints — Verdict-Synthesiser must downgrade when warranted
# ===========================================================================

@pytest.mark.asyncio
async def test_verdict_downgrade_on_high_confound():
    """If confound_score >= 0.6, 'yes' must be downgraded to 'partial'."""
    raw_verdict = {
        "sanity_supported": "yes",
        "sanity_signal":    0.9,
        "sanity_notes":     "strong",
    }
    with patch.object(sanity, "_call_agent", AsyncMock(return_value=raw_verdict)):
        out = await sanity.agent_verdict_synthesiser(
            idea={"title": "x"}, protocol={"tier": 2},
            facts={"facts": ["x"], "summary": "s"},
            alt_explanations={"alternative_explanations": ["leakage"], "confound_score": 0.8},
            stats={"effect_size": 0.5, "ci": "n/a", "p_value_or_seed_variance": "n/a",
                   "power_assessment": "sufficient", "notes": ""},
            reviewer_accepted=True, ran_to_completion=True,
            client=object(), model="dummy",
        )
    assert out["sanity_supported"] == "partial"


@pytest.mark.asyncio
async def test_verdict_downgrade_on_insufficient_power():
    raw_verdict = {"sanity_supported": "yes", "sanity_signal": 0.8, "sanity_notes": "n"}
    with patch.object(sanity, "_call_agent", AsyncMock(return_value=raw_verdict)):
        out = await sanity.agent_verdict_synthesiser(
            idea={"title": "x"}, protocol={"tier": 2},
            facts={"facts": [], "summary": ""},
            alt_explanations={"alternative_explanations": [], "confound_score": 0.1},
            stats={"effect_size": None, "ci": "n/a", "p_value_or_seed_variance": "n/a",
                   "power_assessment": "insufficient", "notes": "one seed"},
            reviewer_accepted=True, ran_to_completion=True,
            client=object(), model="dummy",
        )
    assert out["sanity_supported"] == "partial"


@pytest.mark.asyncio
async def test_verdict_forced_to_untestable_at_tier_zero():
    raw_verdict = {"sanity_supported": "yes", "sanity_signal": 1.0, "sanity_notes": "n"}
    with patch.object(sanity, "_call_agent", AsyncMock(return_value=raw_verdict)):
        out = await sanity.agent_verdict_synthesiser(
            idea={"title": "x"}, protocol={"tier": 0},
            facts={"facts": [], "summary": ""},
            alt_explanations={"alternative_explanations": [], "confound_score": 0.0},
            stats={"effect_size": None, "ci": "n/a", "p_value_or_seed_variance": "n/a",
                   "power_assessment": "n/a", "notes": ""},
            reviewer_accepted=False, ran_to_completion=False,
            client=object(), model="dummy",
        )
    assert out["sanity_supported"] == "untestable"


@pytest.mark.asyncio
async def test_verdict_inconclusive_when_sandbox_did_not_run():
    raw_verdict = {"sanity_supported": "yes", "sanity_signal": 0.9, "sanity_notes": "n"}
    with patch.object(sanity, "_call_agent", AsyncMock(return_value=raw_verdict)):
        out = await sanity.agent_verdict_synthesiser(
            idea={"title": "x"}, protocol={"tier": 2},
            facts={"facts": [], "summary": ""},
            alt_explanations={"alternative_explanations": [], "confound_score": 0.1},
            stats={"effect_size": None, "ci": "n/a", "p_value_or_seed_variance": "n/a",
                   "power_assessment": "n/a", "notes": ""},
            reviewer_accepted=True, ran_to_completion=False,
            client=object(), model="dummy",
        )
    assert out["sanity_supported"] == "inconclusive"


# ===========================================================================
# End-to-end with everything mocked except the sandbox
# ===========================================================================

@pytest.mark.asyncio
async def test_run_sanity_check_tier_zero_skips_execution():
    """If E1 returns tier=0, no sandbox call happens and verdict is untestable."""
    idea = {"title": "Human-trust study", "confidence": 0.9,
            "falsifiable_prediction": "improves trust by 20%",
            "named_baseline": "GPT-4"}

    async def fake_design(idea, **kw):
        return {
            "tier": 0,
            "research_claim": "untestable",
            "datasets": [], "baselines": [], "metrics": [],
            "n_seeds": 1, "wall_budget_seconds": 0,
            "expected_outcome_if_true": "users say so",
            "expected_outcome_if_false": "users don't",
            "abort_conditions": [],
            "justification": "needs human study",
        }, {"draft": {}, "attacks": {}, "scale": {}, "final": {}, "max_tier": 2}

    with patch.object(sanity, "design_protocol", fake_design), \
         patch.object(sanity, "get_llm_client", return_value=object()):
        out = await sanity.run_sanity_check(idea)
    assert out["sanity_tier"] == 0
    assert out["sanity_ran"] is False
    assert out["sanity_supported"] == "untestable"


# ===========================================================================
# Budget map sanity
# ===========================================================================

def test_budget_max_tier_mapping():
    assert sanity.BUDGET_MAX_TIER["smoke"] == 1
    assert sanity.BUDGET_MAX_TIER["benchmark"] == 2
    assert sanity.BUDGET_MAX_TIER["full"] == 3


def test_tier_wall_budgets_monotonically_increase():
    budgets = sanity.TIER_WALL_BUDGET_S
    assert budgets[0] == 0
    assert budgets[1] < budgets[2] < budgets[3]
