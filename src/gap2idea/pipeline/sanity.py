"""Multi-agent experimental sanity stage.

Eight agents across four phases sit between the critic loop and the judge
panel. The experiment size is decided per-idea by negotiation in Phase E1
(tier 0..3), not capped at a fixed line count. Some ideas need a 30-line
smoke test; some need a 200-line benchmark with seeded RNG and effect-size
analysis; some are untestable in a sandbox and the system says so.

Architecture:

  Phase E1 — Protocol design (deliberation)
      Planner ⇄ Adversary ⇄ Scale-Estimator → Protocol-Synthesiser
  Phase E2 — Implementation (iterate, ≤2 rounds)
      Coder ⇄ Reviewer
  Phase E3 — Execution (sandboxed)
      run_sandbox + Diagnostician (≤1 retry)
  Phase E4 — Interpretation (panel + synthesis)
      Analyst + Skeptic + Statistician → Verdict-Synthesiser

Each agent is a system prompt + strict JSON schema + retry-wrapped LLM call.
This is genuine multi-agent: agents on different models can disagree
productively, and the Verdict-Synthesiser is required to return
`inconclusive` when disagreement is unresolved.

The final verdict is surfaced to the judge panel as additional context, so
the panel can calibrate feasibility and novelty against empirical signal.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

from gap2idea.pipeline import resource_compat
from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client, parse_json_response
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)


# ---------- budget tiers ----------

#: Maximum tier the Scale-Estimator may choose per budget setting.
BUDGET_MAX_TIER: dict[str, int] = {"smoke": 1, "benchmark": 2, "full": 3}

#: Wall-clock seconds per tier. Tier 0 = untestable (sandbox not run).
TIER_WALL_BUDGET_S: dict[int, int] = {0: 0, 1: 10, 2: 90, 3: 600}

#: Soft memory cap per tier (POSIX only). Mostly informational on Windows.
TIER_MEM_MB: dict[int, int] = {0: 0, 1: 512, 2: 1024, 3: 2048}


# ---------- per-role default models ----------

#: Conservative defaults. Stronger reasoners do the adversarial / synthesis
#: work; cheaper models handle Planner / Reviewer / Analyst / Statistician /
#: Diagnostician. All overridable via the `models` argument of
#: `run_sanity_check`.
DEFAULT_SANITY_MODELS: dict[str, str] = {
    "planner":          "openai/gpt-4.1-mini",
    "adversary":        "anthropic/claude-sonnet-4",
    "scale_estimator":  "openai/gpt-4.1-mini",
    "protocol_synth":   "anthropic/claude-sonnet-4",
    "coder":            "anthropic/claude-sonnet-4",
    "reviewer":         "openai/gpt-4.1-mini",
    "diagnostician":    "openai/gpt-4.1-mini",
    "analyst":          "openai/gpt-4.1-mini",
    "skeptic":          "anthropic/claude-sonnet-4",
    "statistician":     "openai/gpt-4.1-mini",
    "verdict":          "anthropic/claude-sonnet-4",
}


# ===========================================================================
# JSON schemas (one per agent — strict mode)
# ===========================================================================

PROTOCOL_SCHEMA = {
    "type": "object",
    "properties": {
        "tier": {"type": "integer", "minimum": 0, "maximum": 3},
        "research_claim":           {"type": "string"},
        "datasets":                 {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":   {"type": "string"},
                    "source": {"type": "string"},
                    "size":   {"type": "string"},
                    "why":    {"type": "string"},
                },
                "required": ["name", "source", "size", "why"],
                "additionalProperties": False,
            },
        },
        "baselines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":       {"type": "string"},
                    "why_chosen": {"type": "string"},
                },
                "required": ["name", "why_chosen"],
                "additionalProperties": False,
            },
        },
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name":                 {"type": "string"},
                    "formula_or_reference": {"type": "string"},
                },
                "required": ["name", "formula_or_reference"],
                "additionalProperties": False,
            },
        },
        "n_seeds":                  {"type": "integer", "minimum": 1, "maximum": 30},
        "wall_budget_seconds":      {"type": "integer", "minimum": 0, "maximum": 1200},
        "expected_outcome_if_true": {"type": "string"},
        "expected_outcome_if_false":{"type": "string"},
        "abort_conditions":         {"type": "array", "items": {"type": "string"}},
        "justification":            {"type": "string"},
    },
    "required": [
        "tier", "research_claim", "datasets", "baselines", "metrics",
        "n_seeds", "wall_budget_seconds",
        "expected_outcome_if_true", "expected_outcome_if_false",
        "abort_conditions", "justification",
    ],
    "additionalProperties": False,
}


ATTACKS_SCHEMA = {
    "type": "object",
    "properties": {
        "attacks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "axis":       {"type": "string", "enum": [
                        "baseline_too_weak", "metric_mismatch", "data_too_easy",
                        "scale_insufficient", "confound", "leakage",
                        "infeasible", "vague_claim", "other",
                    ]},
                    "problem":    {"type": "string"},
                    "suggested_fix": {"type": "string"},
                },
                "required": ["axis", "problem", "suggested_fix"],
                "additionalProperties": False,
            },
        },
        "severity": {"type": "integer", "minimum": 0, "maximum": 5,
                     "description": "Overall severity of attacks; 0=no concerns, 5=protocol must change."},
    },
    "required": ["attacks", "severity"],
    "additionalProperties": False,
}


SCALE_SCHEMA = {
    "type": "object",
    "properties": {
        "tier":          {"type": "integer", "minimum": 0, "maximum": 3},
        "justification": {"type": "string"},
        "must_change":   {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tier", "justification", "must_change"],
    "additionalProperties": False,
}


CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "code":  {"type": "string"},
        "notes": {"type": "string"},
    },
    "required": ["code", "notes"],
    "additionalProperties": False,
}


REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "accepts": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "problem":  {"type": "string"},
                    "fix":      {"type": "string"},
                },
                "required": ["severity", "problem", "fix"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["accepts", "issues"],
    "additionalProperties": False,
}


DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "can_fix":       {"type": "boolean"},
        "suggested_fix": {"type": "string"},
    },
    "required": ["can_fix", "suggested_fix"],
    "additionalProperties": False,
}


FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts":   {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "string"},
    },
    "required": ["facts", "summary"],
    "additionalProperties": False,
}


ALT_EXPLANATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "alternative_explanations": {"type": "array", "items": {"type": "string"}},
        "confound_score":           {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["alternative_explanations", "confound_score"],
    "additionalProperties": False,
}


STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "effect_size":              {"type": ["number", "null"]},
        "ci":                       {"type": "string"},
        "p_value_or_seed_variance": {"type": "string"},
        "power_assessment":         {"type": "string",
                                      "enum": ["sufficient", "marginal", "insufficient", "n/a"]},
        "notes":                    {"type": "string"},
    },
    "required": ["effect_size", "ci", "p_value_or_seed_variance",
                 "power_assessment", "notes"],
    "additionalProperties": False,
}


FINAL_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "sanity_supported":      {"type": "string",
                                   "enum": ["yes", "partial", "no", "inconclusive", "untestable"]},
        "sanity_signal":         {"type": "number", "minimum": 0, "maximum": 1},
        "sanity_notes":          {"type": "string"},
    },
    "required": ["sanity_supported", "sanity_signal", "sanity_notes"],
    "additionalProperties": False,
}


# ===========================================================================
# Shared LLM-call helper
# ===========================================================================

@retry(tries=3, base_delay=2.0)
def _call_agent_sync(
    client: OpenAI,
    *,
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """One synchronous LLM call with strict JSON-schema response format."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": True},
        },
        temperature=temperature,
    )
    return parse_json_response(resp.choices[0].message.content)


async def _call_agent(
    client: OpenAI,
    *,
    system: str,
    user: str,
    schema: dict,
    schema_name: str,
    model: str,
    temperature: float = 0.0,
) -> dict:
    """Async wrapper so a slow LLM call doesn't block the event loop. Each
    agent fan-out can therefore overlap with sandbox I/O downstream."""
    return await asyncio.to_thread(
        _call_agent_sync,
        client,
        system=system, user=user,
        schema=schema, schema_name=schema_name,
        model=model, temperature=temperature,
    )


# ===========================================================================
# Phase E1: Protocol design (deliberation)
# ===========================================================================

PLANNER_SYSTEM = (
    "You are an experiment planner. Given a research idea, draft a concrete "
    "experimental protocol. Choose a TIER reflecting how much rigour the "
    "claim requires:\n"
    "  0 — Untestable in a code sandbox (needs humans, proprietary data, hardware).\n"
    "  1 — Smoke test on toy data; verifies direction only (~10s wall).\n"
    "  2 — Benchmark on synthetic controlled data with ≥3 seeds and a baseline (~90s).\n"
    "  3 — Mini-real on a small public dataset (sklearn-bundled or single file), "
    "      proper splits and statistical tests (~600s).\n"
    "Choose the SMALLEST tier that lets the claim be falsified. Use real metric "
    "names. Use real baseline names. Your justification must explain the tier choice."
)


ADVERSARY_SYSTEM = (
    "You are a methodologically adversarial reviewer. Attack the proposed "
    "experimental protocol the way a tough conference reviewer would. Surface "
    "weak baselines, datasets that are too easy, metrics that don't measure "
    "what the claim says, scale that's insufficient for the effect size, "
    "obvious confounds, and leakage. Be specific. If the protocol is genuinely "
    "fine, return zero attacks and severity=0 — do NOT manufacture concerns."
)


SCALE_ESTIMATOR_SYSTEM = (
    "You arbitrate between the Planner (chose a tier) and the Adversary "
    "(attacked the protocol). Decide the FINAL tier. Push UP if the Adversary "
    "shows the tier is too weak for the claim; push DOWN only if the tier is "
    "infeasible *and* a lower tier still falsifies the claim. You MUST respect "
    "the corpus-wide max_tier passed in the user message — never propose a "
    "higher tier than that, even if the Adversary requested one."
)


PROTOCOL_SYNTHESISER_SYSTEM = (
    "You produce a FINAL experimental protocol given (1) the Planner's draft, "
    "(2) the Adversary's attacks, (3) the Scale-Estimator's tier decision. "
    "Apply Adversary fixes that the Scale-Estimator's must_change list endorses. "
    "Keep the final tier exactly as the Scale-Estimator decided. Produce a "
    "protocol that is concrete enough for a code-writing agent to implement directly."
)


async def agent_planner(idea: dict, *, client: OpenAI, model: str) -> dict:
    user = (
        "Idea under test:\n"
        + json.dumps(idea, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON matching the schema."
    )
    return await _call_agent(
        client, system=PLANNER_SYSTEM, user=user,
        schema=PROTOCOL_SCHEMA, schema_name="protocol",
        model=model, temperature=0.0,
    )


async def agent_adversary(idea: dict, draft: dict, *, client: OpenAI, model: str) -> dict:
    user = (
        "Idea:\n" + json.dumps(idea, ensure_ascii=False, indent=2)
        + "\n\nProposed protocol:\n" + json.dumps(draft, ensure_ascii=False, indent=2)
        + "\n\nList up to 5 specific attacks. If the protocol is fine, return "
          "{\"attacks\": [], \"severity\": 0}. Be honest, not performative."
    )
    return await _call_agent(
        client, system=ADVERSARY_SYSTEM, user=user,
        schema=ATTACKS_SCHEMA, schema_name="attacks",
        model=model, temperature=0.2,
    )


async def agent_scale_estimator(
    idea: dict, draft: dict, attacks: dict, *, max_tier: int,
    client: OpenAI, model: str,
) -> dict:
    user = (
        f"Maximum allowed tier (per budget): {max_tier}\n\n"
        "Idea:\n" + json.dumps(idea, ensure_ascii=False, indent=2)
        + "\n\nDraft protocol:\n" + json.dumps(draft, ensure_ascii=False, indent=2)
        + "\n\nAttacks:\n" + json.dumps(attacks, ensure_ascii=False, indent=2)
        + f"\n\nReturn JSON. tier MUST be in [0, {max_tier}]."
    )
    result = await _call_agent(
        client, system=SCALE_ESTIMATOR_SYSTEM, user=user,
        schema=SCALE_SCHEMA, schema_name="scale",
        model=model, temperature=0.0,
    )
    # Hard clamp: enforce budget cap even if the LLM ignored the instruction.
    result["tier"] = max(0, min(int(result["tier"]), max_tier))
    return result


async def agent_protocol_synthesiser(
    idea: dict, draft: dict, attacks: dict, scale_call: dict,
    *, client: OpenAI, model: str,
) -> dict:
    user = (
        "Idea:\n" + json.dumps(idea, ensure_ascii=False, indent=2)
        + "\n\nDraft:\n" + json.dumps(draft, ensure_ascii=False, indent=2)
        + "\n\nAttacks:\n" + json.dumps(attacks, ensure_ascii=False, indent=2)
        + "\n\nScale decision:\n" + json.dumps(scale_call, ensure_ascii=False, indent=2)
        + "\n\nReturn the FINAL protocol JSON. The tier MUST equal "
        + f"{scale_call['tier']}. Apply the must_change fixes."
    )
    final = await _call_agent(
        client, system=PROTOCOL_SYNTHESISER_SYSTEM, user=user,
        schema=PROTOCOL_SCHEMA, schema_name="protocol_final",
        model=model, temperature=0.0,
    )
    # Authoritative tier comes from the Scale-Estimator, not the synthesiser.
    final["tier"] = int(scale_call["tier"])
    # Cap wall_budget_seconds to the tier's standard budget.
    final["wall_budget_seconds"] = min(
        int(final.get("wall_budget_seconds") or 0),
        TIER_WALL_BUDGET_S[final["tier"]],
    )
    return final


async def design_protocol(
    idea: dict, *, budget: str = "benchmark",
    models: dict[str, str] | None = None,
    client: OpenAI | None = None,
) -> tuple[dict, dict]:
    """Run the four-agent E1 deliberation. Returns (final_protocol, trace)."""
    models = {**DEFAULT_SANITY_MODELS, **(models or {})}
    client = client or get_llm_client()
    max_tier = BUDGET_MAX_TIER.get(budget, 2)

    draft   = await agent_planner(idea, client=client, model=models["planner"])
    attacks = await agent_adversary(idea, draft, client=client, model=models["adversary"])
    scale   = await agent_scale_estimator(
        idea, draft, attacks, max_tier=max_tier,
        client=client, model=models["scale_estimator"],
    )
    final   = await agent_protocol_synthesiser(
        idea, draft, attacks, scale,
        client=client, model=models["protocol_synth"],
    )
    return final, {
        "draft": draft, "attacks": attacks, "scale": scale, "final": final,
        "max_tier": max_tier,
    }


# ===========================================================================
# Phase E2: Implementation (Coder ⇄ Reviewer loop)
# ===========================================================================

CODER_SYSTEM = (
    "You implement an experimental protocol in self-contained Python. "
    "Use ONLY: stdlib, numpy, scipy, scikit-learn. NO torch, NO transformers, "
    "NO huggingface, NO network access (it will be blocked).\n\n"
    "Requirements for the code:\n"
    "  • Seeded RNG (use the protocol's n_seeds). Set seeds explicitly.\n"
    "  • Print at least one line `RESULT: <metric_name>=<float_value>` to stdout.\n"
    "  • For multi-seed runs, also print `SEED: <n> METRIC: <name>=<value>` per seed.\n"
    "  • Code must terminate within the protocol's wall_budget_seconds.\n"
    "  • Add small comments explaining each section.\n\n"
    "If you are revising prior code with reviewer issues, address EACH issue. "
    "Size the code to the tier — Tier 1 ≈ 30 LOC, Tier 2 ≈ 80-200 LOC, "
    "Tier 3 ≈ 200-500 LOC. Do not pad."
)


REVIEWER_SYSTEM = (
    "You review experimental code against its protocol. Reject if any of:\n"
    "  • The code does not implement the protocol's metrics, baselines, or datasets.\n"
    "  • RNG is unseeded or n_seeds is ignored.\n"
    "  • Code uses banned imports (torch, transformers, requests, urllib for network).\n"
    "  • Code lacks the `RESULT: <metric>=<value>` print.\n"
    "  • Code obviously won't terminate in wall_budget_seconds.\n"
    "  • Code's logic does not actually test the research_claim.\n"
    "Otherwise accept. Be honest, not performative."
)


async def agent_coder(
    protocol: dict, *,
    prior_code: str | None = None,
    reviewer_issues: list[dict] | None = None,
    client: OpenAI, model: str,
) -> dict:
    user_parts = [
        "Protocol:\n" + json.dumps(protocol, ensure_ascii=False, indent=2),
    ]
    if prior_code is not None:
        user_parts.append("Prior code:\n```python\n" + prior_code + "\n```")
    if reviewer_issues:
        user_parts.append(
            "Reviewer issues to address:\n"
            + json.dumps(reviewer_issues, ensure_ascii=False, indent=2)
        )
    user_parts.append("Return JSON with `code` and `notes`.")
    return await _call_agent(
        client, system=CODER_SYSTEM, user="\n\n".join(user_parts),
        schema=CODE_SCHEMA, schema_name="code",
        model=model, temperature=0.4,
    )


async def agent_reviewer(
    protocol: dict, code: str, *, client: OpenAI, model: str,
) -> dict:
    user = (
        "Protocol:\n" + json.dumps(protocol, ensure_ascii=False, indent=2)
        + "\n\nCode to review:\n```python\n" + code + "\n```\n\n"
        "Return JSON with `accepts` (bool) and `issues` (array)."
    )
    return await _call_agent(
        client, system=REVIEWER_SYSTEM, user=user,
        schema=REVIEW_SCHEMA, schema_name="review",
        model=model, temperature=0.0,
    )


async def implement_protocol(
    protocol: dict, *,
    max_rounds: int = 2,
    models: dict[str, str] | None = None,
    client: OpenAI | None = None,
) -> dict:
    """Coder ⇄ Reviewer loop. Returns
        {"code": str, "rounds": [{"code", "review"}], "accepted": bool}.
    """
    models = {**DEFAULT_SANITY_MODELS, **(models or {})}
    client = client or get_llm_client()

    rounds: list[dict] = []
    code = ""
    issues: list[dict] | None = None
    for r in range(max_rounds):
        gen = await agent_coder(
            protocol,
            prior_code=code or None, reviewer_issues=issues,
            client=client, model=models["coder"],
        )
        code = gen["code"]
        review = await agent_reviewer(
            protocol, code, client=client, model=models["reviewer"],
        )
        rounds.append({"code": code, "review": review, "round": r})
        if review.get("accepts"):
            return {"code": code, "rounds": rounds, "accepted": True}
        issues = review.get("issues") or []
    return {"code": code, "rounds": rounds, "accepted": False}


# ===========================================================================
# Phase E3: Execution (sandboxed)
# ===========================================================================

_RESULT_RE = re.compile(r"^\s*RESULT:\s*([A-Za-z0-9_\-\.]+)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
                        re.MULTILINE)
_SEED_RE = re.compile(
    r"^\s*SEED:\s*(\d+)\s+METRIC:\s*([A-Za-z0-9_\-\.]+)\s*=\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$",
    re.MULTILINE,
)


# Network-blocking stub PRE-pended to every script before execution. We never
# trust the model to do this itself.
#
# Implementation note: we cannot REPLACE `socket.socket` with a bare function,
# because `ssl.SSLSocket(socket.socket)` subclasses it — replacing it breaks
# the entire stdlib network stack on import. Instead we subclass `socket.socket`
# and raise on `connect()` / `connect_ex()` / `sendto()`. This kills outbound
# network without breaking `ssl`, `urllib.request`, or `http.client` at *import*
# time. The actual `urlopen` call is also redirected to a hard fail for
# belt-and-braces.
_NETWORK_STUB = """\
import socket as _sock
_orig_socket = _sock.socket
class _BlockedSocket(_orig_socket):
    def connect(self, *a, **k):
        raise OSError("network disabled by sandbox")
    def connect_ex(self, *a, **k):
        raise OSError("network disabled by sandbox")
    def sendto(self, *a, **k):
        raise OSError("network disabled by sandbox")
_sock.socket = _BlockedSocket
def _blocked_create_connection(*a, **k):
    raise OSError("network disabled by sandbox")
_sock.create_connection = _blocked_create_connection
import urllib.request as _ur
def _blocked_urlopen(*a, **k):
    raise OSError("network disabled by sandbox")
_ur.urlopen = _blocked_urlopen
# --- end sandbox stub ---

"""


def _parse_results(stdout: str) -> list[dict]:
    out: list[dict] = []
    for m in _RESULT_RE.finditer(stdout):
        out.append({"kind": "result", "metric": m.group(1), "value": float(m.group(2))})
    for m in _SEED_RE.finditer(stdout):
        out.append({
            "kind": "seed_metric",
            "seed":   int(m.group(1)),
            "metric": m.group(2),
            "value":  float(m.group(3)),
        })
    return out


def _tail(s: str, max_bytes: int = 8192) -> str:
    if not s:
        return ""
    b = s.encode("utf-8", errors="replace")
    if len(b) <= max_bytes:
        return s
    # tail, not head — interesting output is usually near the end
    return b[-max_bytes:].decode("utf-8", errors="replace")


def run_sandbox(code: str, tier: int, wall_budget_s: int | None = None,
                mem_mb: int | None = None) -> dict:
    """Run `code` in a subprocess with wall-clock and (POSIX) memory limits.

    Tier 0 short-circuits without executing — the protocol declared the claim
    untestable in a sandbox.
    """
    if tier <= 0:
        return {
            "exit_code": None, "stdout_tail": "", "stderr_tail": "",
            "ran_to_completion": False, "wall_seconds": 0.0,
            "parsed_results": [], "skipped_reason": "tier=0 (untestable)",
        }

    wall = wall_budget_s if wall_budget_s and wall_budget_s > 0 else TIER_WALL_BUDGET_S[tier]
    mem  = mem_mb if mem_mb and mem_mb > 0 else TIER_MEM_MB[tier]

    full_code = _NETWORK_STUB + code
    tmpdir = Path(tempfile.mkdtemp(prefix="gap2idea_sanity_"))
    script = tmpdir / "experiment.py"
    script.write_text(full_code, encoding="utf-8")

    # POSIX-only memory limits via preexec_fn. resource_compat is a tiny shim
    # (see module top) that imports `resource` on POSIX and exposes a no-op
    # interface on Windows. This keeps the same code path everywhere.
    preexec_fn = None
    if resource_compat.AVAILABLE:
        def _limits():
            mem_bytes = mem * 1024 * 1024
            resource_compat.setrlimit(resource_compat.RLIMIT_AS,  (mem_bytes, mem_bytes))
            resource_compat.setrlimit(resource_compat.RLIMIT_CPU, (wall + 5, wall + 5))
        preexec_fn = _limits

    # Build a minimal env. Python on Windows requires SYSTEMROOT for hash
    # randomization (CryptGenRandom); without it the interpreter dies before
    # main(). On POSIX we keep PATH empty to harden against shell exec.
    sandbox_env: dict[str, str] = {"PYTHONIOENCODING": "utf-8", "PATH": ""}
    if sys.platform == "win32":
        for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC"):
            v = os.environ.get(k)
            if v:
                sandbox_env[k] = v

    import time
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", str(script)],
            cwd=str(tmpdir),
            env=sandbox_env,
            capture_output=True,
            text=True,
            timeout=wall + 10,
            preexec_fn=preexec_fn,
        )
        ran = True
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        ran = False
        exit_code = -1
        stdout = (e.stdout or b"").decode("utf-8", errors="replace") if isinstance(e.stdout, (bytes, bytearray)) else (e.stdout or "")
        stderr = (e.stderr or b"").decode("utf-8", errors="replace") if isinstance(e.stderr, (bytes, bytearray)) else (e.stderr or "")
        stderr = (stderr or "") + f"\n[sandbox] killed after {wall}s wall-clock timeout"
    wall_seconds = time.perf_counter() - t0

    return {
        "exit_code": exit_code,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "ran_to_completion": ran and exit_code == 0,
        "wall_seconds": wall_seconds,
        "parsed_results": _parse_results(stdout),
        "skipped_reason": None,
    }


DIAGNOSTICIAN_SYSTEM = (
    "You diagnose why an experimental script failed (non-zero exit or no "
    "RESULT line produced). Suggest a MINIMAL fix that keeps the protocol "
    "intact. If the failure is fundamental (banned import, infinite loop, "
    "claim cannot be tested with this approach), set can_fix=false."
)


async def agent_diagnostician(
    code: str, sandbox_result: dict,
    *, client: OpenAI, model: str,
) -> dict:
    user = (
        "Code:\n```python\n" + code + "\n```\n\n"
        "Sandbox result:\n" + json.dumps(
            {k: v for k, v in sandbox_result.items() if k != "parsed_results"},
            ensure_ascii=False, indent=2,
        )
        + "\n\nReturn JSON with `can_fix` and `suggested_fix`."
    )
    return await _call_agent(
        client, system=DIAGNOSTICIAN_SYSTEM, user=user,
        schema=DIAGNOSIS_SCHEMA, schema_name="diagnosis",
        model=model, temperature=0.0,
    )


async def execute_protocol(
    code: str, protocol: dict,
    *,
    models: dict[str, str] | None = None,
    client: OpenAI | None = None,
) -> dict:
    """Run the sandbox, retry once on failure via Diagnostician+Coder.
    Returns {"sandbox": result, "retried": bool, "diagnosis": {...} or None}."""
    models = {**DEFAULT_SANITY_MODELS, **(models or {})}
    client = client or get_llm_client()

    tier = int(protocol.get("tier", 0))
    wall = int(protocol.get("wall_budget_seconds", TIER_WALL_BUDGET_S[tier]))
    result = run_sandbox(code, tier=tier, wall_budget_s=wall)

    needs_retry = (
        result["ran_to_completion"] is False
        or not result["parsed_results"]
    )
    if tier == 0 or not needs_retry:
        return {"sandbox": result, "retried": False, "diagnosis": None,
                "final_code": code}

    diagnosis = await agent_diagnostician(
        code, result, client=client, model=models["diagnostician"],
    )
    if not diagnosis.get("can_fix"):
        return {"sandbox": result, "retried": False, "diagnosis": diagnosis,
                "final_code": code}

    # Apply the diagnostician's fix via one more Coder pass.
    fixed = await agent_coder(
        protocol,
        prior_code=code,
        reviewer_issues=[{
            "severity": "high",
            "problem":  "Run failed in sandbox.",
            "fix":      diagnosis.get("suggested_fix", ""),
        }],
        client=client, model=models["coder"],
    )
    new_code = fixed["code"]
    result2 = run_sandbox(new_code, tier=tier, wall_budget_s=wall)
    return {"sandbox": result2, "retried": True, "diagnosis": diagnosis,
            "final_code": new_code, "first_attempt": result}


# ===========================================================================
# Phase E4: Interpretation (panel + synthesis)
# ===========================================================================

ANALYST_SYSTEM = (
    "You are a careful data analyst. Read the experiment's protocol, code, "
    "and run output. State what the numbers SHOW — no interpretation about "
    "whether the claim is supported, just facts. Each fact must be specific "
    "(metric, value, comparison to baseline if available)."
)


SKEPTIC_SYSTEM = (
    "You are a methodological skeptic. Given the facts an analyst extracted, "
    "list plausible alternative explanations for any apparent effect: "
    "confounds, leakage, selection effects, lucky seed, too-easy data, weak "
    "baseline, metric mismatch. confound_score in [0,1] reflects how serious "
    "these concerns are: 0 = clean experiment, 1 = the result is uninterpretable. "
    "If the experiment did not run or has no meaningful result, return "
    "confound_score=1.0 and an explanation."
)


STATISTICIAN_SYSTEM = (
    "You are a numerical statistician. From the raw RESULT and SEED METRIC "
    "lines in the run output, compute an effect size if comparable values "
    "exist, comment on variance across seeds, and assess power. If there's "
    "only one seed, set power_assessment='insufficient'. If there are no "
    "numbers, set power_assessment='n/a'. effect_size may be null."
)


VERDICT_SYSTEM = (
    "You produce a final verdict on whether the idea's claim survived its "
    "sanity check. Inputs: protocol, analyst facts, skeptic's alternative "
    "explanations and confound_score, statistician's effect/power, code "
    "reviewer acceptance.\n\n"
    "Verdict rules (HARD CONSTRAINTS):\n"
    "  • If the sandbox did not run, return 'inconclusive' or 'untestable'.\n"
    "  • If confound_score >= 0.6, maximum allowed verdict is 'partial'.\n"
    "  • If statistician says 'insufficient' power, max verdict is 'partial'.\n"
    "  • If the protocol's tier was 0, verdict MUST be 'untestable'.\n"
    "sanity_signal in [0,1] reflects your overall confidence the claim "
    "survives. Be honest. Inconclusive is a valid verdict — disagreement is data."
)


async def agent_analyst(
    idea: dict, protocol: dict, code: str, run_output: dict,
    *, client: OpenAI, model: str,
) -> dict:
    user = (
        "Idea:\n" + json.dumps(idea, ensure_ascii=False, indent=2)
        + "\n\nProtocol:\n" + json.dumps(protocol, ensure_ascii=False, indent=2)
        + "\n\nCode (truncated):\n```python\n" + code[:3000] + "\n```"
        + "\n\nRun output (parsed):\n"
        + json.dumps(run_output.get("parsed_results", []), ensure_ascii=False, indent=2)
        + "\n\nStdout tail:\n" + run_output.get("stdout_tail", "")
        + "\n\nReturn JSON with `facts` (array of strings) and `summary` (one sentence)."
    )
    return await _call_agent(
        client, system=ANALYST_SYSTEM, user=user,
        schema=FACTS_SCHEMA, schema_name="analysis",
        model=model, temperature=0.0,
    )


async def agent_skeptic(
    idea: dict, protocol: dict, run_output: dict, facts: dict,
    *, client: OpenAI, model: str,
) -> dict:
    user = (
        "Idea:\n" + json.dumps(idea, ensure_ascii=False, indent=2)
        + "\n\nProtocol:\n" + json.dumps(protocol, ensure_ascii=False, indent=2)
        + "\n\nRun output summary:\n" + json.dumps({
            "ran_to_completion": run_output.get("ran_to_completion"),
            "exit_code":         run_output.get("exit_code"),
            "parsed_results":    run_output.get("parsed_results"),
            "wall_seconds":      run_output.get("wall_seconds"),
        }, ensure_ascii=False, indent=2)
        + "\n\nAnalyst facts:\n" + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON with `alternative_explanations` (array) and `confound_score` (0..1)."
    )
    return await _call_agent(
        client, system=SKEPTIC_SYSTEM, user=user,
        schema=ALT_EXPLANATIONS_SCHEMA, schema_name="confounds",
        model=model, temperature=0.2,
    )


async def agent_statistician(run_output: dict, *, client: OpenAI, model: str) -> dict:
    user = (
        "Run output:\n" + json.dumps({
            "parsed_results": run_output.get("parsed_results", []),
            "stdout_tail":    run_output.get("stdout_tail", ""),
        }, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON matching the schema."
    )
    return await _call_agent(
        client, system=STATISTICIAN_SYSTEM, user=user,
        schema=STATS_SCHEMA, schema_name="stats",
        model=model, temperature=0.0,
    )


async def agent_verdict_synthesiser(
    idea: dict, protocol: dict, facts: dict, alt_explanations: dict,
    stats: dict, reviewer_accepted: bool, ran_to_completion: bool,
    *, client: OpenAI, model: str,
) -> dict:
    user = (
        "Idea:\n" + json.dumps(idea, ensure_ascii=False, indent=2)
        + "\n\nProtocol tier: " + str(protocol.get("tier"))
        + "\nReviewer accepted code: " + str(reviewer_accepted)
        + "\nSandbox ran to completion: " + str(ran_to_completion)
        + "\n\nAnalyst:\n" + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\nSkeptic:\n" + json.dumps(alt_explanations, ensure_ascii=False, indent=2)
        + "\n\nStatistician:\n" + json.dumps(stats, ensure_ascii=False, indent=2)
        + "\n\nReturn JSON with sanity_supported, sanity_signal, sanity_notes."
    )
    verdict = await _call_agent(
        client, system=VERDICT_SYSTEM, user=user,
        schema=FINAL_VERDICT_SCHEMA, schema_name="verdict",
        model=model, temperature=0.0,
    )

    # ----- Hard constraints enforced AFTER the LLM, not just in the prompt -----
    confound = float(alt_explanations.get("confound_score", 0.0) or 0.0)
    tier = int(protocol.get("tier", 0))
    power = str(stats.get("power_assessment", "") or "").lower()

    if tier == 0:
        verdict["sanity_supported"] = "untestable"
    elif not ran_to_completion:
        if verdict["sanity_supported"] == "yes":
            verdict["sanity_supported"] = "inconclusive"
    elif confound >= 0.6 and verdict["sanity_supported"] == "yes":
        verdict["sanity_supported"] = "partial"
    elif power == "insufficient" and verdict["sanity_supported"] == "yes":
        verdict["sanity_supported"] = "partial"

    # sanity_signal clamp
    sig = float(verdict.get("sanity_signal", 0.0) or 0.0)
    verdict["sanity_signal"] = max(0.0, min(1.0, sig))
    return verdict


async def interpret_results(
    idea: dict, protocol: dict, code: str, run_output: dict,
    *, reviewer_accepted: bool,
    models: dict[str, str] | None = None,
    client: OpenAI | None = None,
) -> dict:
    """Fan out Analyst + Skeptic + Statistician in parallel, then synthesise."""
    models = {**DEFAULT_SANITY_MODELS, **(models or {})}
    client = client or get_llm_client()

    facts = await agent_analyst(
        idea, protocol, code, run_output,
        client=client, model=models["analyst"],
    )
    skeptic_task = agent_skeptic(
        idea, protocol, run_output, facts,
        client=client, model=models["skeptic"],
    )
    stats_task = agent_statistician(
        run_output, client=client, model=models["statistician"],
    )
    alt, stats = await asyncio.gather(skeptic_task, stats_task)

    verdict = await agent_verdict_synthesiser(
        idea, protocol, facts, alt, stats,
        reviewer_accepted=reviewer_accepted,
        ran_to_completion=bool(run_output.get("ran_to_completion")),
        client=client, model=models["verdict"],
    )
    return {
        "facts": facts, "skeptic": alt, "stats": stats, "verdict": verdict,
    }


# ===========================================================================
# Top-level entry point
# ===========================================================================

def _skipped_verdict(reason: str, *, tier: int = 0,
                     effect_size: float | None = None,
                     confound_score: float = 1.0) -> dict:
    """Return the standard 'this idea did not go through the sanity stage' verdict."""
    return {
        "sanity_tier":           tier,
        "sanity_ran":            False,
        "sanity_supported":      "inconclusive" if tier > 0 else "untestable",
        "sanity_signal":         0.0,
        "sanity_effect_size":    effect_size,
        "sanity_confound_score": confound_score,
        "sanity_notes":          f"skipped: {reason}",
        "_trace":                {"skipped": True, "reason": reason},
    }


def _gate_passes(idea: dict, accept_score: float = 4.0,
                 critique_history: list[dict] | None = None) -> tuple[bool, str]:
    """Confidence + critic-converged gate. Return (passes, reason_if_not)."""
    conf = float(idea.get("confidence", 0.0) or 0.0)
    if conf < 0.5:
        return False, f"confidence={conf:.2f} < 0.5"
    if critique_history:
        last = critique_history[-1]
        if last.get("verdict") not in ("accept", "revise"):
            return False, f"critic verdict={last.get('verdict')!r}"
        if (last.get("verdict") == "revise"
                and float(last.get("score", 0.0)) < accept_score):
            return False, f"critic score={last.get('score')} < {accept_score}"
    return True, ""


async def run_sanity_check(
    idea: dict,
    *,
    budget: str = "benchmark",
    models: dict[str, str] | None = None,
    critique_history: list[dict] | None = None,
    accept_score: float = 4.0,
    client: OpenAI | None = None,
) -> dict:
    """End-to-end multi-agent experimental sanity check on one idea.

    Returns a flat dict with the FINAL_VERDICT fields PLUS a `_trace` block
    capturing every agent's input + output for offline analysis.
    """
    passes, reason = _gate_passes(idea, accept_score=accept_score,
                                   critique_history=critique_history)
    if not passes:
        return _skipped_verdict(reason, tier=0)

    models = {**DEFAULT_SANITY_MODELS, **(models or {})}
    client = client or get_llm_client()

    # --- E1: protocol ---
    protocol, e1_trace = await design_protocol(
        idea, budget=budget, models=models, client=client,
    )
    tier = int(protocol["tier"])
    log.info("  sanity E1: tier=%s wall=%ss", tier, protocol["wall_budget_seconds"])
    if tier == 0:
        return {
            "sanity_tier":           0,
            "sanity_ran":            False,
            "sanity_supported":      "untestable",
            "sanity_signal":         0.0,
            "sanity_effect_size":    None,
            "sanity_confound_score": 1.0,
            "sanity_notes":          protocol.get("justification", "untestable in sandbox"),
            "_trace": {"e1": e1_trace, "skipped_after_e1": True},
        }

    # --- E2: implementation ---
    impl = await implement_protocol(
        protocol, max_rounds=2, models=models, client=client,
    )
    log.info("  sanity E2: code_rounds=%d accepted=%s", len(impl["rounds"]), impl["accepted"])

    # --- E3: execution ---
    exec_block = await execute_protocol(
        impl["code"], protocol, models=models, client=client,
    )
    run_output = exec_block["sandbox"]
    code_used  = exec_block.get("final_code", impl["code"])
    log.info("  sanity E3: ran=%s exit=%s wall=%.1fs retried=%s",
             run_output["ran_to_completion"], run_output["exit_code"],
             run_output["wall_seconds"], exec_block["retried"])

    # --- E4: interpretation ---
    e4 = await interpret_results(
        idea, protocol, code_used, run_output,
        reviewer_accepted=bool(impl["accepted"]),
        models=models, client=client,
    )

    verdict = e4["verdict"]
    effect_size = e4["stats"].get("effect_size")
    confound = float(e4["skeptic"].get("confound_score", 0.0) or 0.0)

    return {
        "sanity_tier":           tier,
        "sanity_ran":            bool(run_output["ran_to_completion"]),
        "sanity_supported":      verdict["sanity_supported"],
        "sanity_signal":         float(verdict["sanity_signal"]),
        "sanity_effect_size":    effect_size,
        "sanity_confound_score": confound,
        "sanity_notes":          verdict["sanity_notes"],
        "_trace": {
            "e1": e1_trace,
            "e2": impl,
            "e3": exec_block,
            "e4": e4,
        },
    }
