"""Expand a one-paragraph idea into a full paper draft.

Where `openai_ideas.py` produces a short, evidence-grounded idea
(title + RQ + method sketch + eval plan + risks), this module takes
that idea and runs ONE more LLM call to expand it into the structure
of a real conference paper: abstract, intro, related work, method
(with subsections), experimental setup (as a *plan*), expected results,
discussion, conclusion.

Critical design points (so the output is honest and useful, not deceptive):

  • The draft is explicitly a PLAN. No numerical results. No fabricated
    citations. The schema enforces a `human_work_required` field that
    must list what a graduate student still has to do.

  • Related-work entries cite ONLY papers from the input prior_art or
    evidence lists. The model cannot invent paper IDs.

  • The expected_results section describes the *qualitative* pattern
    expected (e.g. "method X should outperform baseline Y on the
    long-tail of metric M"), never specific numbers.

  • All LLM calls go through OpenRouter via `llm.get_llm_client()`. Use
    `parse_json_response` so we tolerate markdown-wrapped JSON.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client, parse_json_response
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)

DEFAULT_DRAFT_MODEL = "openai/gpt-4.1-mini"


# ---------------------------------------------------------------------------
# JSON schema for the full paper draft
# ---------------------------------------------------------------------------

PAPER_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "abstract": {
            "type": "string",
            "description": "150-220 words. Problem -> approach -> evaluation plan -> expected contribution.",
        },
        "introduction": {
            "type": "object",
            "properties": {
                "motivation": {
                    "type": "string",
                    "description": "2-3 paragraphs framing the problem using the evidence gap statements.",
                },
                "contributions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Bullet list of 3-5 specific contributions this paper would make.",
                },
                "paper_structure": {
                    "type": "string",
                    "description": "One short paragraph: 'Section 2 reviews ... Section 3 introduces ...'",
                },
            },
            "required": ["motivation", "contributions", "paper_structure"],
            "additionalProperties": False,
        },
        "related_work": {
            "type": "array",
            "description": "3-8 entries. Each must cite a paper from input prior_art or evidence.",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "title": {"type": "string"},
                    "relevance": {"type": "string"},
                    "how_we_differ": {"type": "string"},
                },
                "required": ["paper_id", "title", "relevance", "how_we_differ"],
                "additionalProperties": False,
            },
        },
        "method": {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "approach": {
                    "type": "string",
                    "description": "Detailed description of the proposed approach. Be concrete.",
                },
                "architecture_or_algorithm": {
                    "type": "string",
                    "description": "Specific architecture/algorithm. Pseudocode-level detail welcome.",
                },
                "training_setup": {
                    "type": "string",
                    "description": "Training procedure / optimisation / loss. Plan, not results.",
                },
            },
            "required": ["overview", "approach", "architecture_or_algorithm", "training_setup"],
            "additionalProperties": False,
        },
        "experimental_setup": {
            "type": "object",
            "properties": {
                "datasets": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Named datasets you would use, with one-sentence justification each.",
                },
                "baselines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Named baseline methods with brief justification.",
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific metrics with brief justification.",
                },
                "implementation_notes": {
                    "type": "string",
                    "description": "Hardware / frameworks / hyperparameter ranges you would explore.",
                },
                "human_work_required": {
                    "type": "string",
                    "description": "Explicit list of things that REQUIRE manual experimentation, "
                                    "graduate-student labour, or domain expertise. Include data "
                                    "collection, ablations, qualitative analysis, etc.",
                },
            },
            "required": ["datasets", "baselines", "metrics",
                         "implementation_notes", "human_work_required"],
            "additionalProperties": False,
        },
        "expected_results": {
            "type": "string",
            "description": "Qualitative patterns expected, NOT specific numbers. "
                           "Use hedging language: 'we expect', 'should', 'likely'.",
        },
        "discussion": {
            "type": "object",
            "properties": {
                "limitations": {"type": "string"},
                "ethical_considerations": {"type": "string"},
                "future_work": {"type": "string"},
            },
            "required": ["limitations", "ethical_considerations", "future_work"],
            "additionalProperties": False,
        },
        "conclusion": {
            "type": "string",
            "description": "One short paragraph. No new claims.",
        },
    },
    "required": [
        "abstract", "introduction", "related_work", "method",
        "experimental_setup", "expected_results", "discussion", "conclusion",
    ],
    "additionalProperties": False,
}


SYSTEM_PROMPT = (
    "You are an expert research paper drafter. Given a research idea, the gap "
    "statements that motivated it, and (optionally) related prior art, produce a "
    "full conference-paper-quality outline in JSON.\n\n"
    "CRITICAL CONSTRAINTS:\n"
    "- This is a PLAN-STAGE draft. You have NOT run any experiments. Do not invent "
    "numerical results, dataset sizes, or specific outcomes.\n"
    "- Use ONLY the provided evidence and prior_art. Do not fabricate citations, "
    "datasets, methods, or numbers.\n"
    "- The experimental_setup section is a research plan: name SPECIFIC datasets, "
    "baselines, and metrics. Briefly justify each.\n"
    "- The expected_results section describes the QUALITATIVE pattern expected "
    "(e.g. 'we expect X to outperform Y on the long-tail subset of M'), NOT "
    "numbers. Hedge appropriately.\n"
    "- experimental_setup.human_work_required must enumerate what a graduate "
    "student still has to do: implementation, running experiments, manual analysis, "
    "data collection, qualitative case studies.\n"
    "- Each related_work entry MUST cite a paper_id from the input prior_art or "
    "evidence. Do not invent paper IDs.\n"
    "- Be specific. Name techniques, datasets, equations. Vague academic prose "
    "('our method shows promise') defeats the purpose.\n"
    "- Tone: scholarly, neutral, hedge claims appropriately."
)


def _build_user_prompt(
    idea: dict, evidence: list[dict], prior_art: list[dict] | None,
) -> str:
    payload = {
        "idea": {
            "title": idea.get("title", ""),
            "research_question": idea.get("research_question", ""),
            "method_sketch": idea.get("method_sketch", ""),
            "evaluation_plan": idea.get("evaluation_plan", ""),
            "expected_contribution": idea.get("expected_contribution", ""),
            "assumptions_and_risks": idea.get("assumptions_and_risks", ""),
            "themes": [
                t for t in [idea.get("label_a"), idea.get("label_b")] if t
            ],
        },
        "evidence": evidence or [],
        "prior_art": prior_art or [],
    }
    return (
        "Draft a full paper plan for the following idea.\n"
        "Return JSON matching the schema. Remember: NO fabricated numbers, NO "
        "fabricated citations.\n\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )


@retry(tries=3, base_delay=2.0)
def _call_drafter(client: OpenAI, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "paper_draft", "schema": PAPER_DRAFT_SCHEMA, "strict": True,
            },
        },
        temperature=0.4,
    )
    return parse_json_response(resp.choices[0].message.content)


def draft_full_paper(
    idea: dict,
    evidence: list[dict] | None = None,
    prior_art: list[dict] | None = None,
    *,
    model: str = DEFAULT_DRAFT_MODEL,
) -> dict:
    """Expand an idea into a full paper draft.

    Args:
      idea:      the flat idea dict (from ideas.tsv) — must have title,
                 research_question, method_sketch, evaluation_plan,
                 expected_contribution, assumptions_and_risks.
      evidence:  list of {paper_id, gap_sentence, paragraph_text, ...} that
                 motivated the idea. Defaults to parsing idea['evidence_used_json'].
      prior_art: list of {paperId|paper_id, title, abstract} from a Semantic
                 Scholar (or similar) search. Used to populate related_work.
                 Can be empty — related_work will then only cite evidence papers.
      model:     OpenRouter slug.

    Returns:
      Dict matching PAPER_DRAFT_SCHEMA.
    """
    if evidence is None:
        # Best effort: parse evidence_used_json off the idea row.
        raw = idea.get("evidence_used_json")
        if isinstance(raw, str) and raw.strip():
            try:
                evidence = json.loads(raw)
            except json.JSONDecodeError:
                evidence = []
        elif isinstance(raw, list):
            evidence = raw
        else:
            evidence = []

    client = get_llm_client()
    prompt = _build_user_prompt(idea, evidence, prior_art)
    log.info("Drafting full paper for: %s", (idea.get("title") or "")[:60])
    data = _call_drafter(client, prompt, model=model)

    # Defence in depth: enforce that related_work only references known paper_ids.
    known_ids = {str(e.get("paper_id", "")) for e in (evidence or [])}
    known_ids |= {str(p.get("paperId") or p.get("paper_id") or "") for p in (prior_art or [])}
    known_ids.discard("")
    if known_ids:
        cleaned = [rw for rw in data.get("related_work", []) if str(rw.get("paper_id", "")) in known_ids]
        if len(cleaned) != len(data.get("related_work", [])):
            log.warning(
                "Dropped %d related-work entries that cited unknown paper_ids",
                len(data.get("related_work", [])) - len(cleaned),
            )
        data["related_work"] = cleaned

    return data
