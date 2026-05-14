"""Single source of truth for LLM client construction.

We use the OpenAI Python SDK pointed at OpenRouter's OpenAI-compatible
endpoint. This lets us call any model on OpenRouter (OpenAI, Anthropic,
Google, Meta, …) without rewriting our code per provider — just change
the `model=` string.

Environment variables read:
  OPENROUTER_API_KEY   (required)
  OPENROUTER_BASE_URL  (optional, default https://openrouter.ai/api/v1)
  OPENROUTER_REFERRER  (optional, sent as HTTP-Referer header)
  OPENROUTER_TITLE     (optional, sent as X-Title header — shows on the
                        OpenRouter leaderboard if you opt in)

Model-name convention on OpenRouter:
  <provider>/<model>            e.g. openai/gpt-4.1-mini
                                     anthropic/claude-sonnet-4
                                     google/gemini-2.5-flash
                                     meta-llama/llama-3.3-70b-instruct
"""
from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# The default workhorse: cheap, fast, supports strict JSON-schema response
# format (which we rely on everywhere). Change at the call-site for stronger
# generation or to A/B against another provider.
DEFAULT_MODEL = "openai/gpt-4.1-mini"

# The judge default. Set differently to mitigate self-evaluation bias when
# generator and judge would otherwise be the same model.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def get_llm_client() -> OpenAI:
    """Return an OpenAI-SDK client wired to OpenRouter.

    Raises RuntimeError if `OPENROUTER_API_KEY` is missing — fail fast
    rather than letting the SDK return cryptic 401s after retries.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Fallback: allow OPENAI_API_KEY for backwards compat / direct OpenAI use.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys "
                "and add it to your .env file."
            )

    headers: dict[str, str] = {}
    if referrer := os.getenv("OPENROUTER_REFERRER"):
        headers["HTTP-Referer"] = referrer
    if title := os.getenv("OPENROUTER_TITLE"):
        headers["X-Title"] = title

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENROUTER_BASE_URL", DEFAULT_BASE_URL),
        default_headers=headers or None,
    )


# ----------------------------------------------------------------------
# Robust JSON extraction
# ----------------------------------------------------------------------
# OpenAI models honour `response_format={"type": "json_schema", "strict": True}`
# and return raw JSON. Anthropic / Google / open-source models routed through
# OpenRouter often ignore the strict flag and return JSON wrapped in markdown
# code fences, prose, or both. We accept all three.

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json_response(content: str | None) -> dict:
    """Extract a JSON object from an LLM response. Raises ValueError if none.

    Tries, in order:
      1. Direct json.loads (OpenAI strict mode)
      2. Strip ```json ... ``` or ``` ... ``` markdown fences
      3. Grab the largest {...} substring and parse it
    """
    if not content or not content.strip():
        raise ValueError("LLM response was empty")
    text = content.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) markdown fence
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3) bare object — take the largest {...} balance-correct substring
    m = _BARE_OBJECT_RE.search(text)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # try truncating from the right until it parses
            for i in range(len(candidate), 0, -1):
                if candidate[i - 1] == "}":
                    try:
                        return json.loads(candidate[:i])
                    except json.JSONDecodeError:
                        continue

    raise ValueError(
        f"Could not extract JSON from LLM response. First 300 chars: {text[:300]!r}"
    )
