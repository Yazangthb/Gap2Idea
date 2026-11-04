"""Single source of truth for LLM client construction.

We use the OpenAI Python SDK pointed at an OpenAI-compatible endpoint. Two
providers are supported and chosen by the `LLM_PROVIDER` env var (or auto-
detected — see `active_provider`):

  openrouter  any model on OpenRouter (OpenAI, Anthropic, Google, Meta, …)
              via `<provider>/<model>` slugs, e.g. openai/gpt-4.1-mini.
  yandex      Yandex Cloud Foundation Models (YandexGPT). Yandex uses model
              URIs of the form `gpt://<folder-id>/yandexgpt/latest`, but the
              rest of the codebase keeps passing OpenRouter-style slugs — the
              Yandex client transparently rewrites them (see `_YandexClient`),
              so no call-site changes when you switch providers. YandexGPT's
              OpenAI-compatible endpoint supports strict `json_schema` /
              `json_object` response formats, which we rely on everywhere.

Environment variables read:
  LLM_PROVIDER         (optional) "yandex" | "openrouter". If unset, we pick
                       "yandex" when YANDEX_API_KEY+YANDEX_FOLDER_ID are set,
                       else "openrouter".
  OpenRouter:
    OPENROUTER_API_KEY   (required for openrouter)
    OPENROUTER_BASE_URL  (optional, default https://openrouter.ai/api/v1)
    OPENROUTER_REFERRER  (optional, sent as HTTP-Referer header)
    OPENROUTER_TITLE     (optional, sent as X-Title header)
  Yandex:
    YANDEX_API_KEY       (required for yandex) — a Cloud API key (sent as a
                         Bearer token; the SDK's default auth works).
    YANDEX_FOLDER_ID     (required for yandex) — used to build the model URI.
    YANDEX_MODEL         (optional, default "yandexgpt") — the variant to map
                         every slug to: "yandexgpt" (Pro) or "yandexgpt-lite".
    YANDEX_BASE_URL      (optional, default https://llm.api.cloud.yandex.net/v1)
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
# generation or to A/B against another provider. Under the Yandex provider this
# (and every other slug) is rewritten to a Yandex model URI automatically.
DEFAULT_MODEL = "openai/gpt-4.1-mini"

# The judge default. Set differently to mitigate self-evaluation bias when
# generator and judge would otherwise be the same model. NB: under a single
# provider (e.g. Yandex-only) generator and judge collapse to the same model —
# keep an OpenRouter key configured for cross-provider judge panels.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-4"

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/v1"


def active_provider() -> str:
    """Which LLM provider is in effect: 'yandex' or 'openrouter'.

    `LLM_PROVIDER` wins if set; otherwise we default to Yandex when its
    credentials are present, else OpenRouter.
    """
    p = os.getenv("LLM_PROVIDER")
    if p:
        return p.strip().lower()
    if os.getenv("YANDEX_API_KEY") and os.getenv("YANDEX_FOLDER_ID"):
        return "yandex"
    return "openrouter"


# ----------------------------------------------------------------------
# Yandex: model-rewriting proxy over the OpenAI SDK
# ----------------------------------------------------------------------
# Yandex model URIs look like gpt://<folder-id>/yandexgpt/latest. The rest of
# the codebase passes OpenRouter-style slugs ("openai/gpt-4.1-mini"), CLI
# --model defaults, and judge panels. Rather than touch every call-site, we wrap
# the OpenAI client so chat.completions.create() rewrites `model` on the way
# through. Any slug that isn't already a gpt:// URI maps to the configured
# variant; an explicit gpt:// URI passes through untouched.

class _YandexCompletions:
    def __init__(self, completions, parent: "_YandexClient"):
        self._completions = completions
        self._parent = parent

    def create(self, *args, **kwargs):
        if "model" in kwargs:
            kwargs["model"] = self._parent.to_model(kwargs["model"])
        return self._completions.create(*args, **kwargs)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._completions, name)


class _YandexChat:
    def __init__(self, chat, parent: "_YandexClient"):
        self._chat = chat
        self.completions = _YandexCompletions(chat.completions, parent)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._chat, name)


class _YandexClient:
    """OpenAI-SDK client for Yandex Cloud that rewrites model slugs to Yandex
    URIs. All attributes other than `chat` proxy through to the real client."""

    def __init__(self, client: OpenAI, folder_id: str, variant: str):
        self._client = client
        self._folder_id = folder_id
        self._variant = variant
        self.chat = _YandexChat(client.chat, self)

    def to_model(self, model: str | None) -> str:
        if model and model.startswith("gpt://"):
            return model
        return f"gpt://{self._folder_id}/{self._variant}/latest"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._client, name)


def _make_yandex_client() -> _YandexClient:
    api_key = os.getenv("YANDEX_API_KEY")
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    if not api_key or not folder_id:
        raise RuntimeError(
            "LLM_PROVIDER=yandex but YANDEX_API_KEY / YANDEX_FOLDER_ID are not set. "
            "Add both to your .env (get them in the Yandex Cloud console)."
        )
    variant = os.getenv("YANDEX_MODEL", "yandexgpt").strip()
    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("YANDEX_BASE_URL", YANDEX_BASE_URL),
    )
    return _YandexClient(client, folder_id, variant)


def _make_openrouter_client() -> OpenAI:
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


def get_llm_client(provider: str | None = None):
    """Return an OpenAI-SDK-compatible client for the active provider.

    `provider` overrides `active_provider()` for a single call (e.g. force
    OpenRouter for a cross-provider judge panel even when Yandex is the default).
    Raises RuntimeError with an actionable message if the chosen provider's
    credentials are missing — fail fast rather than 401 after retries.
    """
    provider = (provider or active_provider()).lower()
    if provider == "yandex":
        return _make_yandex_client()
    return _make_openrouter_client()


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
