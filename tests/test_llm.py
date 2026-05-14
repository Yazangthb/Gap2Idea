"""Tests for the OpenRouter client factory and JSON parsing helper."""
from __future__ import annotations

import importlib

import pytest

from gap2idea.pipeline.llm import parse_json_response


@pytest.fixture
def fresh_llm(monkeypatch):
    """Reload gap2idea.pipeline.llm after monkeypatching env vars, so the
    `load_dotenv()` at import time doesn't override our overrides.
    """
    def _reload():
        import gap2idea.pipeline.llm as mod
        return importlib.reload(mod)
    return _reload


def test_missing_key_raises(monkeypatch, fresh_llm):
    # The module reload calls load_dotenv() which would repopulate the env
    # from .env. Reload first, then strip keys before calling get_llm_client.
    mod = fresh_llm()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        mod.get_llm_client()


def test_uses_openrouter_when_key_set(monkeypatch, fresh_llm):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    mod = fresh_llm()
    client = mod.get_llm_client()
    # The SDK exposes base_url as a httpx URL; cast to str for the assertion.
    assert "openrouter.ai" in str(client.base_url)
    assert client.api_key == "test-key-1"


def test_falls_back_to_openai_key(monkeypatch, fresh_llm):
    # Reload first so load_dotenv() runs with the real .env, then strip the
    # OpenRouter key + plant a legacy OpenAI key — exercises the fallback path.
    mod = fresh_llm()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-key")
    client = mod.get_llm_client()
    assert client.api_key == "legacy-key"


def test_custom_base_url(monkeypatch, fresh_llm):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.test/v1")
    mod = fresh_llm()
    client = mod.get_llm_client()
    assert "example.test" in str(client.base_url)


def test_attribution_headers_when_set(monkeypatch, fresh_llm):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("OPENROUTER_REFERRER", "https://gap2idea.example")
    monkeypatch.setenv("OPENROUTER_TITLE", "Gap2Idea-test")
    mod = fresh_llm()
    client = mod.get_llm_client()
    # default_headers on the SDK client surfaces in client.default_headers (mapping).
    headers = dict(client.default_headers or {})
    assert headers.get("HTTP-Referer") == "https://gap2idea.example"
    assert headers.get("X-Title") == "Gap2Idea-test"


def test_default_model_constants(fresh_llm):
    mod = fresh_llm()
    assert "/" in mod.DEFAULT_MODEL  # provider/model format
    assert "/" in mod.DEFAULT_JUDGE_MODEL
    # generator and judge default to different providers for cross-eval
    assert mod.DEFAULT_MODEL.split("/")[0] != mod.DEFAULT_JUDGE_MODEL.split("/")[0]


# ---------- parse_json_response ----------

def test_parse_json_direct():
    assert parse_json_response('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_json_strips_markdown_fence():
    """Regression: Anthropic via OpenRouter wraps JSON in ```json ... ``` blocks."""
    raw = '```json\n{"score": 3, "why": "ok"}\n```'
    assert parse_json_response(raw) == {"score": 3, "why": "ok"}


def test_parse_json_strips_plain_fence():
    raw = '```\n{"score": 3}\n```'
    assert parse_json_response(raw) == {"score": 3}


def test_parse_json_extracts_bare_object_from_prose():
    raw = 'Sure! Here is the JSON:\n{"a": 1, "b": [2, 3]}\nLet me know if you need more.'
    assert parse_json_response(raw) == {"a": 1, "b": [2, 3]}


def test_parse_json_handles_nested_braces():
    raw = '```json\n{"outer": {"inner": [1, 2, 3]}, "x": "y"}\n```'
    out = parse_json_response(raw)
    assert out["outer"]["inner"] == [1, 2, 3]
    assert out["x"] == "y"


def test_parse_json_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        parse_json_response("")
    with pytest.raises(ValueError, match="empty"):
        parse_json_response(None)


def test_parse_json_unparseable_raises():
    with pytest.raises(ValueError, match="Could not extract JSON"):
        parse_json_response("This response has no JSON anywhere in it whatsoever.")
