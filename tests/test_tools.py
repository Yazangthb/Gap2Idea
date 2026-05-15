"""Unit tests for tools.py (the shared tool surface).

All tests are async-aware via pytest-asyncio. We stub out the corpus
state so they're hermetic — no real artifacts directory needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gap2idea import tools as T


@pytest.fixture
def fake_corpus(tmp_path: Path, monkeypatch):
    """Inject a fake CorpusHandle into tools._CORPUS so all tools work offline."""
    gaps = pd.DataFrame([
        {"id": "p1", "gap_type": "future_work", "gap_sentence": "study X under noise",
         "paragraph_text": "para", "confidence": 0.9, "cluster_id": 0},
        {"id": "p2", "gap_type": "limitation",  "gap_sentence": "lacks evaluation on Y",
         "paragraph_text": "para", "confidence": 0.85, "cluster_id": 0},
        {"id": "p3", "gap_type": "future_work", "gap_sentence": "extend to multilingual setup",
         "paragraph_text": "para", "confidence": 0.95, "cluster_id": 1},
    ])
    clusters = pd.DataFrame([
        {"cluster_id": 0, "theme_label": "Robustness Under Noise", "n_items": 2, "n_papers": 2, "avg_conf": 0.875},
        {"cluster_id": 1, "theme_label": "Multilingual Generalisation", "n_items": 1, "n_papers": 1, "avg_conf": 0.95},
    ])
    labels = pd.DataFrame([
        {"cluster_id": 0, "theme_label": "Robustness Under Noise", "keywords": "noise, robust, study"},
        {"cluster_id": 1, "theme_label": "Multilingual Generalisation", "keywords": "multilingual, extend"},
    ])
    pairs = pd.DataFrame([
        {"cluster_a": 0, "cluster_b": 1, "label_a": "Robustness Under Noise",
         "label_b": "Multilingual Generalisation", "bridge_score": 0.62,
         "cosine_sim": 0.45, "paper_overlap": 0.0, "type_complementarity": 0.5},
    ])
    methods = pd.DataFrame([
        {"id": "p4", "method_type": "algorithm",
         "method_sentence": "We propose a denoising layer for transformers.",
         "paragraph_text": "para", "confidence": 0.92},
    ])
    rng = np.random.default_rng(0)
    gap_emb = rng.normal(size=(3, 8))
    gap_emb /= np.linalg.norm(gap_emb, axis=1, keepdims=True)
    method_emb = rng.normal(size=(1, 8))
    method_emb /= np.linalg.norm(method_emb, axis=1, keepdims=True)

    handle = T.CorpusHandle(
        gaps=gaps, clusters=clusters, labels=labels, pairs=pairs,
        embeddings=gap_emb, methods=methods, method_embeddings=method_emb,
        ideas_path=tmp_path / "ideas.tsv",
        root=tmp_path,
    )
    monkeypatch.setattr(T, "_CORPUS", handle)
    return handle


# ---------- list_themes ----------

@pytest.mark.asyncio
async def test_list_themes(fake_corpus):
    out = await T.list_themes()
    assert len(out) == 2
    assert out[0]["theme_label"] == "Robustness Under Noise"
    assert out[0]["n_papers"] == 2


# ---------- get_theme ----------

@pytest.mark.asyncio
async def test_get_theme_known(fake_corpus):
    theme = await T.get_theme(0)
    assert theme["cluster_id"] == 0
    assert theme["theme_label"] == "Robustness Under Noise"
    assert theme["n_papers"] == 2
    assert len(theme["papers"]) == 2
    # Each paper has its gaps
    paper_ids = {p["paper_id"] for p in theme["papers"]}
    assert paper_ids == {"p1", "p2"}


@pytest.mark.asyncio
async def test_get_theme_unknown(fake_corpus):
    out = await T.get_theme(99)
    assert "error" in out


# ---------- get_evidence ----------

@pytest.mark.asyncio
async def test_get_evidence_returns_k_rows(fake_corpus):
    ev = await T.get_evidence(0, k=2)
    assert len(ev) == 2
    assert {e["paper_id"] for e in ev} <= {"p1", "p2"}
    assert all("gap_sentence" in e for e in ev)


# ---------- check_evidence_overlap ----------

@pytest.mark.asyncio
async def test_check_evidence_overlap_full():
    fed = [{"paper_id": "p1", "gap_sentence": "abc"}]
    used = [{"paper_id": "p1", "gap_sentence": "abc"}]
    out = await T.check_evidence_overlap(used, fed)
    assert out["overlap"] == 1.0
    assert out["n_grounded"] == 1
    assert out["hallucinated"] == []


@pytest.mark.asyncio
async def test_check_evidence_overlap_partial():
    fed = [{"paper_id": "p1", "gap_sentence": "abc"}]
    used = [
        {"paper_id": "p1", "gap_sentence": "abc"},
        {"paper_id": "made_up", "gap_sentence": "hallucinated"},
    ]
    out = await T.check_evidence_overlap(used, fed)
    assert out["overlap"] == 0.5
    assert out["n_grounded"] == 1
    assert len(out["hallucinated"]) == 1


# ---------- list_ideas / get_idea / save_idea ----------

@pytest.mark.asyncio
async def test_save_then_list_then_get(fake_corpus):
    idea = {
        "title": "Test idea", "research_question": "?", "method_sketch": ".",
        "evaluation_plan": ".", "expected_contribution": ".",
        "assumptions_and_risks": ".", "idea_confidence": 0.7,
        "novelty_score": 0.8, "mode": "within",
    }
    res = await T.save_idea(idea)
    assert res["saved"]
    assert res["total_ideas"] == 1

    lst = await T.list_ideas()
    assert len(lst) == 1
    assert lst[0]["title"] == "Test idea"

    one = await T.get_idea(0)
    assert one["title"] == "Test idea"


@pytest.mark.asyncio
async def test_list_ideas_filters_by_novelty(fake_corpus):
    await T.save_idea({"title": "low_nov", "novelty_score": 0.1, "idea_confidence": 0.9, "mode": "within"})
    await T.save_idea({"title": "hi_nov",  "novelty_score": 0.9, "idea_confidence": 0.9, "mode": "within"})

    lst = await T.list_ideas(min_novelty=0.5)
    titles = {r["title"] for r in lst}
    assert "hi_nov" in titles
    assert "low_nov" not in titles


@pytest.mark.asyncio
async def test_get_idea_out_of_range(fake_corpus):
    out = await T.get_idea(99)
    assert "error" in out


# ---------- dispatch ----------

@pytest.mark.asyncio
async def test_dispatch_invokes_correct_tool(fake_corpus):
    res = await T.dispatch("list_themes", {})
    assert isinstance(res, list)
    assert len(res) == 2


@pytest.mark.asyncio
async def test_dispatch_rejects_unknown_tool():
    with pytest.raises(ValueError, match="unknown tool"):
        await T.dispatch("frobnicate")


# ---------- tool definitions are well-formed ----------

def test_tool_definitions_have_required_keys():
    for tool in T.TOOL_DEFINITIONS:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool
        assert tool["inputSchema"].get("type") == "object"


def test_every_defined_tool_has_an_implementation():
    for tool in T.TOOL_DEFINITIONS:
        name = tool["name"]
        assert name in T.TOOL_FNS, f"Tool {name} declared but not implemented"
