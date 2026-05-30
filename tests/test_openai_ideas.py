"""Unit tests for non-API helpers in openai_ideas.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gap2idea.pipeline.openai_ideas import (
    IDEA_SCHEMA,
    _build_method_gap_prompt,
    _build_user_prompt,
    _build_within_prompt,
    _cosine,
    _diverse_evidence,
    _evidence_payload,
)


def test_cosine_orthogonal_and_parallel():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    c = np.array([2.0, 0.0, 0.0])
    assert abs(_cosine(a, b)) < 1e-9
    assert abs(_cosine(a, c) - 1.0) < 1e-9


def test_cosine_zero_vector_safe():
    assert _cosine(np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])) == 0.0


def _gap(pid, conf, sent, cid=0):
    return {
        "id": pid, "gap_type": "future_work", "confidence": conf,
        "gap_sentence": sent, "paragraph_text": sent, "cluster_id": cid,
    }


def test_diverse_evidence_picks_varied_sentences():
    df = pd.DataFrame([
        _gap("p1", 0.95, "graph neural networks struggle with long-range dependencies"),
        _gap("p2", 0.90, "graph neural networks struggle with long-range dependencies tomorrow"),
        _gap("p3", 0.80, "evaluation protocols for continual learning are inconsistent across benchmarks"),
        _gap("p4", 0.50, "OOD detection methods require labelled validation sets"),
    ])
    picked = _diverse_evidence(df, cluster_id=0, k=2)
    sents = " | ".join(picked["gap_sentence"].tolist())
    # The top-confidence row is always picked first; the second must come from
    # a different topic, not the near-duplicate paraphrase.
    assert "graph neural" in sents
    assert ("continual learning" in sents) or ("OOD" in sents)


def test_diverse_evidence_empty_cluster_returns_empty():
    df = pd.DataFrame(columns=["id", "gap_sentence", "paragraph_text", "confidence", "gap_type", "cluster_id"])
    out = _diverse_evidence(df, cluster_id=5, k=3)
    assert out.empty


def test_diverse_evidence_k_larger_than_n_returns_all():
    df = pd.DataFrame([
        _gap("p1", 0.9, "a one"),
        _gap("p2", 0.8, "b two"),
    ])
    out = _diverse_evidence(df, 0, k=10)
    assert len(out) == 2


def test_evidence_payload_shape():
    df = pd.DataFrame([_gap("p1", 0.9, "abc")])
    payload = _evidence_payload(df)
    assert isinstance(payload, list) and len(payload) == 1
    assert {"paper_id", "gap_type", "confidence", "gap_sentence", "paragraph_text"} <= set(payload[0])


def test_build_user_prompt_includes_payload():
    p = _build_user_prompt(1, 2, "A", "B", [{"paper_id": "x"}], [{"paper_id": "y"}])
    assert '"theme_a_label": "A"' in p
    assert '"theme_b_label": "B"' in p
    assert "schema" in p


# ---------- PR-2: falsifiability + named-baseline schema and prompts ----------

def test_idea_schema_has_falsifiable_prediction_and_named_baseline():
    idea_props = IDEA_SCHEMA["properties"]["idea"]["properties"]
    assert "falsifiable_prediction" in idea_props
    assert "named_baseline" in idea_props
    required = IDEA_SCHEMA["properties"]["idea"]["required"]
    assert "falsifiable_prediction" in required
    assert "named_baseline" in required


def test_bridge_prompt_mentions_falsifiable_prediction_and_named_baseline():
    p = _build_user_prompt(1, 2, "A", "B", [{"paper_id": "x"}], [{"paper_id": "y"}])
    assert "falsifiable_prediction" in p
    assert "named_baseline" in p


def test_within_prompt_mentions_falsifiable_prediction_and_named_baseline():
    p = _build_within_prompt(1, "A", [{"paper_id": "x"}])
    assert "falsifiable_prediction" in p
    assert "named_baseline" in p


def test_method_gap_prompt_mentions_falsifiable_prediction_and_named_baseline():
    p = _build_method_gap_prompt(1, "A", [{"paper_id": "x"}], [{"paper_id": "y"}])
    assert "falsifiable_prediction" in p
    assert "named_baseline" in p
