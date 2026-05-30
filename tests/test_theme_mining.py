"""Unit tests for theme_mining: bridge score, clean, summaries, pairs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gap2idea.pipeline.theme_mining import (
    _peak,
    _type_complementarity,
    _type_dist,
    build_cluster_pairs,
    build_cluster_summary,
    clean_gaps,
    keyword_label,
)


# ---------- peak ----------

def test_peak_zero_at_boundaries():
    assert _peak(0.0) == 0.0
    assert _peak(1.0) == 0.0


def test_peak_one_at_peak():
    assert _peak(0.45) == 1.0


def test_peak_monotone_each_side():
    assert _peak(0.10) < _peak(0.30) < _peak(0.45)
    assert _peak(0.45) > _peak(0.60) > _peak(0.90)


# ---------- type_complementarity ----------

def test_type_complementarity_identical_is_zero():
    d = {"limitation": 0.6, "future_work": 0.4}
    assert _type_complementarity(d, d) == 0.0


def test_type_complementarity_orthogonal_is_one():
    d1 = {"limitation": 1.0}
    d2 = {"future_work": 1.0}
    assert _type_complementarity(d1, d2) == 1.0


def test_type_dist_empty():
    assert _type_dist([]) == {}


# ---------- clean_gaps ----------

def test_clean_gaps_filters_short_and_low_conf():
    df = pd.DataFrame(
        [
            {"id": "p1", "gap_sentence": "a" * 25, "paragraph_text": "blah",
             "confidence": 0.9, "gap_type": "future_work"},
            {"id": "p2", "gap_sentence": "short", "paragraph_text": "blah",
             "confidence": 0.9, "gap_type": "future_work"},
            {"id": "p3", "gap_sentence": "y" * 25, "paragraph_text": "blah",
             "confidence": 0.2, "gap_type": "future_work"},
            {"id": "p1", "gap_sentence": "a" * 25, "paragraph_text": "blah",
             "confidence": 0.9, "gap_type": "future_work"},  # dup
        ]
    )
    out = clean_gaps(df, min_conf=0.5)
    assert len(out) == 1
    assert out.iloc[0]["id"] == "p1"


def test_clean_gaps_round_trips_section_type():
    """section_type must survive cleaning so downstream graph/critic stages
    can use it as an edge feature / context field."""
    df = pd.DataFrame(
        [
            {"id": "p1", "gap_sentence": "a" * 25, "paragraph_text": "blah",
             "confidence": 0.9, "gap_type": "future_work",
             "section_type": "future_work"},
            {"id": "p2", "gap_sentence": "b" * 25, "paragraph_text": "blah",
             "confidence": 0.9, "gap_type": "limitation",
             "section_type": "limitations"},
        ]
    )
    out = clean_gaps(df, min_conf=0.5)
    assert "section_type" in out.columns
    assert set(out["section_type"]) == {"future_work", "limitations"}


def test_clean_gaps_tolerates_missing_section_type_column():
    """Older gaps.tsv files written before PR-1 lack section_type. clean_gaps
    must default to "" so reading them doesn't crash."""
    df = pd.DataFrame(
        [
            {"id": "p1", "gap_sentence": "a" * 25, "paragraph_text": "blah",
             "confidence": 0.9, "gap_type": "future_work"},
        ]
    )
    out = clean_gaps(df, min_conf=0.5)
    assert "section_type" in out.columns
    assert out.iloc[0]["section_type"] == ""


# ---------- keyword_label ----------

def test_keyword_label_returns_something():
    sents = [
        "Graph neural networks struggle with long-range dependencies.",
        "Long-range message passing in GNNs needs further study.",
        "We do not address long-range structural patterns in GNNs.",
    ]
    label = keyword_label(sents, k=5)
    assert isinstance(label, str)
    assert "," in label  # multi-term output


def test_keyword_label_handles_singleton():
    assert keyword_label(["only one sentence"]) == ""


# ---------- build_cluster_pairs ----------

def _toy_gaps_and_embeddings():
    """Two clusters with very different gap_types and zero paper overlap."""
    rows = []
    # cluster 0: 3 papers, type=limitation
    for i, pid in enumerate(["p1", "p2", "p3"]):
        rows.append(
            {"id": pid, "gap_sentence": "lim " * 10 + str(i),
             "paragraph_text": "", "confidence": 0.9,
             "gap_type": "limitation", "cluster_id": 0}
        )
    # cluster 1: 3 different papers, type=future_work
    for i, pid in enumerate(["q1", "q2", "q3"]):
        rows.append(
            {"id": pid, "gap_sentence": "fw " * 10 + str(i),
             "paragraph_text": "", "confidence": 0.9,
             "gap_type": "future_work", "cluster_id": 1}
        )
    df = pd.DataFrame(rows)
    # Build cluster-separated embeddings at moderate similarity (sim ~ 0.45)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 8))
    b = rng.normal(size=(3, 8))
    # Pull b half-way toward a's centroid so their centroids cosine-align ~ 0.45
    a_c = a.mean(axis=0)
    b_c = b.mean(axis=0)
    target = 0.55 * a_c + 0.45 * b_c
    b = b - b_c + target
    X = np.vstack([a, b])
    return df, X


def test_build_cluster_pairs_columns_and_bridge_in_unit_interval():
    df, X = _toy_gaps_and_embeddings()
    labels = {0: "Limitations", 1: "Future"}
    pairs = build_cluster_pairs(df, X, labels, top_n=10)
    assert {"cluster_a", "cluster_b", "cosine_sim", "paper_overlap",
            "type_complementarity", "bridge_score"} <= set(pairs.columns)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert 0.0 <= row["bridge_score"] <= 1.0
    assert row["paper_overlap"] == 0.0
    assert row["type_complementarity"] == 1.0


def test_build_cluster_pairs_returns_empty_for_one_cluster():
    df = pd.DataFrame(
        [
            {"id": "p1", "gap_sentence": "x" * 25, "paragraph_text": "",
             "confidence": 0.9, "gap_type": "limitation", "cluster_id": 0},
            {"id": "p2", "gap_sentence": "y" * 25, "paragraph_text": "",
             "confidence": 0.9, "gap_type": "limitation", "cluster_id": 0},
        ]
    )
    X = np.random.default_rng(0).normal(size=(2, 4))
    out = build_cluster_pairs(df, X, {0: "L"}, top_n=5)
    assert out.empty


def test_build_cluster_summary_columns():
    df, _ = _toy_gaps_and_embeddings()
    summary = build_cluster_summary(df, {0: "Lim", 1: "FW"})
    assert {"cluster_id", "n_items", "n_papers", "avg_conf", "theme_label"} <= set(summary.columns)
    assert set(summary["theme_label"].tolist()) == {"Lim", "FW"}
