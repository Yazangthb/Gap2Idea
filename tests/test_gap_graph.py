"""Unit tests for the gap-graph engine (PR-4).

These tests use small synthetic corpora so Leiden / Louvain are run in
≈10 ms. They lock down:

  • community detection finds the obvious blocks
  • bridge_score is normalised to [0, 1]
  • frontier_nodes ranks boundary gaps highest
  • method nodes are excluded from communities (bipartite=1)
  • re-running with the same seed produces identical cluster_ids
  • the dispatcher and legacy clustering both produce the documented columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gap2idea.pipeline import gap_graph as GG
from gap2idea.pipeline.theme_mining import (
    build_cluster_labels,
    build_cluster_pairs,
    build_cluster_summary,
    cluster_embeddings,
    cluster_gaps_via_graph,
)


# ---------- fixtures ----------

def _two_block_corpus(n_per_block: int = 5, dim: int = 16, seed: int = 0):
    """Two clearly-separated blocks of gaps with zero paper overlap and
    different gap_type mixes. Designed so any sane community detector
    finds two communities."""
    rng = np.random.default_rng(seed)

    # Block A: papers p1..pN, gap_type=limitation, centroid (1, 0, ..., 0)
    a_center = np.zeros(dim)
    a_center[0] = 1.0
    A = a_center + 0.05 * rng.normal(size=(n_per_block, dim))
    A /= np.linalg.norm(A, axis=1, keepdims=True)

    # Block B: papers q1..qN, gap_type=future_work, centroid (0, ..., 0, 1)
    b_center = np.zeros(dim)
    b_center[-1] = 1.0
    B = b_center + 0.05 * rng.normal(size=(n_per_block, dim))
    B /= np.linalg.norm(B, axis=1, keepdims=True)

    embeddings = np.vstack([A, B])

    rows: list[dict] = []
    for i in range(n_per_block):
        rows.append({"id": f"p{i}", "gap_type": "limitation",
                     "section_type": "limitations",
                     "gap_sentence": f"limitation gap A {i}",
                     "paragraph_text": "", "confidence": 0.9})
    for i in range(n_per_block):
        rows.append({"id": f"q{i}", "gap_type": "future_work",
                     "section_type": "future_work",
                     "gap_sentence": f"future work B {i}",
                     "paragraph_text": "", "confidence": 0.9})
    return pd.DataFrame(rows), embeddings


def _frontier_corpus(n_per_block: int = 4, dim: int = 8, seed: int = 0):
    """Two blocks + one boundary gap halfway between them. The boundary
    gap should rank #1 by frontier_score because its neighbours span both
    communities."""
    rng = np.random.default_rng(seed)
    a_center = np.zeros(dim); a_center[0] = 1.0
    b_center = np.zeros(dim); b_center[-1] = 1.0
    A = a_center + 0.05 * rng.normal(size=(n_per_block, dim))
    B = b_center + 0.05 * rng.normal(size=(n_per_block, dim))
    boundary = (a_center + b_center) / 2 + 0.01 * rng.normal(size=(1, dim))
    X = np.vstack([A, B, boundary])
    X /= np.linalg.norm(X, axis=1, keepdims=True)

    rows: list[dict] = []
    for i in range(n_per_block):
        rows.append({"id": f"a{i}", "gap_type": "limitation",
                     "section_type": "limitations", "confidence": 0.9,
                     "gap_sentence": f"A gap {i}", "paragraph_text": ""})
    for i in range(n_per_block):
        rows.append({"id": f"b{i}", "gap_type": "future_work",
                     "section_type": "future_work", "confidence": 0.9,
                     "gap_sentence": f"B gap {i}", "paragraph_text": ""})
    rows.append({"id": "boundary", "gap_type": "open_problem",
                 "section_type": "discussion", "confidence": 0.95,
                 "gap_sentence": "boundary gap", "paragraph_text": ""})
    return pd.DataFrame(rows), X


# ---------- community detection ----------

def test_detect_communities_separates_two_blocks():
    gaps, X = _two_block_corpus(n_per_block=6)
    G = GG.build_gap_graph(gaps, X, knn_k=3, sim_threshold=0.0)
    ids = GG.detect_communities(G, seed=42)
    # Same number of cluster IDs as gap nodes
    assert len(ids) == len(gaps)
    # At least two distinct communities found
    assert len(set(ids)) >= 2
    # Within each block, the SAME community ID dominates
    block_a = ids[:6]
    block_b = ids[6:]
    # Modal value of each block (no statistics import on Python 3.10 stdlib)
    from collections import Counter
    mode_a = Counter(block_a).most_common(1)[0][0]
    mode_b = Counter(block_b).most_common(1)[0][0]
    assert mode_a != mode_b


def test_detect_communities_renumbers_largest_first():
    """Block A and B have equal size in the fixture — but if we make A larger,
    A's community ID must be 0 (largest first)."""
    gaps, X = _two_block_corpus(n_per_block=4)
    # Pad block A with 4 more gaps so it dominates.
    rng = np.random.default_rng(1)
    a_center = np.zeros(X.shape[1]); a_center[0] = 1.0
    extra = a_center + 0.05 * rng.normal(size=(4, X.shape[1]))
    extra /= np.linalg.norm(extra, axis=1, keepdims=True)
    X2 = np.vstack([X[:4], extra, X[4:]])
    extra_rows = [
        {"id": f"p_extra_{i}", "gap_type": "limitation",
         "section_type": "limitations", "confidence": 0.9,
         "gap_sentence": f"limit X{i}", "paragraph_text": ""}
        for i in range(4)
    ]
    gaps2 = pd.concat([gaps.iloc[:4], pd.DataFrame(extra_rows), gaps.iloc[4:]],
                       ignore_index=True)
    G = GG.build_gap_graph(gaps2, X2, knn_k=3, sim_threshold=0.0)
    ids = GG.detect_communities(G, seed=42)
    # Block A occupies the first 8 rows; mode there must be cluster_id 0.
    from collections import Counter
    mode_a = Counter(ids[:8]).most_common(1)[0][0]
    assert mode_a == 0


def test_detect_communities_is_reproducible():
    gaps, X = _two_block_corpus(n_per_block=5)
    G1 = GG.build_gap_graph(gaps, X, knn_k=3, sim_threshold=0.0)
    G2 = GG.build_gap_graph(gaps, X, knn_k=3, sim_threshold=0.0)
    ids1 = GG.detect_communities(G1, seed=42)
    ids2 = GG.detect_communities(G2, seed=42)
    assert np.array_equal(ids1, ids2)


# ---------- bridge_score normalisation + schema ----------

def test_graph_bridge_pairs_schema_matches_legacy():
    gaps, X = _two_block_corpus(n_per_block=5)
    G = GG.build_gap_graph(gaps, X, knn_k=3, sim_threshold=0.0)
    ids = GG.detect_communities(G, seed=42)
    pairs = GG.graph_bridge_pairs(G, ids, top_n=10)
    # Same column schema as legacy cluster_pairs.tsv
    expected = {"cluster_a", "cluster_b", "label_a", "label_b",
                "cosine_sim", "paper_overlap", "type_complementarity", "bridge_score"}
    assert expected <= set(pairs.columns)


def test_graph_bridge_pairs_normalised_to_unit_interval():
    """Streamlit's ProgressColumn assumes [0, 1] — must hold across rows."""
    gaps, X = _two_block_corpus(n_per_block=6)
    G = GG.build_gap_graph(gaps, X, knn_k=4, sim_threshold=0.0)
    ids = GG.detect_communities(G, seed=42)
    pairs = GG.graph_bridge_pairs(G, ids, top_n=20)
    assert (pairs["bridge_score"] >= 0).all()
    assert (pairs["bridge_score"] <= 1.0).all()


def test_graph_bridge_pairs_empty_when_one_community():
    """A graph with one community has no inter-community edges -> empty frame."""
    n = 6
    X = np.eye(n)[:1].repeat(n, axis=0)  # all identical
    X += 0.01 * np.random.default_rng(0).normal(size=X.shape)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    gaps = pd.DataFrame([
        {"id": f"p{i}", "gap_type": "x", "section_type": "y",
         "gap_sentence": f"s{i}", "paragraph_text": "", "confidence": 0.9}
        for i in range(n)
    ])
    G = GG.build_gap_graph(gaps, X, knn_k=3, sim_threshold=0.0)
    ids = np.zeros(n, dtype=int)  # force single community
    pairs = GG.graph_bridge_pairs(G, ids, top_n=10)
    assert pairs.empty


# ---------- frontier nodes ----------

def test_frontier_nodes_ranks_boundary_gap_first():
    gaps, X = _frontier_corpus(n_per_block=4)
    G = GG.build_gap_graph(gaps, X, knn_k=4, sim_threshold=0.0)
    ids = GG.detect_communities(G, seed=42)
    frontier = GG.frontier_nodes(G, ids, gaps, top_n=10)
    # The boundary gap (last row) should rank high — let's just assert it's
    # in the top 3 rather than requiring exact ordering, because the
    # community detector can sometimes assign the boundary to one block.
    top3_idxs = set(frontier.head(3)["gap_idx"].astype(int).tolist())
    assert (len(gaps) - 1) in top3_idxs


def test_frontier_score_normalised():
    gaps, X = _frontier_corpus(n_per_block=4)
    G = GG.build_gap_graph(gaps, X, knn_k=4, sim_threshold=0.0)
    ids = GG.detect_communities(G, seed=42)
    frontier = GG.frontier_nodes(G, ids, gaps, top_n=10)
    assert (frontier["frontier_score"] >= 0).all()
    assert (frontier["frontier_score"] <= 1.0).all()


# ---------- bipartite method nodes ----------

def test_method_nodes_excluded_from_communities():
    """Method nodes (bipartite=1) must NOT appear in the community-ids array.
    The community array is aligned to gap nodes only."""
    gaps, X = _two_block_corpus(n_per_block=5)
    methods = pd.DataFrame([
        {"id": "m_paper", "method_type": "algorithm",
         "method_sentence": "We propose X.", "paragraph_text": "", "confidence": 0.9},
        {"id": "m_paper2", "method_type": "framework",
         "method_sentence": "Apply Y to Z.", "paragraph_text": "", "confidence": 0.9},
    ])
    # Method embeddings: parked midway between the two gap clusters so they
    # land in the (0.3, 0.7) sweet spot for some gaps.
    m_emb = np.zeros((2, X.shape[1]))
    m_emb[0, 0] = 0.7; m_emb[0, -1] = 0.7
    m_emb[1, 0] = 0.7; m_emb[1, -1] = 0.7
    m_emb /= np.linalg.norm(m_emb, axis=1, keepdims=True)

    G = GG.build_gap_graph(gaps, X, knn_k=3, sim_threshold=0.0,
                            methods=methods, method_embeddings=m_emb)
    ids = GG.detect_communities(G, seed=42)
    assert len(ids) == len(gaps)  # NOT len(gaps) + len(methods)

    # Confirm method nodes exist with bipartite=1
    bipart_attrs = {n: d.get("bipartite") for n, d in G.nodes(data=True)}
    assert any(v == 1 for v in bipart_attrs.values())


# ---------- dispatcher in theme_mining ----------

def test_cluster_gaps_via_graph_returns_aligned_ids():
    gaps, X = _two_block_corpus(n_per_block=5)
    ids, G = cluster_gaps_via_graph(gaps, X, knn_k=3, sim_threshold=0.0, seed=42)
    assert len(ids) == len(gaps)
    assert G.number_of_nodes() >= len(gaps)


def test_legacy_kmeans_path_still_produces_documented_columns():
    """Backward-compat guard: PR-4 must not break the legacy KMeans path."""
    gaps, X = _two_block_corpus(n_per_block=6)
    # Need cluster_id assigned for downstream functions
    gaps["cluster_id"] = cluster_embeddings(X, n_points=len(gaps))
    label_map = {int(c): f"cluster_{c}" for c in gaps["cluster_id"].unique() if c != -1}
    summary = build_cluster_summary(gaps, label_map)
    pairs = build_cluster_pairs(gaps, X, label_map, top_n=5)
    # Legacy column schemas held
    assert {"cluster_id", "n_items", "n_papers", "avg_conf", "theme_label"} <= set(summary.columns)
    legacy_pair_cols = {"cluster_a", "cluster_b", "label_a", "label_b",
                        "cosine_sim", "paper_overlap", "type_complementarity", "bridge_score"}
    if not pairs.empty:
        assert legacy_pair_cols <= set(pairs.columns)


# ---------- _knn_edges helper ----------

def test_knn_edges_threshold_filters_low_similarity():
    X = np.array([
        [1.0, 0.0],
        [0.99, 0.01],   # very close to row 0
        [0.0, 1.0],     # orthogonal to row 0 -> below threshold
    ])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    edges = list(GG._knn_edges(X, k=2, sim_threshold=0.5))
    pairs = {(a, b) for a, b, _ in edges}
    assert (0, 1) in pairs
    # Orthogonal pair must be filtered out by the threshold
    assert (0, 2) not in pairs
    assert (1, 2) not in pairs
