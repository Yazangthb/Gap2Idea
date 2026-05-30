"""Multi-relational gap graph: replaces clustering as the structural primitive.

Nodes are individual gap sentences, not cluster centroids. Edges encode four
relations:

  - **semantic** : top-k cosine ≥ threshold over normalised embeddings
  - **paper**    : two gaps from the same paper
  - **section**  : two gaps from the same `section_type` (limitations,
                   future_work, discussion, ...) with cosine ≥ 0.3 floor
  - **method**   : bipartite edge from a gap to a method node when their
                   cosine sits in the (0.30, 0.70) sweet spot — method nodes
                   carry `bipartite=1` and are EXCLUDED from community
                   detection.

Operators replace clustering operators:

  - `detect_communities`     →  Leiden (if leidenalg) else NetworkX Louvain.
                                Returns a per-gap `cluster_id` array — written
                                to `gaps_with_clusters.tsv` for backward compat.
  - `graph_bridge_pairs`     →  edge-betweenness summed per (community_a,
                                community_b) pair, normalised to [0, 1]. Same
                                column schema as legacy `cluster_pairs.tsv` so
                                Streamlit / generation / orchestrator are
                                drop-in compatible.
  - `graph_individual_pairs` →  one row per high-betweenness gap-gap edge
                                (the *real* structural bridges that
                                centroid-pair scoring averages away).
  - `frontier_nodes`         →  gaps with low intra-community degree and high
                                neighbour-community diversity. The long-tail
                                seeds HDBSCAN currently throws away.

All artifacts written by this module are computed deterministically from
`(seed, embedding_matrix, gaps)` — re-running on the same input always yields
the same `cluster_id` integers (we renumber communities post-hoc by
n_papers desc).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)


# ---------- defaults ----------

DEFAULT_KNN_K = 8
DEFAULT_SIM_THRESHOLD = 0.45        # min cosine for a semantic edge
DEFAULT_SECTION_FLOOR = 0.30        # min cosine for a section edge
DEFAULT_METHOD_LOW = 0.30           # method sweet-spot lower bound
DEFAULT_METHOD_HIGH = 0.70          # method sweet-spot upper bound

DEFAULT_EDGE_WEIGHTS: dict[str, float] = {
    "semantic": 1.0,
    "paper":    0.5,
    "section":  0.3,
    "method":   0.4,
}


# ---------- graph construction ----------

def _knn_edges(X: np.ndarray, k: int, sim_threshold: float) -> Iterable[tuple[int, int, float]]:
    """Yield (i, j, cosine) edges for each node's top-k neighbours above
    `sim_threshold`. Assumes `X` rows are L2-normalised.
    """
    if X.size == 0:
        return
    n = X.shape[0]
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)  # exclude self
    # Argpartition top-k per row
    k_eff = min(k, n - 1) if n > 1 else 0
    if k_eff <= 0:
        return
    top_idx = np.argpartition(-S, kth=k_eff - 1, axis=1)[:, :k_eff]
    for i in range(n):
        for j in top_idx[i]:
            sim = float(S[i, j])
            if sim < sim_threshold:
                continue
            # Emit each unordered pair only once
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            yield a, b, sim


def build_gap_graph(
    gaps: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    knn_k: int = DEFAULT_KNN_K,
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    edge_weights: dict[str, float] | None = None,
    methods: pd.DataFrame | None = None,
    method_embeddings: np.ndarray | None = None,
) -> nx.Graph:
    """Build a multi-relational gap graph.

    Gap nodes are integer indices `0..len(gaps)-1`. Method nodes (if any) are
    strings `"m:<idx>"` and carry `bipartite=1` so community detection can
    exclude them.
    """
    edge_weights = {**DEFAULT_EDGE_WEIGHTS, **(edge_weights or {})}
    G = nx.Graph()

    if len(gaps) != embeddings.shape[0]:
        raise ValueError(
            f"gaps ({len(gaps)}) and embeddings ({embeddings.shape[0]}) must align"
        )

    # ---- nodes (gaps) ----
    for i, r in enumerate(gaps.itertuples(index=False)):
        G.add_node(
            i,
            bipartite=0,
            paper_id=str(getattr(r, "id", "")),
            gap_type=str(getattr(r, "gap_type", "")),
            section_type=str(getattr(r, "section_type", "")),
            confidence=float(getattr(r, "confidence", 0.0) or 0.0),
        )

    # ---- semantic edges (top-k above threshold) ----
    sem_w = edge_weights["semantic"]
    n_sem = 0
    for a, b, sim in _knn_edges(embeddings, knn_k, sim_threshold):
        if G.has_edge(a, b):
            # Already added from the symmetric pass; keep max sim
            G[a][b]["cosine"] = max(G[a][b]["cosine"], sim)
            continue
        G.add_edge(a, b, weight=sem_w * sim, edge_type="semantic", cosine=sim)
        n_sem += 1

    # ---- paper edges (same paper_id) ----
    paper_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, n in enumerate(G.nodes(data=True)):
        # n is a (node_id, data) tuple; we want pure index iteration
        pass
    for node, data in G.nodes(data=True):
        if data.get("bipartite") == 0:
            pid = data.get("paper_id") or ""
            if pid:
                paper_to_idx[pid].append(int(node))
    n_paper = 0
    paper_w = edge_weights["paper"]
    for pid, idxs in paper_to_idx.items():
        if len(idxs) < 2:
            continue
        for a, b in combinations(sorted(idxs), 2):
            sim = float(embeddings[a] @ embeddings[b])
            if G.has_edge(a, b):
                # Upgrade weight to include the paper-relation; keep edge_type
                # as a sorted multi-type tag so later analysis can introspect.
                G[a][b]["weight"] = max(G[a][b]["weight"], paper_w)
                et = G[a][b].get("edge_type", "")
                if "paper" not in et:
                    G[a][b]["edge_type"] = (et + "+paper") if et else "paper"
            else:
                G.add_edge(a, b, weight=paper_w, edge_type="paper", cosine=sim)
                n_paper += 1

    # ---- section edges (same non-empty section_type AND cosine >= floor) ----
    section_to_idx: dict[str, list[int]] = defaultdict(list)
    for node, data in G.nodes(data=True):
        if data.get("bipartite") == 0:
            st = data.get("section_type") or ""
            if st:
                section_to_idx[st].append(int(node))
    n_section = 0
    section_w = edge_weights["section"]
    for st, idxs in section_to_idx.items():
        if len(idxs) < 2:
            continue
        for a, b in combinations(sorted(idxs), 2):
            sim = float(embeddings[a] @ embeddings[b])
            if sim < DEFAULT_SECTION_FLOOR:
                continue
            if G.has_edge(a, b):
                G[a][b]["weight"] = max(G[a][b]["weight"], section_w * sim)
                et = G[a][b].get("edge_type", "")
                if "section" not in et:
                    G[a][b]["edge_type"] = (et + "+section") if et else "section"
            else:
                G.add_edge(a, b, weight=section_w * sim, edge_type="section", cosine=sim)
                n_section += 1

    # ---- method edges (bipartite, sweet-spot only) ----
    n_method = 0
    if methods is not None and not methods.empty and method_embeddings is not None and method_embeddings.size > 0:
        method_w = edge_weights["method"]
        # Pairwise cosines: (n_methods, n_gaps)
        M = method_embeddings @ embeddings.T
        for m_idx in range(method_embeddings.shape[0]):
            mid = f"m:{m_idx}"
            method_row = methods.iloc[m_idx]
            G.add_node(
                mid,
                bipartite=1,
                paper_id=str(method_row.get("id", "")),
                method_type=str(method_row.get("method_type", "")),
                method_sentence=str(method_row.get("method_sentence", "")),
            )
            row = M[m_idx]
            mask = (row >= DEFAULT_METHOD_LOW) & (row <= DEFAULT_METHOD_HIGH)
            for g_idx in np.where(mask)[0]:
                sim = float(row[g_idx])
                G.add_edge(int(g_idx), mid,
                           weight=method_w * sim, edge_type="method", cosine=sim)
                n_method += 1

    log.info(
        "gap_graph: %d nodes (%d gaps, %d methods), edges sem=%d paper=%d section=%d method=%d",
        G.number_of_nodes(),
        sum(1 for _, d in G.nodes(data=True) if d.get("bipartite") == 0),
        sum(1 for _, d in G.nodes(data=True) if d.get("bipartite") == 1),
        n_sem, n_paper, n_section, n_method,
    )
    return G


def _gap_subgraph(G: nx.Graph) -> nx.Graph:
    """Return the induced subgraph over gap nodes only (excludes method nodes).
    Used for community detection and betweenness — method edges only inform
    `retrieve_methods` ranking, not community structure."""
    gap_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    return G.subgraph(gap_nodes).copy()


# ---------- community detection ----------

def _renumber_by_size(
    communities: np.ndarray, paper_ids: list[str],
) -> np.ndarray:
    """Renumber community IDs so larger (more papers, then more items) comes
    first. Makes `cluster_id` stable across runs and matches the legacy
    behaviour of build_cluster_summary's sort order."""
    items_per_comm: Counter = Counter(communities.tolist())
    papers_per_comm: dict[int, set[str]] = defaultdict(set)
    for c, pid in zip(communities, paper_ids):
        papers_per_comm[int(c)].add(str(pid))
    keys = sorted(
        items_per_comm.keys(),
        key=lambda c: (-len(papers_per_comm[c]), -items_per_comm[c], c),
    )
    remap = {old: new for new, old in enumerate(keys)}
    return np.array([remap[int(c)] for c in communities], dtype=int)


def detect_communities(G: nx.Graph, *, resolution: float = 1.0, seed: int = 42) -> np.ndarray:
    """Run Leiden (preferred) or NetworkX Louvain (fallback) on the gap-induced
    subgraph. Returns a per-node `cluster_id` array aligned to the *sorted*
    gap node ids (i.e. position i corresponds to the i-th gap)."""
    gap_subgraph = _gap_subgraph(G)
    nodes_sorted = sorted(gap_subgraph.nodes())
    if not nodes_sorted:
        return np.zeros(0, dtype=int)

    # Try Leiden via igraph + leidenalg.
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore

        # Map networkx node ids -> igraph 0..n-1
        idx_of = {n: i for i, n in enumerate(nodes_sorted)}
        ig_edges = [
            (idx_of[u], idx_of[v]) for u, v in gap_subgraph.edges()
            if u in idx_of and v in idx_of
        ]
        weights = [
            float(gap_subgraph[u][v].get("weight", 1.0))
            for u, v in gap_subgraph.edges()
            if u in idx_of and v in idx_of
        ]
        g = ig.Graph(n=len(nodes_sorted), edges=ig_edges, directed=False)
        g.es["weight"] = weights
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=resolution, seed=seed, weights="weight",
        )
        raw = np.array(part.membership, dtype=int)
        log.info("Leiden found %d communities (resolution=%.2f)", raw.max() + 1 if raw.size else 0, resolution)
    except ImportError:
        # NetworkX Louvain fallback — same notion of communities; less
        # principled resolution control but always available.
        log.info("leidenalg unavailable; using NetworkX Louvain fallback")
        communities = nx.community.louvain_communities(
            gap_subgraph, weight="weight", resolution=resolution, seed=seed,
        )
        member_of: dict[int, int] = {}
        for cid, comm in enumerate(communities):
            for n in comm:
                member_of[n] = cid
        raw = np.array([member_of[n] for n in nodes_sorted], dtype=int)
        log.info("Louvain found %d communities (resolution=%.2f)", raw.max() + 1 if raw.size else 0, resolution)

    # Renumber for stability across runs (largest community = id 0).
    paper_ids = [str(G.nodes[n].get("paper_id", "")) for n in nodes_sorted]
    return _renumber_by_size(raw, paper_ids)


# ---------- bridge pairs (community-level, legacy schema) ----------

def graph_bridge_pairs(
    G: nx.Graph,
    community_ids: np.ndarray,
    *,
    top_n: int = 30,
    edge_betweenness_k: int | None = 64,
) -> pd.DataFrame:
    """Compute community-pair bridge scores from edge betweenness.

    `bridge_score` is the sum of edge-betweenness centrality for every
    inter-community semantic edge connecting communities A and B, normalised
    to [0, 1] across the returned frame (so Streamlit's existing
    `ProgressColumn(min=0, max=1)` keeps working).

    Returns the legacy column schema: `cluster_a, cluster_b, label_a,
    label_b, cosine_sim, paper_overlap, type_complementarity, bridge_score`.
    `label_a` / `label_b` are filled in by the caller via the label_map.
    `cosine_sim` is the mean cosine of inter-community semantic edges (so
    downstream plots remain interpretable as a "how related" axis).
    """
    gap_subgraph = _gap_subgraph(G)
    nodes_sorted = sorted(gap_subgraph.nodes())
    if not nodes_sorted or community_ids.size == 0:
        return pd.DataFrame(columns=[
            "cluster_a", "cluster_b", "label_a", "label_b",
            "cosine_sim", "paper_overlap", "type_complementarity", "bridge_score",
        ])

    node_to_comm = {n: int(c) for n, c in zip(nodes_sorted, community_ids)}

    # Edge betweenness, sampled for speed on big graphs.
    n_nodes = gap_subgraph.number_of_nodes()
    k_arg = None if (edge_betweenness_k is None or edge_betweenness_k >= n_nodes) else min(edge_betweenness_k, n_nodes)
    eb = nx.edge_betweenness_centrality(
        gap_subgraph, k=k_arg, normalized=True, weight=None, seed=42,
    )

    # Aggregate per community pair.
    bridge_sum: dict[tuple[int, int], float] = defaultdict(float)
    cosines: dict[tuple[int, int], list[float]] = defaultdict(list)
    for (u, v), score in eb.items():
        ca, cb = node_to_comm[u], node_to_comm[v]
        if ca == cb:
            continue
        key = (min(ca, cb), max(ca, cb))
        bridge_sum[key] += float(score)
        cosines[key].append(float(gap_subgraph[u][v].get("cosine", 0.0)))

    # Papers + gap_type distributions per community for the legacy columns.
    papers_per_c: dict[int, set[str]] = defaultdict(set)
    types_per_c: dict[int, Counter] = defaultdict(Counter)
    for n in nodes_sorted:
        c = node_to_comm[n]
        d = G.nodes[n]
        if d.get("paper_id"):
            papers_per_c[c].add(d["paper_id"])
        if d.get("gap_type"):
            types_per_c[c][d["gap_type"]] += 1

    def _type_complement(a: Counter, b: Counter) -> float:
        keys = set(a) | set(b)
        if not keys:
            return 0.0
        ta = sum(a.values()) or 1
        tb = sum(b.values()) or 1
        l1 = sum(abs(a[k] / ta - b[k] / tb) for k in keys)
        return l1 / 2.0

    rows: list[dict] = []
    for (ca, cb), score in bridge_sum.items():
        pa, pb = papers_per_c[ca], papers_per_c[cb]
        overlap = (len(pa & pb) / max(1, len(pa | pb)))
        type_comp = _type_complement(types_per_c[ca], types_per_c[cb])
        rows.append({
            "cluster_a": int(ca),
            "cluster_b": int(cb),
            "label_a": "",
            "label_b": "",
            "cosine_sim": float(np.mean(cosines[(ca, cb)])) if cosines[(ca, cb)] else 0.0,
            "paper_overlap": float(overlap),
            "type_complementarity": float(type_comp),
            "bridge_score_raw": float(score),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "cluster_a", "cluster_b", "label_a", "label_b",
            "cosine_sim", "paper_overlap", "type_complementarity", "bridge_score",
        ])

    df = pd.DataFrame(rows)
    # Down-weight pairs whose communities share many papers (centroid-bridge
    # legacy semantics) and that are *identical* type-wise.
    df["bridge_score_raw"] = (
        df["bridge_score_raw"] * (1.0 - df["paper_overlap"]) *
        (0.5 + 0.5 * df["type_complementarity"])
    )

    # Normalise to [0, 1] so the downstream Streamlit ProgressColumn keeps
    # rendering correctly. Empty/zero is safe.
    max_score = float(df["bridge_score_raw"].max() or 0.0)
    df["bridge_score"] = (df["bridge_score_raw"] / max_score) if max_score > 0 else 0.0
    df = df.drop(columns=["bridge_score_raw"])
    df = df.sort_values("bridge_score", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ---------- individual-gap bridges (new artifact) ----------

def graph_individual_pairs(
    G: nx.Graph,
    community_ids: np.ndarray,
    gaps: pd.DataFrame,
    *,
    top_n: int = 50,
    edge_betweenness_k: int | None = 64,
) -> pd.DataFrame:
    """One row per high-betweenness inter-community gap-gap edge.

    These are the *real* structural bridges. Each row identifies the two
    gaps, their papers, their communities, the edge cosine, and a normalised
    bridge_score in [0, 1].
    """
    gap_subgraph = _gap_subgraph(G)
    nodes_sorted = sorted(gap_subgraph.nodes())
    if not nodes_sorted or community_ids.size == 0:
        return pd.DataFrame(columns=[
            "gap_a_idx", "gap_b_idx", "paper_a", "paper_b",
            "cluster_a", "cluster_b", "cosine", "betweenness", "bridge_score",
        ])

    node_to_comm = {n: int(c) for n, c in zip(nodes_sorted, community_ids)}
    n_nodes = gap_subgraph.number_of_nodes()
    k_arg = None if (edge_betweenness_k is None or edge_betweenness_k >= n_nodes) else min(edge_betweenness_k, n_nodes)
    eb = nx.edge_betweenness_centrality(
        gap_subgraph, k=k_arg, normalized=True, weight=None, seed=42,
    )

    rows: list[dict] = []
    for (u, v), score in eb.items():
        ca, cb = node_to_comm[u], node_to_comm[v]
        if ca == cb:
            continue
        pa = str(gaps.iloc[u]["id"]) if u < len(gaps) else ""
        pb = str(gaps.iloc[v]["id"]) if v < len(gaps) else ""
        rows.append({
            "gap_a_idx": int(u),
            "gap_b_idx": int(v),
            "paper_a":   pa,
            "paper_b":   pb,
            "cluster_a": int(min(ca, cb)),
            "cluster_b": int(max(ca, cb)),
            "cosine":    float(gap_subgraph[u][v].get("cosine", 0.0)),
            "betweenness": float(score),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "gap_a_idx", "gap_b_idx", "paper_a", "paper_b",
            "cluster_a", "cluster_b", "cosine", "betweenness", "bridge_score",
        ])

    df = pd.DataFrame(rows)
    max_b = float(df["betweenness"].max() or 0.0)
    df["bridge_score"] = (df["betweenness"] / max_b) if max_b > 0 else 0.0
    df = df.sort_values("bridge_score", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ---------- frontier nodes (new artifact) ----------

def frontier_nodes(
    G: nx.Graph,
    community_ids: np.ndarray,
    gaps: pd.DataFrame,
    *,
    top_n: int = 50,
) -> pd.DataFrame:
    """Score each gap by a frontier metric: low intra-community degree share
    AND high count of distinct neighbour communities.

    These are the gaps that *bridge* multiple themes from a single position —
    exactly the long-tail novelty seeds HDBSCAN currently drops as noise.
    """
    gap_subgraph = _gap_subgraph(G)
    nodes_sorted = sorted(gap_subgraph.nodes())
    if not nodes_sorted or community_ids.size == 0:
        return pd.DataFrame(columns=[
            "gap_idx", "paper_id", "cluster_id", "gap_sentence",
            "confidence", "intra_degree", "n_neighbour_clusters",
            "frontier_score",
        ])

    node_to_comm = {n: int(c) for n, c in zip(nodes_sorted, community_ids)}
    rows: list[dict] = []
    for n in nodes_sorted:
        my_c = node_to_comm[n]
        neighs = list(gap_subgraph.neighbors(n))
        total = len(neighs)
        if total == 0:
            intra_frac, neighbour_clusters = 0.0, 0
        else:
            intra = sum(1 for v in neighs if node_to_comm[v] == my_c)
            intra_frac = intra / total
            neighbour_clusters = len({node_to_comm[v] for v in neighs if node_to_comm[v] != my_c})
        # Higher score for low intra-frac AND many neighbour communities.
        # Confidence weight breaks ties toward higher-confidence gaps.
        conf = float(G.nodes[n].get("confidence", 0.0) or 0.0)
        score = (1.0 - intra_frac) * (np.log1p(neighbour_clusters)) + 0.05 * conf
        gap_sentence = str(gaps.iloc[n]["gap_sentence"]) if n < len(gaps) else ""
        paper_id = str(gaps.iloc[n]["id"]) if n < len(gaps) else ""
        rows.append({
            "gap_idx": int(n),
            "paper_id": paper_id,
            "cluster_id": my_c,
            "gap_sentence": gap_sentence,
            "confidence": conf,
            "intra_degree": int(total - sum(1 for v in neighs if node_to_comm[v] != my_c)),
            "n_neighbour_clusters": int(neighbour_clusters),
            "frontier_score": float(score),
        })

    df = pd.DataFrame(rows)
    # Normalise frontier_score to [0, 1] for readability.
    max_s = float(df["frontier_score"].max() or 0.0)
    if max_s > 0:
        df["frontier_score"] = df["frontier_score"] / max_s
    df = df.sort_values("frontier_score", ascending=False).reset_index(drop=True)
    return df.head(top_n)
