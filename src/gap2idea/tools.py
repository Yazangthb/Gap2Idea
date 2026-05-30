"""Single source of truth for the agentic tool surface.

Design principle: every tool an agent needs is implemented exactly once,
here, as a plain async Python function. The MCP server is a thin wrapper
that exposes these. In-process agents (the critic, the orchestrator) call
them directly. This keeps the contract identical no matter how the agent
is wired and lets us unit-test the tools without spinning up a server.

Tools fall into four groups:

  • READ        list_themes · get_theme · get_evidence · list_ideas · get_idea
  • RETRIEVAL   retrieve_methods · search_prior_art
  • CHECK       score_novelty · check_evidence_overlap
  • WRITE       save_idea
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gap2idea.config import get_paths
from gap2idea.pipeline.openai_ideas import _retrieve_methods_for_cluster
from gap2idea.pipeline.semantic_scholar import S2Client
from gap2idea.utils import get_logger

log = get_logger(__name__)


# ===========================================================================
# Tool definitions (JSON Schema) — shared between MCP and tests
# ===========================================================================

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "list_themes",
        "description": "List every research-gap theme (cluster) in the corpus, with size and label.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_theme",
        "description": "Get the full details of one theme: label, keywords, papers and gap sentences.",
        "inputSchema": {
            "type": "object",
            "properties": {"cluster_id": {"type": "integer"}},
            "required": ["cluster_id"],
        },
    },
    {
        "name": "get_evidence",
        "description": "Sample diverse gap-sentence evidence from one theme (greedy Jaccard diversification).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cluster_id": {"type": "integer"},
                "k": {"type": "integer", "default": 4, "minimum": 1, "maximum": 20},
            },
            "required": ["cluster_id"],
        },
    },
    {
        "name": "retrieve_methods",
        "description": "Retrieve method-claim sentences from the method library whose centroid similarity to the chosen theme falls in the sweet spot (default 0.30-0.70). Used by method-gap idea mode.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cluster_id": {"type": "integer"},
                "top_k": {"type": "integer", "default": 5},
                "sim_low": {"type": "number", "default": 0.30},
                "sim_high": {"type": "number", "default": 0.70},
            },
            "required": ["cluster_id"],
        },
    },
    {
        "name": "search_prior_art",
        "description": "Search Semantic Scholar for papers matching a query. Returns up to 10 hits with title + abstract.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 10, "minimum": 1, "maximum": 25},
            },
            "required": ["query"],
        },
    },
    {
        "name": "score_novelty",
        "description": "Compute 1 - max_cosine(idea, S2_hit_abstract) for an idea title + research question. Returns the closest prior work.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "research_question": {"type": "string"},
                "method_sketch": {"type": "string", "default": ""},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["title", "research_question"],
        },
    },
    {
        "name": "check_evidence_overlap",
        "description": "Given a list of (paper_id, gap_sentence) the LLM claims to have used, return the fraction that actually appeared in the input evidence. 1.0 = perfectly grounded; <1.0 = hallucinated citations.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "evidence_used": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string"},
                            "gap_sentence": {"type": "string"},
                        },
                    },
                },
                "fed_evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string"},
                            "gap_sentence": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["evidence_used", "fed_evidence"],
        },
    },
    {
        "name": "list_ideas",
        "description": "List the saved ideas in the corpus library.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_novelty": {"type": "number", "default": 0.0},
                "min_confidence": {"type": "number", "default": 0.0},
                "mode": {"type": "string", "enum": ["bridge", "within", "method-gap", "any"], "default": "any"},
            },
            "required": [],
        },
    },
    {
        "name": "get_idea",
        "description": "Get one saved idea by row index (as listed by list_ideas).",
        "inputSchema": {
            "type": "object",
            "properties": {"idx": {"type": "integer"}},
            "required": ["idx"],
        },
    },
    {
        "name": "save_idea",
        "description": "Append a new idea to artifacts/ideas.tsv.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "idea": {
                    "type": "object",
                    "description": "Flat idea dict — title, research_question, method_sketch, evaluation_plan, expected_contribution, assumptions_and_risks, evidence_used (list), confidence, mode, cluster_a, label_a, ...",
                },
            },
            "required": ["idea"],
        },
    },
    # PR-4: gap-graph view tools. Empty result if --method graph was not run.
    {
        "name": "list_frontier",
        "description": (
            "List the top frontier gaps from the gap graph: gaps sitting on "
            "community boundaries (low intra-community degree, many neighbour "
            "communities). These are the long-tail novelty seeds the legacy "
            "clustering throws away as noise. Empty if `theme-mine --method graph` "
            "was not run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            },
            "required": [],
        },
    },
    {
        "name": "list_gap_pairs",
        "description": (
            "List the top inter-community gap-gap structural bridges from the "
            "gap graph. Each row identifies two individual gaps (not centroids) "
            "connected by a high-betweenness edge. Empty if `theme-mine "
            "--method graph` was not run."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "default": 20, "minimum": 1, "maximum": 100},
            },
            "required": [],
        },
    },
]


# ===========================================================================
# Lazy state — load corpus artifacts once per process
# ===========================================================================

@dataclass
class CorpusHandle:
    """All the corpus state agents need. Loaded lazily; cached for re-use."""
    gaps: pd.DataFrame
    clusters: pd.DataFrame
    labels: pd.DataFrame
    pairs: pd.DataFrame
    embeddings: np.ndarray | None
    methods: pd.DataFrame
    method_embeddings: np.ndarray | None
    ideas_path: Path
    root: Path
    # PR-4: gap-graph artifacts (optional; empty when running KMeans mode)
    frontier: pd.DataFrame = None         # type: ignore[assignment]
    gap_pairs: pd.DataFrame = None        # type: ignore[assignment]

    @property
    def label_map(self) -> dict[int, str]:
        if self.labels.empty:
            return {}
        return dict(zip(self.labels["cluster_id"].astype(int), self.labels["theme_label"]))


_CORPUS: CorpusHandle | None = None


def _load_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def _embed_methods(methods: pd.DataFrame, out_path: Path) -> np.ndarray | None:
    if methods.empty:
        return None
    if out_path.exists() and len(np.load(out_path)) == len(methods):
        return np.load(out_path)
    from gap2idea.pipeline.theme_mining import embed_sentences

    emb = embed_sentences(methods["method_sentence"].tolist())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, emb)
    return emb


def load_corpus(root: str | Path = ".", refresh: bool = False) -> CorpusHandle:
    """Load all corpus artifacts into memory. Cached per process."""
    global _CORPUS
    if _CORPUS is not None and not refresh:
        return _CORPUS

    paths = get_paths(root)
    gap_emb_path = paths.artifacts / "gap_embeddings.npy"
    methods_tsv = paths.data / "methods.tsv"
    method_emb_path = paths.artifacts / "method_embeddings.npy"

    methods_df = _load_tsv(methods_tsv)

    _CORPUS = CorpusHandle(
        gaps=_load_tsv(paths.artifacts / "gaps_with_clusters.tsv"),
        clusters=_load_tsv(paths.artifacts / "cluster_summary.tsv"),
        labels=_load_tsv(paths.artifacts / "cluster_labels.tsv"),
        pairs=_load_tsv(paths.artifacts / "cluster_pairs.tsv"),
        embeddings=np.load(gap_emb_path) if gap_emb_path.exists() else None,
        methods=methods_df,
        method_embeddings=_embed_methods(methods_df, method_emb_path),
        ideas_path=paths.artifacts / "ideas.tsv",
        root=Path(root),
        frontier=_load_tsv(paths.artifacts / "gap_frontier.tsv"),
        gap_pairs=_load_tsv(paths.artifacts / "gap_pairs.tsv"),
    )
    return _CORPUS


# ===========================================================================
# Tool implementations
# ===========================================================================

async def list_themes(root: str | Path = ".") -> list[dict]:
    c = load_corpus(root)
    if c.clusters.empty:
        return []
    return [
        {
            "cluster_id": int(r["cluster_id"]),
            "theme_label": r.get("theme_label", ""),
            "n_papers": int(r.get("n_papers", 0)),
            "n_items": int(r.get("n_items", 0)),
            "avg_confidence": float(r.get("avg_conf", 0.0)),
        }
        for _, r in c.clusters.iterrows()
    ]


async def get_theme(cluster_id: int, root: str | Path = ".") -> dict:
    c = load_corpus(root)
    if c.clusters.empty:
        return {"error": "no clusters loaded"}
    crow = c.clusters[c.clusters["cluster_id"] == cluster_id]
    if crow.empty:
        return {"error": f"cluster_id {cluster_id} not found"}
    crow = crow.iloc[0]

    kw = ""
    if not c.labels.empty:
        kw_row = c.labels[c.labels["cluster_id"] == cluster_id]
        if not kw_row.empty and "keywords" in kw_row.columns:
            kw = str(kw_row.iloc[0]["keywords"])

    sub = c.gaps[c.gaps["cluster_id"] == cluster_id]
    papers = [
        {
            "paper_id": str(pid),
            "n_gaps": len(grp),
            "gap_sentences": grp["gap_sentence"].tolist(),
            "gap_types": grp["gap_type"].tolist() if "gap_type" in grp.columns else [],
            "max_confidence": float(grp["confidence"].max()),
        }
        for pid, grp in sub.groupby("id")
    ]
    return {
        "cluster_id": int(cluster_id),
        "theme_label": crow.get("theme_label", ""),
        "keywords": kw,
        "n_papers": int(crow.get("n_papers", 0)),
        "n_gaps": int(crow.get("n_items", 0)),
        "avg_confidence": float(crow.get("avg_conf", 0.0)),
        "papers": papers,
    }


async def get_evidence(cluster_id: int, k: int = 4, root: str | Path = ".") -> list[dict]:
    from gap2idea.pipeline.openai_ideas import _diverse_evidence, _evidence_payload

    c = load_corpus(root)
    ev_df = _diverse_evidence(c.gaps, int(cluster_id), k=int(k))
    if ev_df.empty:
        return []
    return _evidence_payload(ev_df)


async def retrieve_methods(
    cluster_id: int,
    top_k: int = 5,
    sim_low: float = 0.30,
    sim_high: float = 0.70,
    root: str | Path = ".",
) -> list[dict]:
    c = load_corpus(root)
    if c.embeddings is None or c.methods.empty or c.method_embeddings is None:
        return []
    df = _retrieve_methods_for_cluster(
        int(cluster_id), c.embeddings, c.gaps, c.methods, c.method_embeddings,
        top_k=int(top_k), sim_low=float(sim_low), sim_high=float(sim_high),
    )
    if df.empty:
        return []
    return [
        {
            "paper_id": str(r["id"]),
            "method_type": str(r.get("method_type", "")),
            "method_sentence": str(r["method_sentence"]),
            "paragraph_text": str(r.get("paragraph_text", "")),
            "similarity_to_gap_cluster": float(r["_sim_to_gap_cluster"]),
        }
        for _, r in df.iterrows()
    ]


async def search_prior_art(query: str, limit: int = 10) -> list[dict]:
    try:
        hits = S2Client().search(query, limit=int(limit))
    except Exception as e:
        log.warning("S2 search failed: %s", e)
        return []
    return [
        {
            "paperId": h.get("paperId"),
            "title": h.get("title") or "",
            "year": h.get("year"),
            "venue": h.get("venue") or "",
            "abstract": (h.get("abstract") or "")[:1200],
            "url": h.get("url") or "",
            "citationCount": h.get("citationCount", 0),
        }
        for h in hits
    ]


async def score_novelty(
    title: str, research_question: str, method_sketch: str = "", top_k: int = 10,
) -> dict:
    from gap2idea.pipeline.openai_ideas import novelty_check
    from sentence_transformers import SentenceTransformer

    idea = {
        "title": title,
        "research_question": research_question,
        "method_sketch": method_sketch or "",
    }
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return novelty_check(idea, S2Client(), embedder, top_k=int(top_k))


async def check_evidence_overlap(
    evidence_used: list[dict], fed_evidence: list[dict],
) -> dict:
    fed_keys = {
        (str(e.get("paper_id", "")), str(e.get("gap_sentence", "")).strip())
        for e in fed_evidence
    }
    if not evidence_used:
        return {"overlap": 0.0, "n_used": 0, "n_grounded": 0, "hallucinated": []}
    hits = 0
    hallucinated = []
    for e in evidence_used:
        key = (str(e.get("paper_id", "")), str(e.get("gap_sentence", "")).strip())
        if key in fed_keys:
            hits += 1
        else:
            hallucinated.append(e)
    return {
        "overlap": hits / len(evidence_used),
        "n_used": len(evidence_used),
        "n_grounded": hits,
        "hallucinated": hallucinated,
    }


async def list_ideas(
    min_novelty: float = 0.0,
    min_confidence: float = 0.0,
    mode: str = "any",
    root: str | Path = ".",
) -> list[dict]:
    c = load_corpus(root)
    df = _load_tsv(c.ideas_path)
    if df.empty:
        return []
    if "novelty_score" in df.columns:
        df = df[pd.to_numeric(df["novelty_score"], errors="coerce").fillna(0) >= min_novelty]
    if "idea_confidence" in df.columns:
        df = df[pd.to_numeric(df["idea_confidence"], errors="coerce").fillna(0) >= min_confidence]
    if mode != "any" and "mode" in df.columns:
        df = df[df["mode"] == mode]
    return df.reset_index(drop=True).to_dict(orient="records")


async def get_idea(idx: int, root: str | Path = ".") -> dict:
    rows = await list_ideas(root=root)
    if not rows or idx < 0 or idx >= len(rows):
        return {"error": f"index {idx} out of range; have {len(rows)} ideas"}
    return rows[idx]


async def save_idea(idea: dict, root: str | Path = ".") -> dict:
    c = load_corpus(root)
    existing = _load_tsv(c.ideas_path)
    new = pd.DataFrame([idea])
    merged = pd.concat([existing, new], ignore_index=True) if not existing.empty else new
    c.ideas_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(c.ideas_path, sep="\t", index=False)
    return {"saved": True, "total_ideas": len(merged)}


# ---------- PR-4: gap-graph view tools ----------

async def list_frontier(top_n: int = 20, root: str | Path = ".") -> list[dict]:
    """Top frontier gaps. Empty if the graph artifact wasn't produced."""
    c = load_corpus(root)
    df = c.frontier if c.frontier is not None else _load_tsv(Path("."))
    if df is None or df.empty:
        return []
    if "frontier_score" in df.columns:
        df = df.sort_values("frontier_score", ascending=False)
    return df.head(int(top_n)).to_dict(orient="records")


async def list_gap_pairs(top_n: int = 20, root: str | Path = ".") -> list[dict]:
    """Top inter-community gap-gap bridges from the gap graph."""
    c = load_corpus(root)
    df = c.gap_pairs if c.gap_pairs is not None else _load_tsv(Path("."))
    if df is None or df.empty:
        return []
    if "bridge_score" in df.columns:
        df = df.sort_values("bridge_score", ascending=False)
    return df.head(int(top_n)).to_dict(orient="records")


# ===========================================================================
# Dispatch helper used by both MCP server and orchestrator
# ===========================================================================

TOOL_FNS = {
    "list_themes": list_themes,
    "get_theme": get_theme,
    "get_evidence": get_evidence,
    "retrieve_methods": retrieve_methods,
    "search_prior_art": search_prior_art,
    "score_novelty": score_novelty,
    "check_evidence_overlap": check_evidence_overlap,
    "list_ideas": list_ideas,
    "get_idea": get_idea,
    "save_idea": save_idea,
    # PR-4
    "list_frontier": list_frontier,
    "list_gap_pairs": list_gap_pairs,
}


async def dispatch(name: str, arguments: dict | None = None, root: str | Path = ".") -> Any:
    """Call a tool by name. Used by MCP server + orchestrator."""
    if name not in TOOL_FNS:
        raise ValueError(f"unknown tool: {name}")
    args = dict(arguments or {})
    # Most tools accept root; pass it through if the function signature permits.
    fn = TOOL_FNS[name]
    import inspect
    sig = inspect.signature(fn)
    if "root" in sig.parameters and "root" not in args:
        args["root"] = root
    return await fn(**args)
