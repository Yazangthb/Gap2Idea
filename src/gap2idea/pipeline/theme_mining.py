"""Theme mining: embed gaps -> cluster -> label -> score cluster pairs.

Two thesis-quality upgrades over the original notebook:

1. **LLM cluster labels.** TF-IDF keyword stuffing ("dynamic, graph, neural,
   networks, ...") is replaced with short human-readable theme names produced
   by an LLM call. TF-IDF terms are retained as a fallback signal and as
   per-cluster keywords for display.

2. **Bridge-score pair selection.** The previous approach picked the *most
   similar* cluster pairs, which produces redundant ideas. We instead score
   each pair on three axes and rank by the composite "bridge_score":

       bridge_score = peak(sim, 0.45) * (1 - paper_overlap) * type_complementarity

   - `peak(sim, 0.45)`: triangular peak at moderate similarity (0.45). Pairs
     with sim==0.45 score 1.0; sim==0 or sim==0.9 score near 0. Encodes the
     intuition that the best ideas come from *related but distinct* themes.
   - `paper_overlap`: Jaccard overlap of source-paper IDs. Penalises pairs
     mined from the same authors / same papers.
   - `type_complementarity`: 1 when the two clusters draw from different
     gap_type mixes (e.g. one is mostly "future_work", the other "limitation"),
     0 when identical.
"""
from __future__ import annotations

import json
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client, parse_json_response
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)


# ---------- cleaning ----------

def clean_gaps(gaps: pd.DataFrame, min_conf: float = 0.5) -> pd.DataFrame:
    gaps = gaps.copy()
    for c in ["gap_sentence", "paragraph_text"]:
        if c in gaps.columns:
            gaps[c] = gaps[c].astype(str).str.replace("\n", " ").str.strip()
    # section_type carries the dominant paper section (limitations/future_work/
    # discussion/fallback/tail) the gap was extracted from. Older gaps.tsv files
    # written before this column was added lack it — default to "" so downstream
    # consumers (graph edge features, evidence payloads) can treat empty as
    # "unknown" rather than crashing.
    if "section_type" not in gaps.columns:
        gaps["section_type"] = ""
    else:
        gaps["section_type"] = gaps["section_type"].fillna("").astype(str).str.strip()
    gaps["confidence"] = pd.to_numeric(gaps["confidence"], errors="coerce").fillna(0.0)
    gaps = gaps[(gaps["confidence"] >= min_conf) & (gaps["gap_sentence"].str.len() >= 20)]
    gaps = gaps.drop_duplicates(subset=["id", "gap_sentence"]).reset_index(drop=True)
    return gaps


# ---------- embeddings ----------

def embed_sentences(sentences: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(model_name)
    X = embedder.encode(
        sentences, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    return X


# ---------- clustering ----------

def cluster_embeddings(X: np.ndarray, n_points: int) -> np.ndarray:
    if n_points < 150:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        k_candidates = list(range(4, min(12, n_points)))
        best_k, best_s = None, -1.0
        for k in k_candidates:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            lbl = km.fit_predict(X)
            try:
                s = silhouette_score(X, lbl)
            except ValueError:
                continue
            if s > best_s:
                best_k, best_s = k, s
        log.info("KMeans selected k=%s (silhouette=%.3f)", best_k, best_s)
        km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        return km.fit_predict(X)

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(8, n_points // 40), min_samples=2, metric="euclidean"
    )
    return clusterer.fit_predict(X)


# ---------- cluster keyword extraction ----------

def keyword_label(sentences: list[str], k: int = 6) -> str:
    if len(sentences) < 2:
        return ""
    v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    try:
        M = v.fit_transform(sentences)
    except ValueError:
        return ""
    scores = np.asarray(M.mean(axis=0)).ravel()
    terms = np.array(v.get_feature_names_out())
    top = terms[np.argsort(scores)[::-1][:k]]
    return ", ".join(top)


# ---------- LLM cluster labels ----------

LABEL_SYSTEM = (
    "You name research themes. Given example sentences describing research gaps "
    "from multiple papers, return a short (3-7 word) noun-phrase theme label. "
    "No quotes, no punctuation at the end. The label should read like a section "
    "heading in a survey paper (e.g. 'Long-Tail Generalisation in Vision Models')."
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "string"}},
    "required": ["label"],
    "additionalProperties": False,
}


@retry(tries=3, base_delay=2.0)
def _llm_label_one(client, sentences: list[str], keywords: str, model: str) -> str:
    sample = "\n".join(f"- {s}" for s in sentences[:8])
    user = (
        f"Keywords: {keywords}\n"
        f"Example gap sentences:\n{sample}\n\n"
        "Return JSON with a single field `label`."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LABEL_SYSTEM},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "theme_label", "schema": LABEL_SCHEMA, "strict": True},
        },
        temperature=0.2,
    )
    return parse_json_response(resp.choices[0].message.content)["label"].strip()


def build_cluster_labels(
    gaps: pd.DataFrame,
    use_llm: bool = True,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """Return one row per real cluster with `theme_label` (LLM) + `keywords` (TF-IDF)."""
    client = None
    if use_llm:
        try:
            client = get_llm_client()
        except Exception as e:
            log.warning("LLM client unavailable, falling back to keyword labels: %s", e)
            client = None

    rows = []
    for cid in sorted(gaps["cluster_id"].unique()):
        if cid == -1:
            continue
        sents = gaps[gaps.cluster_id == cid]["gap_sentence"].tolist()
        keywords = keyword_label(sents, k=8)
        label = keywords  # fallback default
        if client is not None and sents:
            try:
                label = _llm_label_one(client, sents, keywords, model=model)
            except Exception as e:
                log.warning("LLM label for cluster %s failed: %s", cid, e)
        rows.append(
            {"cluster_id": int(cid), "theme_label": label, "keywords": keywords}
        )
    return pd.DataFrame(rows).sort_values("cluster_id").reset_index(drop=True)


# ---------- cluster summaries ----------

def build_cluster_summary(gaps: pd.DataFrame, label_map: dict[int, str]) -> pd.DataFrame:
    def top_examples(df, cid, n=3):
        ex = df[df.cluster_id == cid].sort_values("confidence", ascending=False).head(n)
        return ex[["id", "gap_type", "confidence", "gap_sentence"]].to_dict("records")

    cs = (
        gaps[gaps.cluster_id != -1]
        .groupby("cluster_id")
        .agg(
            n_items=("gap_sentence", "count"),
            n_papers=("id", "nunique"),
            avg_conf=("confidence", "mean"),
        )
        .reset_index()
    )
    cs["theme_label"] = cs["cluster_id"].map(label_map)
    cs["examples"] = cs["cluster_id"].apply(lambda cid: json.dumps(top_examples(gaps, cid, 3), ensure_ascii=False))
    return cs.sort_values(["n_papers", "n_items", "avg_conf"], ascending=False).reset_index(drop=True)


# ---------- bridge-score pair selection ----------

def _peak(sim: float, peak_at: float = 0.45) -> float:
    """Triangular peak: 1 at `peak_at`, linearly decaying to 0 at 0 and 1."""
    if sim <= 0 or sim >= 1:
        return 0.0
    if sim <= peak_at:
        return sim / peak_at
    return (1.0 - sim) / (1.0 - peak_at)


def _type_dist(types: Iterable[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    total = 0
    for t in types:
        counts[t] = counts.get(t, 0.0) + 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _type_complementarity(d1: dict[str, float], d2: dict[str, float]) -> float:
    """1 - L1/2 = how *different* two gap-type mixtures are. Range [0, 1]."""
    keys = set(d1) | set(d2)
    if not keys:
        return 0.0
    l1 = sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)
    return l1 / 2.0


def build_cluster_pairs(
    gaps: pd.DataFrame,
    X: np.ndarray,
    label_map: dict[int, str],
    top_n: int = 30,
    sim_peak: float = 0.45,
) -> pd.DataFrame:
    """Score every cluster pair on the bridge_score and return the top_n."""
    cids = sorted(c for c in gaps["cluster_id"].unique() if c != -1)
    if len(cids) < 2:
        return pd.DataFrame(
            columns=[
                "cluster_a", "cluster_b", "label_a", "label_b",
                "cosine_sim", "paper_overlap", "type_complementarity", "bridge_score",
            ]
        )

    centroids = {}
    papers_per_c: dict[int, set[str]] = {}
    types_per_c: dict[int, dict[str, float]] = {}
    for cid in cids:
        idx = np.where(gaps["cluster_id"].values == cid)[0]
        centroids[cid] = X[idx].mean(axis=0)
        sub = gaps.iloc[idx]
        papers_per_c[cid] = set(sub["id"].astype(str).tolist())
        types_per_c[cid] = _type_dist(sub.get("gap_type", pd.Series(dtype=str)).astype(str).tolist())

    C = np.stack([centroids[c] for c in cids])
    S = cosine_similarity(C, C)
    idx_of = {c: i for i, c in enumerate(cids)}

    pairs = []
    for ca, cb in combinations(cids, 2):
        sim = float(S[idx_of[ca], idx_of[cb]])
        pa, pb = papers_per_c[ca], papers_per_c[cb]
        overlap = len(pa & pb) / max(1, len(pa | pb))
        type_comp = _type_complementarity(types_per_c[ca], types_per_c[cb])
        bridge = _peak(sim, sim_peak) * (1.0 - overlap) * (0.5 + 0.5 * type_comp)
        pairs.append(
            {
                "cluster_a": ca,
                "cluster_b": cb,
                "label_a": label_map.get(ca, ""),
                "label_b": label_map.get(cb, ""),
                "cosine_sim": sim,
                "paper_overlap": overlap,
                "type_complementarity": type_comp,
                "bridge_score": bridge,
            }
        )
    df = pd.DataFrame(pairs).sort_values("bridge_score", ascending=False).reset_index(drop=True)
    return df.head(top_n)
