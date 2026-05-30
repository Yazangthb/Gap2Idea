"""Clustering-quality benchmark for gap-statement themes.

Grid: clusterer x embedding backbone.

Metrics per cell:
  intrinsic geometry      silhouette (cosine), Davies-Bouldin, Calinski-Harabasz
  topic coherence         NPMI averaged over top-N tokens per cluster
                          (gensim's CoherenceModel would be the standard
                           library, but it does not build on Python 3.14;
                           NPMI is implemented inline here)
  stability               mean+std Adjusted Rand Index over `n_bootstrap`
                          80%-with-replacement resamples versus full-corpus labels

The bench is parametric in the corpus (`--gaps-tsv`). On tiny corpora
(< 30 statements) the intrinsic scores are noisy by construction; we
still emit them so plumbing is verified end-to-end.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)

DEFAULT_EMBEDDERS = (
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "intfloat/e5-base-v2",
    "BAAI/bge-small-en-v1.5",
)
DEFAULT_CLUSTERERS = ("kmeans", "agglomerative", "hdbscan", "hdbscan_umap", "bertopic")
TOPIC_TOP_N = 10
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


@dataclass
class GridCell:
    clusterer: str
    embedder: str
    labels: np.ndarray | None = None
    n_clusters: int = 0
    n_noise: int = 0
    notes: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------

def load_gaps(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    sents = df["gap_sentence"].astype(str).tolist()
    log.info("Loaded %d gap sentences from %s", len(sents), path)
    return sents


# ----------------------------------------------------------------------
# Embedding (cached per-model on disk)
# ----------------------------------------------------------------------

def embed_texts(texts: list[str], model_name: str, cache_dir: Path) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"emb__{model_name.replace('/', '_')}.npy"
    if cache.exists():
        X = np.load(cache)
        if X.shape[0] == len(texts):
            log.info("Reusing cached embeddings %s (shape %s)", cache, X.shape)
            return X
    log.info("Encoding %d texts with %s", len(texts), model_name)
    model = SentenceTransformer(model_name)
    X = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                     show_progress_bar=False)
    np.save(cache, X)
    return X


# ----------------------------------------------------------------------
# Clustering
# ----------------------------------------------------------------------

def _sweep_k(X: np.ndarray, k_max: int) -> int:
    """Pick k by silhouette over [2, k_max]. Fallback to 2 on tiny corpora."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best_k, best_s = 2, -1.0
    for k in range(2, max(3, k_max + 1)):
        if k >= X.shape[0]:
            break
        km = KMeans(n_clusters=k, n_init=5, random_state=0).fit(X)
        try:
            s = silhouette_score(X, km.labels_, metric="cosine")
        except ValueError:
            continue
        if s > best_s:
            best_s, best_k = s, k
    return best_k


def fit_clusters(X: np.ndarray, texts: list[str], method: str) -> tuple[np.ndarray, list[str]]:
    """Return labels and any notes. Labels of -1 mean "noise" for HDBSCAN."""
    n = X.shape[0]
    k_max = min(8, max(2, n // 2))
    notes: list[str] = []

    if method == "kmeans":
        from sklearn.cluster import KMeans
        k = _sweep_k(X, k_max)
        labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(X)
        notes.append(f"k={k}")

    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        k = _sweep_k(X, k_max)
        labels = AgglomerativeClustering(n_clusters=k, metric="cosine",
                                         linkage="average").fit_predict(X)
        notes.append(f"k={k}")

    elif method == "hdbscan":
        import hdbscan
        mcs = max(2, min(5, n // 4))
        clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
        labels = clusterer.fit_predict(X)
        notes.append(f"min_cluster_size={mcs}")

    elif method == "hdbscan_umap":
        # Mitigate the curse of dimensionality: UMAP to ~10d first, then HDBSCAN.
        # Standard recipe from the BERTopic/HDBSCAN documentation.
        import hdbscan
        from umap import UMAP
        umap_n_components = min(10, max(2, n - 1))
        umap_n_neighbors = max(2, min(15, n - 1))
        try:
            X_low = UMAP(n_neighbors=umap_n_neighbors, n_components=umap_n_components,
                         metric="cosine", random_state=0).fit_transform(X)
            mcs = max(2, min(5, n // 4))
            labels = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean") \
                            .fit_predict(X_low)
            notes.append(f"umap_dim={umap_n_components} min_cluster_size={mcs}")
        except Exception as e:  # noqa: BLE001
            notes.append(f"hdbscan_umap failed: {e}")
            labels = np.full((n,), -1)

    elif method == "bertopic":
        from bertopic import BERTopic
        from umap import UMAP
        # Force a small UMAP that won't crash on tiny corpora
        umap_n = max(2, min(5, n - 1))
        umap = UMAP(n_neighbors=max(2, min(5, n - 1)), n_components=min(5, umap_n),
                    metric="cosine", random_state=0)
        try:
            topic_model = BERTopic(umap_model=umap, min_topic_size=max(2, n // 10),
                                   verbose=False, calculate_probabilities=False)
            labels, _ = topic_model.fit_transform(texts, embeddings=X)
            labels = np.asarray(labels)
        except Exception as e:
            notes.append(f"bertopic failed: {e}")
            labels = np.full((n,), -1)

    elif method == "leiden_graph":
        # PR-5: gap-graph community detection (Leiden if leidenalg installed,
        # NetworkX Louvain fallback). The benchmark feeds only `texts` + `X`
        # so we synthesise a minimal gaps frame; paper/section/method edges
        # collapse to empty and the graph reduces to a kNN semantic graph,
        # which is precisely what isolates the community detector for
        # head-to-head comparison.
        import pandas as pd

        from gap2idea.pipeline import gap_graph as GG

        synth_gaps = pd.DataFrame([
            {"id": f"row_{i}", "gap_type": "", "section_type": "",
             "gap_sentence": t, "paragraph_text": "", "confidence": 1.0}
            for i, t in enumerate(texts)
        ])
        try:
            X_norm = X / np.maximum(1e-9, np.linalg.norm(X, axis=1, keepdims=True))
            G = GG.build_gap_graph(synth_gaps, X_norm, knn_k=8, sim_threshold=0.3)
            labels = GG.detect_communities(G, resolution=1.0, seed=0)
            notes.append(f"n_communities={int(labels.max()) + 1 if labels.size else 0}")
        except Exception as e:
            notes.append(f"leiden_graph failed: {e}")
            labels = np.full((n,), -1)

    else:
        raise ValueError(f"unknown clusterer: {method}")

    return labels, notes


# ----------------------------------------------------------------------
# Intrinsic geometry
# ----------------------------------------------------------------------

def intrinsic_scores(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Silhouette (cosine), Davies-Bouldin, Calinski-Harabasz.

    Requires ≥ 2 non-noise clusters and ≥ 1 point per cluster.
    Returns NaN for any metric that cannot be computed.
    """
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)
    mask = labels != -1
    if mask.sum() < 4 or len(np.unique(labels[mask])) < 2:
        return {"silhouette": float("nan"),
                "davies_bouldin": float("nan"),
                "calinski_harabasz": float("nan")}
    Xm, Lm = X[mask], labels[mask]
    out: dict[str, float] = {}
    for name, fn, kw in (
        ("silhouette", silhouette_score, {"metric": "cosine"}),
        ("davies_bouldin", davies_bouldin_score, {}),
        ("calinski_harabasz", calinski_harabasz_score, {}),
    ):
        try:
            out[name] = float(fn(Xm, Lm, **kw))
        except Exception:
            out[name] = float("nan")
    return out


# ----------------------------------------------------------------------
# Topic coherence — inline NPMI (gensim doesn't build on py3.14)
# ----------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def _doc_word_sets(texts: list[str]) -> list[set[str]]:
    return [set(_tokenize(t)) for t in texts]


def _cluster_top_words(texts: list[str], labels: np.ndarray, top_n: int,
                       stop: frozenset[str]) -> dict[int, list[str]]:
    """Per-cluster top-N tokens by frequency (with stopword + length filters)."""
    by_cluster: dict[int, Counter] = {}
    for t, lab in zip(texts, labels):
        if lab == -1:
            continue
        cnt = by_cluster.setdefault(int(lab), Counter())
        for tok in _tokenize(t):
            if tok in stop or len(tok) < 4:
                continue
            cnt[tok] += 1
    return {c: [w for w, _ in cnt.most_common(top_n)] for c, cnt in by_cluster.items()}


def npmi_score(texts: list[str], labels: np.ndarray, top_n: int = TOPIC_TOP_N) -> float:
    """Mean pairwise NPMI over top-N words within each cluster.

    Reference corpus = the input texts themselves (each text = one "document").
    Returns NaN if no clusters / fewer than 2 words per cluster.
    """
    docs = _doc_word_sets(texts)
    n_docs = len(docs)
    if n_docs < 2:
        return float("nan")

    # Frequencies in the reference corpus
    df_count: Counter = Counter()
    for s in docs:
        df_count.update(s)
    co_count: dict[tuple[str, str], int] = {}

    stop = frozenset(["that", "with", "this", "from", "have", "into", "such",
                      "their", "these", "than", "when", "which", "would", "where",
                      "while", "been", "more", "most", "some", "what", "thus",
                      "they", "them", "also", "since", "between", "based", "using"])

    cluster_words = _cluster_top_words(texts, labels, top_n, stop)
    if not cluster_words:
        return float("nan")

    scores: list[float] = []
    for words in cluster_words.values():
        if len(words) < 2:
            continue
        pair_scores: list[float] = []
        for i, wi in enumerate(words):
            for wj in words[i + 1:]:
                a, b = (wi, wj) if wi < wj else (wj, wi)
                if (a, b) not in co_count:
                    co_count[(a, b)] = sum(1 for s in docs if a in s and b in s)
                co = co_count[(a, b)]
                if co == 0 or df_count[a] == 0 or df_count[b] == 0:
                    continue
                p_a, p_b, p_ab = df_count[a] / n_docs, df_count[b] / n_docs, co / n_docs
                pmi = math.log(p_ab / (p_a * p_b))
                npmi = pmi / (-math.log(p_ab))
                pair_scores.append(npmi)
        if pair_scores:
            scores.append(sum(pair_scores) / len(pair_scores))

    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))


# ----------------------------------------------------------------------
# Stability — bootstrap ARI
# ----------------------------------------------------------------------

def bootstrap_ari(X: np.ndarray, texts: list[str], method: str, full_labels: np.ndarray,
                  n_bootstrap: int = 10, sample_frac: float = 0.8,
                  seed: int = 0) -> tuple[float, float, int]:
    """Resample with replacement, recluster, measure ARI vs labels at same indices."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    size = max(4, int(n * sample_frac))
    scores: list[float] = []
    for b in range(n_bootstrap):
        idx = rng.choice(n, size=size, replace=True)
        Xb = X[idx]
        tb = [texts[i] for i in idx]
        try:
            lb, _ = fit_clusters(Xb, tb, method)
        except Exception:
            continue
        ref = full_labels[idx]
        # ARI is invariant to label permutation; -1 is treated as just another label.
        try:
            scores.append(float(adjusted_rand_score(ref, lb)))
        except Exception:
            continue
    if not scores:
        return float("nan"), float("nan"), 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def run_benchmark(
    gaps_tsv: Path,
    out_dir: Path,
    clusterers: Iterable[str] = DEFAULT_CLUSTERERS,
    embedders: Iterable[str] = DEFAULT_EMBEDDERS,
    n_bootstrap: int = 10,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "embeddings"

    texts = load_gaps(gaps_tsv)
    if len(texts) < 6:
        log.warning("Only %d gap sentences — intrinsic metrics will be noisy.", len(texts))

    rows: list[dict] = []
    for emb in embedders:
        X = embed_texts(texts, emb, cache_dir)
        for cls in clusterers:
            log.info("=== %s × %s ===", cls, emb)
            try:
                labels, notes = fit_clusters(X, texts, cls)
            except Exception as e:  # noqa: BLE001
                log.warning("  fit failed: %s", e)
                rows.append({"clusterer": cls, "embedder": emb, "metric": "n_clusters",
                             "value": 0, "extra": f"fit_failed: {e}"})
                continue

            uniq = [u for u in np.unique(labels) if u != -1]
            n_clusters = len(uniq)
            n_noise = int((labels == -1).sum())

            geom = intrinsic_scores(X, labels)
            npmi = npmi_score(texts, labels)
            mean_ari, std_ari, k_runs = bootstrap_ari(
                X, texts, cls, labels, n_bootstrap=n_bootstrap, seed=0,
            )

            note_str = ";".join(notes)
            base = {"clusterer": cls, "embedder": emb, "extra": note_str}
            metric_items: list[tuple[str, float]] = [
                ("n_clusters", float(n_clusters)),
                ("n_noise", float(n_noise)),
                *((k, float(v)) for k, v in geom.items()),
                ("npmi", npmi),
                ("bootstrap_mean_ari", mean_ari),
                ("bootstrap_std_ari", std_ari),
                ("bootstrap_n_runs", float(k_runs)),
            ]
            for k, v in metric_items:
                rows.append({**base, "metric": k, "value": float(v)})
            log.info("  n_clusters=%d  silhouette=%.3f  npmi=%.3f  ari=%.3f±%.3f",
                     n_clusters, geom["silhouette"], npmi, mean_ari, std_ari)

    df = pd.DataFrame(rows)
    metrics_path = out_dir / "metrics.tsv"
    df.to_csv(metrics_path, sep="\t", index=False)
    log.info("Wrote %s", metrics_path)

    write_report(df, texts, out_dir / "REPORT.md")

    try:
        from gap2idea.pipeline.clustering_bench_plots import make_all_plots
        make_all_plots(out_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("Plot generation failed: %s", e)

    return metrics_path


def write_report(metrics: pd.DataFrame, texts: list[str], out_md: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Clustering-quality benchmark (N={len(texts)} gap sentences)\n")
    if len(texts) < 30:
        lines.append("> ⚠ Corpus is small (< 30 statements). Intrinsic scores are noisy "
                     "and bootstrap ARI may swing wildly. Use a larger gaps.tsv for "
                     "publication-grade numbers.\n")

    headline = ["silhouette", "davies_bouldin", "calinski_harabasz", "npmi",
                "bootstrap_mean_ari", "n_clusters"]
    for metric in headline:
        sub = metrics[metrics["metric"] == metric]
        if sub.empty:
            continue
        pivot = sub.pivot(index="clusterer", columns="embedder", values="value")
        lines.append(f"\n## {metric}\n")
        lines.append(pivot.round(3).to_markdown())
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_md)
