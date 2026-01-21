import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_gaps(gaps: pd.DataFrame, min_conf: float = 0.5) -> pd.DataFrame:
    gaps = gaps.copy()
    for c in ["gap_sentence", "paragraph_text"]:
        gaps[c] = gaps[c].astype(str).str.replace("\n", " ").str.strip()
    gaps["confidence"] = pd.to_numeric(gaps["confidence"], errors="coerce").fillna(0.0)
    gaps = gaps[(gaps["confidence"] >= min_conf) & (gaps["gap_sentence"].str.len() >= 20)]
    gaps = gaps.drop_duplicates(subset=["id", "gap_sentence"]).reset_index(drop=True)
    return gaps

def embed_sentences(sentences: list[str], model_name="all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(model_name)
    X = embedder.encode(sentences, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return X

def cluster_embeddings(X: np.ndarray, n_points: int) -> np.ndarray:
    # small data -> kmeans, large -> hdbscan
    if n_points < 150:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        k_candidates = list(range(4, min(12, n_points)))
        best_k, best_s = None, -1
        for k in k_candidates:
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            lbl = km.fit_predict(X)
            s = silhouette_score(X, lbl)
            if s > best_s:
                best_k, best_s = k, s
        km = KMeans(n_clusters=best_k, random_state=42, n_init="auto")
        return km.fit_predict(X)

    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(8, n_points // 40),
        min_samples=2,
        metric="euclidean",
    )
    return clusterer.fit_predict(X)

def keyword_label(sentences: list[str], k=6) -> str:
    v = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_features=2000)
    M = v.fit_transform(sentences)
    scores = np.asarray(M.mean(axis=0)).ravel()
    terms = np.array(v.get_feature_names_out())
    top = terms[np.argsort(scores)[::-1][:k]]
    return ", ".join(top)

def build_cluster_labels(gaps: pd.DataFrame) -> pd.DataFrame:
    labels = []
    for cid in sorted(gaps["cluster_id"].unique()):
        if cid == -1:
            continue
        sents = gaps[gaps.cluster_id == cid]["gap_sentence"].tolist()
        labels.append({"cluster_id": int(cid), "theme_label": keyword_label(sents, k=6)})
    return pd.DataFrame(labels).sort_values("cluster_id")

def build_cluster_summary(gaps: pd.DataFrame, cluster_labels: dict[int, str]) -> pd.DataFrame:
    def top_examples(df, cid, n=3):
        ex = df[df.cluster_id == cid].sort_values("confidence", ascending=False).head(n)
        return ex[["id","gap_type","confidence","gap_sentence"]].to_dict("records")

    cs = (
        gaps[gaps.cluster_id != -1]
        .groupby("cluster_id")
        .agg(
            n_items=("gap_sentence","count"),
            n_papers=("id","nunique"),
            avg_conf=("confidence","mean"),
        )
        .reset_index()
    )
    cs["theme_label"] = cs["cluster_id"].map(cluster_labels)
    cs["examples"] = cs["cluster_id"].apply(lambda cid: top_examples(gaps, cid, 3))
    return cs.sort_values(["n_papers","n_items","avg_conf"], ascending=False).reset_index(drop=True)

def build_cluster_pairs(gaps: pd.DataFrame, X: np.ndarray, cluster_labels: dict[int, str], top_n: int = 30) -> pd.DataFrame:
    centroids = {}
    for cid in sorted(gaps["cluster_id"].unique()):
        if cid == -1:
            continue
        idx = np.where(gaps["cluster_id"].values == cid)[0]
        centroids[int(cid)] = X[idx].mean(axis=0)

    cids = sorted(centroids.keys())
    if len(cids) < 2:
        return pd.DataFrame(columns=["cluster_a","cluster_b","cosine_sim","label_a","label_b"])

    C = np.stack([centroids[c] for c in cids])
    S = cosine_similarity(C, C)

    pairs = []
    for i, ca in enumerate(cids):
        for j in range(i+1, len(cids)):
            cb = cids[j]
            pairs.append((ca, cb, float(S[i, j])))

    pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:top_n]
    df = pd.DataFrame(pairs, columns=["cluster_a","cluster_b","cosine_sim"])
    df["label_a"] = df["cluster_a"].map(cluster_labels)
    df["label_b"] = df["cluster_b"].map(cluster_labels)
    return df