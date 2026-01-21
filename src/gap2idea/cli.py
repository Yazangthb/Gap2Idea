import argparse
import numpy as np
from gap2idea.config import get_paths
from gap2idea.io import read_tsv, write_tsv
from gap2idea.pipeline.theme_mining import (
    clean_gaps, embed_sentences, cluster_embeddings,
    build_cluster_labels, build_cluster_summary, build_cluster_pairs
)

def cmd_theme_mine(args):
    paths = get_paths(args.root)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    gaps = read_tsv(args.gaps_tsv)
    gaps = clean_gaps(gaps, min_conf=args.min_conf)

    write_tsv(gaps, paths.artifacts / "gaps_clean.tsv")

    X = embed_sentences(gaps["gap_sentence"].tolist(), model_name=args.embed_model)
    np.save(paths.artifacts / "gap_embeddings.npy", X)

    gaps["cluster_id"] = cluster_embeddings(X, n_points=len(gaps))
    write_tsv(gaps, paths.artifacts / "gaps_with_clusters.tsv")

    labels_df = build_cluster_labels(gaps)
    write_tsv(labels_df, paths.artifacts / "cluster_labels.tsv")
    label_map = {int(r.cluster_id): r.theme_label for r in labels_df.itertuples(index=False)}

    summary_df = build_cluster_summary(gaps, label_map)
    write_tsv(summary_df, paths.artifacts / "cluster_summary.tsv")

    pairs_df = build_cluster_pairs(gaps, X, label_map, top_n=args.top_pairs)
    write_tsv(pairs_df, paths.artifacts / "cluster_pairs.tsv")

    print("Wrote artifacts to:", paths.artifacts)

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("theme-mine")
    t.add_argument("--root", default=".")
    t.add_argument("--gaps-tsv", required=True)
    t.add_argument("--min-conf", type=float, default=0.5)
    t.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    t.add_argument("--top-pairs", type=int, default=30)
    t.set_defaults(func=cmd_theme_mine)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()