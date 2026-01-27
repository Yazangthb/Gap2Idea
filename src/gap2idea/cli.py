import argparse
import json
import numpy as np
from gap2idea.config import get_paths
from gap2idea.io import read_tsv, write_tsv
from gap2idea.pipeline.theme_mining import (
    clean_gaps, embed_sentences, cluster_embeddings,
    build_cluster_labels, build_cluster_summary, build_cluster_pairs
)
from gap2idea.pipeline.arxiv_select import select_arxiv_subset
from gap2idea.pipeline.pdf_text import load_ids_from_tsv, download_and_extract_texts
from gap2idea.pipeline.sections import extract_sections_from_pdfs
from gap2idea.pipeline.openai_gaps import extract_gaps_from_sections


def _normalize_max_papers(value: int | None) -> int | None:
    if value is None:
        return None
    return None if value <= 0 else value


def run_theme_mine(root: str, gaps_tsv: str, min_conf: float, embed_model: str, top_pairs: int) -> None:
    paths = get_paths(root)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    gaps = read_tsv(gaps_tsv)
    gaps = clean_gaps(gaps, min_conf=min_conf)

    write_tsv(gaps, paths.artifacts / "gaps_clean.tsv")

    X = embed_sentences(gaps["gap_sentence"].tolist(), model_name=embed_model)
    np.save(paths.artifacts / "gap_embeddings.npy", X)

    gaps["cluster_id"] = cluster_embeddings(X, n_points=len(gaps))
    write_tsv(gaps, paths.artifacts / "gaps_with_clusters.tsv")

    labels_df = build_cluster_labels(gaps)
    write_tsv(labels_df, paths.artifacts / "cluster_labels.tsv")
    label_map = {int(r.cluster_id): r.theme_label for r in labels_df.itertuples(index=False)}

    summary_df = build_cluster_summary(gaps, label_map)
    write_tsv(summary_df, paths.artifacts / "cluster_summary.tsv")

    pairs_df = build_cluster_pairs(gaps, X, label_map, top_n=top_pairs)
    write_tsv(pairs_df, paths.artifacts / "cluster_pairs.tsv")

    print("Wrote artifacts to:", paths.artifacts)

def cmd_theme_mine(args):
    run_theme_mine(
        root=args.root,
        gaps_tsv=args.gaps_tsv,
        min_conf=args.min_conf,
        embed_model=args.embed_model,
        top_pairs=args.top_pairs,
    )


def cmd_select_arxiv(args):
    paths = get_paths(args.root)
    paths.data.mkdir(parents=True, exist_ok=True)
    target_categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    df = select_arxiv_subset(
        metadata_path=args.metadata,
        output_path=args.output,
        target_categories=target_categories if target_categories else None,
        min_year=args.min_year,
        n_papers=args.n_papers,
        random_state=args.random_state,
    )
    print(f"Selected {len(df)} papers -> {args.output}")


def cmd_fetch_pdfs(args):
    paths = get_paths(args.root)
    ids = load_ids_from_tsv(args.papers_tsv)
    stats = download_and_extract_texts(
        ids,
        pdf_dir=paths.pdfs,
        txt_dir=paths.texts,
        download_workers=args.download_workers,
        extract_workers=args.extract_workers,
        max_pages=args.max_pages,
        min_text_chars=args.min_text_chars,
        min_pdf_bytes=args.min_pdf_bytes,
    )
    print(json.dumps(stats, indent=2))


def cmd_extract_sections(args):
    paths = get_paths(args.root)
    ids = load_ids_from_tsv(args.papers_tsv) if args.papers_tsv else None
    df = extract_sections_from_pdfs(paths.pdfs, args.output, ids=ids)
    print(f"Extracted sections: {len(df)} -> {args.output}")


def cmd_extract_gaps(args):
    max_papers = _normalize_max_papers(args.max_papers)
    df = extract_gaps_from_sections(
        sections_jsonl=args.sections_jsonl,
        out_tsv=args.output,
        model=args.model,
        max_papers=max_papers,
        flush_every=args.flush_every,
    )
    print(f"Gaps TSV rows: {len(df)} -> {args.output}")


def cmd_run_pipeline(args):
    paths = get_paths(args.root)
    paths.data.mkdir(parents=True, exist_ok=True)

    target_categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    select_arxiv_subset(
        metadata_path=args.metadata,
        output_path=args.papers_tsv,
        target_categories=target_categories if target_categories else None,
        min_year=args.min_year,
        n_papers=args.n_papers,
        random_state=args.random_state,
    )

    ids = load_ids_from_tsv(args.papers_tsv)
    stats = download_and_extract_texts(
        ids,
        pdf_dir=paths.pdfs,
        txt_dir=paths.texts,
        download_workers=args.download_workers,
        extract_workers=args.extract_workers,
        max_pages=args.max_pages,
        min_text_chars=args.min_text_chars,
        min_pdf_bytes=args.min_pdf_bytes,
    )
    print(json.dumps(stats, indent=2))

    sec_df = extract_sections_from_pdfs(paths.pdfs, args.sections_jsonl, ids=ids)
    print(f"Extracted sections: {len(sec_df)} -> {args.sections_jsonl}")

    max_papers = _normalize_max_papers(args.max_papers)
    gaps_df = extract_gaps_from_sections(
        sections_jsonl=args.sections_jsonl,
        out_tsv=args.gaps_tsv,
        model=args.model,
        max_papers=max_papers,
        flush_every=args.flush_every,
    )
    print(f"Gaps TSV rows: {len(gaps_df)} -> {args.gaps_tsv}")

    run_theme_mine(
        root=args.root,
        gaps_tsv=args.gaps_tsv,
        min_conf=args.min_conf,
        embed_model=args.embed_model,
        top_pairs=args.top_pairs,
    )

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

    s = sub.add_parser("select-arxiv")
    s.add_argument("--root", default=".")
    s.add_argument("--metadata", default="data/arxiv-metadata-oai-snapshot.json")
    s.add_argument("--output", default="data/papers_subset.tsv")
    s.add_argument("--categories", default="cs.LG,stat.ML")
    s.add_argument("--min-year", type=int, default=2021)
    s.add_argument("--n-papers", type=int, default=250)
    s.add_argument("--random-state", type=int, default=42)
    s.set_defaults(func=cmd_select_arxiv)

    f = sub.add_parser("fetch-pdfs")
    f.add_argument("--root", default=".")
    f.add_argument("--papers-tsv", default="data/papers_subset.tsv")
    f.add_argument("--download-workers", type=int, default=16)
    f.add_argument("--extract-workers", type=int, default=8)
    f.add_argument("--max-pages", type=int, default=12)
    f.add_argument("--min-text-chars", type=int, default=500)
    f.add_argument("--min-pdf-bytes", type=int, default=10_000)
    f.set_defaults(func=cmd_fetch_pdfs)

    x = sub.add_parser("extract-sections")
    x.add_argument("--root", default=".")
    x.add_argument("--papers-tsv", default=None)
    x.add_argument("--output", default="data/sections_extracted.jsonl")
    x.set_defaults(func=cmd_extract_sections)

    g = sub.add_parser("extract-gaps")
    g.add_argument("--sections-jsonl", default="data/sections_extracted.jsonl")
    g.add_argument("--output", default="data/gaps_openai.tsv")
    g.add_argument("--model", default="gpt-4.1-mini")
    g.add_argument("--max-papers", type=int, default=50)
    g.add_argument("--flush-every", type=int, default=25)
    g.set_defaults(func=cmd_extract_gaps)

    r = sub.add_parser("run-pipeline")
    r.add_argument("--root", default=".")
    r.add_argument("--metadata", default="data/arxiv-metadata-oai-snapshot.json")
    r.add_argument("--papers-tsv", default="data/papers_subset.tsv")
    r.add_argument("--categories", default="cs.LG,stat.ML")
    r.add_argument("--min-year", type=int, default=2021)
    r.add_argument("--n-papers", type=int, default=250)
    r.add_argument("--random-state", type=int, default=42)
    r.add_argument("--download-workers", type=int, default=16)
    r.add_argument("--extract-workers", type=int, default=8)
    r.add_argument("--max-pages", type=int, default=12)
    r.add_argument("--min-text-chars", type=int, default=500)
    r.add_argument("--min-pdf-bytes", type=int, default=10_000)
    r.add_argument("--sections-jsonl", default="data/sections_extracted.jsonl")
    r.add_argument("--gaps-tsv", default="data/gaps_openai.tsv")
    r.add_argument("--model", default="gpt-4.1-mini")
    r.add_argument("--max-papers", type=int, default=50)
    r.add_argument("--flush-every", type=int, default=25)
    r.add_argument("--min-conf", type=float, default=0.5)
    r.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    r.add_argument("--top-pairs", type=int, default=30)
    r.set_defaults(func=cmd_run_pipeline)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
