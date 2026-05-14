"""End-to-end CLI for Gap2Idea.

Stages:

  select-papers     query a corpus from Semantic Scholar (or sample a local
                    arxiv-metadata snapshot) and write data/papers_subset.tsv
  download-pdfs     fetch arxiv PDFs into data/pdfs/
  extract-text      PDFs -> data/paper_texts.jsonl  (PyMuPDF)
  extract-sections  paper_texts.jsonl -> data/sections_extracted.jsonl
                    (Limitations / Future Work / Discussion finder)
  extract-gaps      sections -> data/gaps.tsv   (OpenAI structured outputs)
  theme-mine        gaps.tsv -> artifacts/{embeddings, clusters, labels,
                    cluster_summary, cluster_pairs}.* using bridge-scoring
  fetch-metadata    enrich every paper ID with Semantic Scholar metadata
  generate-ideas    cluster_pairs + gaps -> artifacts/{ideas.tsv, ideas_full.jsonl}
  evaluate-ideas    ideas_full.jsonl -> artifacts/idea_eval.tsv + report.md
  run-all           orchestrate every stage from `extract-text` onwards
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gap2idea.config import get_paths
from gap2idea.io import read_tsv, write_tsv
from gap2idea.utils import get_logger, set_seed

log = get_logger("gap2idea.cli")


# ---------- select-papers ----------

def cmd_select_papers(args):
    from gap2idea.pipeline.arxiv_select import (
        select_from_semantic_scholar, select_from_snapshot,
    )

    paths = get_paths(args.root)
    paths.data.mkdir(parents=True, exist_ok=True)

    if args.source == "s2":
        df = select_from_semantic_scholar(query=args.query, n=args.n)
    else:
        snap = Path(args.snapshot) if args.snapshot else paths.data / "arxiv-metadata-oai-snapshot.json"
        df = select_from_snapshot(
            snap, target_cats=set(args.cats.split(",")),
            min_year=args.min_year, n=args.n, seed=args.seed,
        )
    out = paths.data / "papers_subset.tsv"
    write_tsv(df, out)
    log.info("Wrote %d papers to %s", len(df), out)


# ---------- download-pdfs ----------

def cmd_download_pdfs(args):
    from gap2idea.pipeline.arxiv_select import download_pdfs

    paths = get_paths(args.root)
    papers = read_tsv(args.papers_tsv or (paths.data / "papers_subset.tsv"))
    arxiv_ids = papers["id"].astype(str).tolist()
    res = download_pdfs(arxiv_ids, paths.pdfs, max_workers=args.workers)
    write_tsv(res, paths.data / "pdf_download_log.tsv")


# ---------- extract-text ----------

def cmd_extract_text(args):
    from gap2idea.pipeline.pdf_text import extract_all

    paths = get_paths(args.root)
    extract_all(
        pdfs_dir=paths.pdfs,
        out_jsonl=paths.data / "paper_texts.jsonl",
        max_workers=args.workers,
        max_pages=args.max_pages,
    )


# ---------- extract-sections ----------

def cmd_extract_sections(args):
    from gap2idea.pipeline.sections import extract_all_sections

    paths = get_paths(args.root)
    extract_all_sections(
        texts_jsonl=paths.data / "paper_texts.jsonl",
        out_jsonl=paths.data / "sections_extracted.jsonl",
    )


# ---------- extract-gaps ----------

def cmd_extract_gaps(args):
    from gap2idea.pipeline.openai_gaps import extract_gaps

    paths = get_paths(args.root)
    extract_gaps(
        sections_jsonl=paths.data / "sections_extracted.jsonl",
        out_tsv=paths.data / "gaps.tsv",
        model=args.model,
        resume=not args.no_resume,
    )


# ---------- extract-methods ----------

def cmd_extract_methods(args):
    """Mine method-claim sentences from abstracts/intros. Output feeds
    `generate-ideas --mode method-gap`."""
    from gap2idea.pipeline.openai_methods import extract_methods

    paths = get_paths(args.root)
    extract_methods(
        texts_jsonl=paths.data / "paper_texts.jsonl",
        out_tsv=paths.data / "methods.tsv",
        model=args.model,
        resume=not args.no_resume,
    )


# ---------- theme-mine ----------

def cmd_theme_mine(args):
    from gap2idea.pipeline.theme_mining import (
        build_cluster_labels, build_cluster_pairs, build_cluster_summary,
        clean_gaps, cluster_embeddings, embed_sentences,
    )

    set_seed(args.seed)
    paths = get_paths(args.root)
    paths.artifacts.mkdir(parents=True, exist_ok=True)

    gaps_path = Path(args.gaps_tsv) if args.gaps_tsv else paths.data / "gaps.tsv"
    gaps = read_tsv(gaps_path)
    gaps = clean_gaps(gaps, min_conf=args.min_conf)
    write_tsv(gaps, paths.artifacts / "gaps_clean.tsv")

    X = embed_sentences(gaps["gap_sentence"].tolist(), model_name=args.embed_model)
    np.save(paths.artifacts / "gap_embeddings.npy", X)

    gaps["cluster_id"] = cluster_embeddings(X, n_points=len(gaps))
    write_tsv(gaps, paths.artifacts / "gaps_with_clusters.tsv")

    labels_df = build_cluster_labels(gaps, use_llm=not args.no_llm_labels, model=args.llm_label_model)
    write_tsv(labels_df, paths.artifacts / "cluster_labels.tsv")
    label_map = dict(zip(labels_df["cluster_id"].astype(int), labels_df["theme_label"]))

    summary_df = build_cluster_summary(gaps, label_map)
    write_tsv(summary_df, paths.artifacts / "cluster_summary.tsv")

    pairs_df = build_cluster_pairs(gaps, X, label_map, top_n=args.top_pairs, sim_peak=args.sim_peak)
    write_tsv(pairs_df, paths.artifacts / "cluster_pairs.tsv")

    log.info("Wrote artifacts to: %s", paths.artifacts)


# ---------- fetch-metadata ----------

def cmd_fetch_metadata(args):
    from gap2idea.pipeline.semantic_scholar import S2Client, flatten_paper

    paths = get_paths(args.root)
    paths.artifacts.mkdir(parents=True, exist_ok=True)
    gaps = read_tsv(args.gaps_tsv or (paths.artifacts / "gaps_with_clusters.tsv"))
    arxiv_ids = sorted({str(x).strip() for x in gaps["id"].tolist() if str(x).strip()})
    log.info("Fetching S2 metadata for %d unique papers...", len(arxiv_ids))

    client = S2Client()
    raw = client.get_batch_by_arxiv(arxiv_ids, chunk_size=args.chunk_size)
    rows = [flatten_paper(rec, fallback_id=aid) for aid, rec in zip(arxiv_ids, raw)]
    df = pd.DataFrame(rows)
    out = paths.artifacts / "papers_metadata.tsv"
    write_tsv(df, out)
    n_hits = int((df["title"] != "").sum())
    log.info("Wrote %d rows (%d with non-empty titles) to %s", len(df), n_hits, out)


# ---------- generate-ideas ----------

def cmd_generate_ideas(args):
    paths = get_paths(args.root)
    gaps = read_tsv(paths.artifacts / "gaps_with_clusters.tsv")
    labels = read_tsv(paths.artifacts / "cluster_labels.tsv")
    out_tsv = paths.artifacts / "ideas.tsv"
    out_jsonl = paths.artifacts / "ideas_full.jsonl"

    if args.mode == "bridge":
        from gap2idea.pipeline.openai_ideas import generate_ideas_batch
        pairs = read_tsv(paths.artifacts / "cluster_pairs.tsv")
        generate_ideas_batch(
            gaps=gaps, pairs=pairs, cluster_labels=labels,
            out_tsv=out_tsv, out_jsonl=out_jsonl,
            n_pairs=args.n_pairs, model=args.model,
            check_novelty=not args.no_novelty, k_evidence=args.k_evidence,
        )
    elif args.mode == "within":
        from gap2idea.pipeline.openai_ideas import generate_ideas_within_clusters
        generate_ideas_within_clusters(
            gaps=gaps, cluster_labels=labels,
            out_tsv=out_tsv, out_jsonl=out_jsonl,
            n_clusters=args.n_pairs,  # reuse arg as "max ideas to generate"
            model=args.model,
            check_novelty=not args.no_novelty,
            k_evidence=max(args.k_evidence, 6),
        )
    elif args.mode == "method-gap":
        from gap2idea.pipeline.openai_ideas import generate_ideas_method_gap

        gap_embeddings = np.load(paths.artifacts / "gap_embeddings.npy")
        methods_tsv = paths.data / "methods.tsv"
        if not methods_tsv.exists():
            raise SystemExit(
                "method-gap mode requires data/methods.tsv. Run:\n"
                "  gap2idea extract-methods"
            )
        methods = read_tsv(methods_tsv)

        # Embed methods on demand (cached as artifact)
        method_emb_path = paths.artifacts / "method_embeddings.npy"
        if method_emb_path.exists() and len(np.load(method_emb_path)) == len(methods):
            method_embeddings = np.load(method_emb_path)
        else:
            from gap2idea.pipeline.theme_mining import embed_sentences
            method_embeddings = embed_sentences(methods["method_sentence"].tolist())
            np.save(method_emb_path, method_embeddings)

        generate_ideas_method_gap(
            gaps=gaps, gap_embeddings=gap_embeddings,
            methods=methods, method_embeddings=method_embeddings,
            cluster_labels=labels,
            out_tsv=out_tsv, out_jsonl=out_jsonl,
            n_clusters=args.n_pairs, model=args.model,
            check_novelty=not args.no_novelty,
            k_gap_evidence=args.k_evidence,
            k_method_evidence=args.k_methods,
            sim_low=args.sim_low, sim_high=args.sim_high,
        )
    else:
        raise SystemExit(f"Unknown --mode: {args.mode}")


# ---------- evaluate-ideas ----------

def cmd_evaluate_ideas(args):
    from gap2idea.pipeline.evaluation import evaluate_ideas, write_report

    paths = get_paths(args.root)
    eval_df = evaluate_ideas(
        ideas_jsonl=paths.artifacts / "ideas_full.jsonl",
        out_tsv=paths.artifacts / "idea_eval.tsv",
        judge_model=args.judge_model,
    )
    ideas_df = pd.DataFrame()
    try:
        ideas_df = read_tsv(paths.artifacts / "ideas.tsv")
    except Exception:
        pass
    write_report(eval_df, ideas_df, paths.artifacts / "evaluation_report.md")


# ---------- run-all ----------

def cmd_run_all(args):
    """End-to-end from PDFs onwards. Skips select-papers/download-pdfs since
    those have side effects you may want to gate manually."""
    log.info("=== STAGE: extract-text ===")
    cmd_extract_text(args)
    log.info("=== STAGE: extract-sections ===")
    cmd_extract_sections(args)
    log.info("=== STAGE: extract-gaps ===")
    cmd_extract_gaps(args)
    log.info("=== STAGE: theme-mine ===")
    cmd_theme_mine(args)
    log.info("=== STAGE: fetch-metadata ===")
    cmd_fetch_metadata(args)
    log.info("=== STAGE: generate-ideas ===")
    cmd_generate_ideas(args)
    log.info("=== STAGE: evaluate-ideas ===")
    cmd_evaluate_ideas(args)
    log.info("=== DONE ===")


# ---------- main ----------

def main():
    p = argparse.ArgumentParser(prog="gap2idea")
    p.add_argument("--root", default=".")
    sub = p.add_subparsers(dest="cmd", required=True)

    # select-papers
    sp = sub.add_parser("select-papers", help="Pick a paper corpus")
    sp.add_argument("--source", choices=["s2", "snapshot"], default="s2")
    sp.add_argument("--query", default="machine learning")
    sp.add_argument("--snapshot", default=None, help="Path to arxiv-metadata-oai-snapshot.json")
    sp.add_argument("--cats", default="cs.LG,stat.ML")
    sp.add_argument("--min-year", type=int, default=2021)
    sp.add_argument("--n", type=int, default=100)
    sp.add_argument("--seed", type=int, default=42)
    sp.set_defaults(func=cmd_select_papers)

    # download-pdfs
    dp = sub.add_parser("download-pdfs", help="Download PDFs from arXiv")
    dp.add_argument("--papers-tsv", default=None)
    dp.add_argument("--workers", type=int, default=8)
    dp.set_defaults(func=cmd_download_pdfs)

    # extract-text
    et = sub.add_parser("extract-text", help="Extract text from every PDF in data/pdfs")
    et.add_argument("--workers", type=int, default=8)
    et.add_argument("--max-pages", type=int, default=None)
    et.set_defaults(func=cmd_extract_text)

    # extract-sections
    es = sub.add_parser("extract-sections", help="Find limitations/future_work sections in paper texts")
    es.set_defaults(func=cmd_extract_sections)

    # extract-gaps
    eg = sub.add_parser("extract-gaps", help="LLM gap extraction over sections")
    eg.add_argument("--model", default="openai/gpt-4.1-mini",
                    help="OpenRouter model slug, e.g. openai/gpt-4.1-mini, anthropic/claude-sonnet-4")
    eg.add_argument("--no-resume", action="store_true")
    eg.set_defaults(func=cmd_extract_gaps)

    # extract-methods
    em = sub.add_parser("extract-methods", help="LLM method-claim extraction over abstracts/intros")
    em.add_argument("--model", default="openai/gpt-4.1-mini")
    em.add_argument("--no-resume", action="store_true")
    em.set_defaults(func=cmd_extract_methods)

    # theme-mine
    tm = sub.add_parser("theme-mine", help="Embed -> cluster -> label -> pair")
    tm.add_argument("--gaps-tsv", default=None)
    tm.add_argument("--min-conf", type=float, default=0.5)
    tm.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    tm.add_argument("--top-pairs", type=int, default=30)
    tm.add_argument("--sim-peak", type=float, default=0.45,
                    help="Cosine similarity that maximises bridge_score")
    tm.add_argument("--no-llm-labels", action="store_true",
                    help="Skip OpenAI cluster labels; use TF-IDF keywords only")
    tm.add_argument("--llm-label-model", default="openai/gpt-4.1-mini",
                    help="OpenRouter model slug for cluster-label generation")
    tm.add_argument("--seed", type=int, default=42)
    tm.set_defaults(func=cmd_theme_mine)

    # fetch-metadata
    fm = sub.add_parser("fetch-metadata", help="Fetch paper metadata from Semantic Scholar")
    fm.add_argument("--gaps-tsv", default=None)
    fm.add_argument("--chunk-size", type=int, default=100)
    fm.set_defaults(func=cmd_fetch_metadata)

    # generate-ideas
    gi = sub.add_parser("generate-ideas", help="Generate research ideas with novelty check")
    gi.add_argument(
        "--mode", choices=["bridge", "within", "method-gap"], default="bridge",
        help=(
            "bridge: pair gap-clusters in the bridge-score sweet spot (default). "
            "within: synthesise one idea per cluster from its gaps. "
            "method-gap: apply retrieved methods (data/methods.tsv) to each gap-cluster."
        ),
    )
    gi.add_argument("--n-pairs", type=int, default=10,
                    help="Max ideas to generate (interpreted as #pairs / #clusters depending on mode)")
    gi.add_argument("--model", default="openai/gpt-4.1-mini",
                    help="OpenRouter model slug for idea generation")
    gi.add_argument("--no-novelty", action="store_true")
    gi.add_argument("--k-evidence", type=int, default=4,
                    help="Gap-evidence rows per cluster fed to the LLM")
    gi.add_argument("--k-methods", type=int, default=5,
                    help="(method-gap only) candidate method statements per cluster")
    gi.add_argument("--sim-low", type=float, default=0.30,
                    help="(method-gap only) lower bound of method <-> gap-cluster similarity sweet spot")
    gi.add_argument("--sim-high", type=float, default=0.70,
                    help="(method-gap only) upper bound of method <-> gap-cluster similarity sweet spot")
    gi.set_defaults(func=cmd_generate_ideas)

    # evaluate-ideas
    ev = sub.add_parser("evaluate-ideas", help="LLM-as-judge rubric scoring")
    ev.add_argument("--judge-model", default="anthropic/claude-sonnet-4",
                    help="OpenRouter model slug for the judge. Default differs from generator to mitigate self-eval bias.")
    ev.set_defaults(func=cmd_evaluate_ideas)

    # run-all
    ra = sub.add_parser("run-all", help="Run extract-text through evaluate-ideas")
    ra.add_argument("--model", default="openai/gpt-4.1-mini")
    ra.add_argument("--no-resume", action="store_true")
    ra.add_argument("--gaps-tsv", default=None)
    ra.add_argument("--min-conf", type=float, default=0.5)
    ra.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ra.add_argument("--top-pairs", type=int, default=30)
    ra.add_argument("--sim-peak", type=float, default=0.45)
    ra.add_argument("--no-llm-labels", action="store_true")
    ra.add_argument("--llm-label-model", default="openai/gpt-4.1-mini")
    ra.add_argument("--seed", type=int, default=42)
    ra.add_argument("--n-pairs", type=int, default=10)
    ra.add_argument("--no-novelty", action="store_true")
    ra.add_argument("--k-evidence", type=int, default=4)
    ra.add_argument("--judge-model", default="anthropic/claude-sonnet-4")
    ra.add_argument("--chunk-size", type=int, default=100)
    ra.add_argument("--workers", type=int, default=8)
    ra.add_argument("--max-pages", type=int, default=None)
    ra.set_defaults(func=cmd_run_all)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
