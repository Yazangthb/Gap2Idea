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
                    --mode {bridge,within,method-gap,orchestrated}
                    --use-critic            (single critique-revise loop)
  evaluate-ideas    ideas_full.jsonl -> artifacts/idea_eval.tsv + report.md
                    --judges <comma-list>   (multi-judge panel)
  export-ideas      ideas.tsv -> artifacts/exports/*.tex or library.pdf
  serve-mcp         expose corpus + ops as an MCP server (stdio)
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
    import asyncio

    paths = get_paths(args.root)
    gaps = read_tsv(paths.artifacts / "gaps_with_clusters.tsv")
    labels = read_tsv(paths.artifacts / "cluster_labels.tsv")
    out_tsv = paths.artifacts / "ideas.tsv"
    out_jsonl = paths.artifacts / "ideas_full.jsonl"

    # Orchestrated mode: full multi-agent (critic + revise + judge panel)
    if args.mode == "orchestrated":
        from gap2idea.pipeline.orchestrator import orchestrate_batch

        gap_emb_path = paths.artifacts / "gap_embeddings.npy"
        gap_embeddings = np.load(gap_emb_path) if gap_emb_path.exists() else None
        methods_tsv = paths.data / "methods.tsv"
        methods = read_tsv(methods_tsv) if methods_tsv.exists() else None

        method_embeddings = None
        if methods is not None and not methods.empty:
            meth_emb_path = paths.artifacts / "method_embeddings.npy"
            if meth_emb_path.exists() and len(np.load(meth_emb_path)) == len(methods):
                method_embeddings = np.load(meth_emb_path)
            else:
                from gap2idea.pipeline.theme_mining import embed_sentences
                method_embeddings = embed_sentences(methods["method_sentence"].tolist())
                np.save(meth_emb_path, method_embeddings)

        pairs = read_tsv(paths.artifacts / "cluster_pairs.tsv") if args.orchestrate_mode == "bridge" else None

        asyncio.run(orchestrate_batch(
            gaps=gaps, cluster_labels=labels,
            out_tsv=out_tsv, out_jsonl=out_jsonl,
            mode=args.orchestrate_mode,
            pairs=pairs,
            gap_embeddings=gap_embeddings,
            methods=methods, method_embeddings=method_embeddings,
            n=args.n_pairs, model=args.model,
            critic_model=args.critic_model,
            judge_panel=args.judges.split(",") if args.judges else None,
            max_critic_iterations=args.max_critic_iter,
            k_evidence=args.k_evidence, k_methods=args.k_methods,
            sim_low=args.sim_low, sim_high=args.sim_high,
            check_novelty=not args.no_novelty,
            enable_sanity=getattr(args, "sanity", True),
            sanity_budget=getattr(args, "sanity_budget", "benchmark"),
        ))
        return

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
    from gap2idea.pipeline.evaluation import (
        evaluate_ideas, evaluate_ideas_panel, write_report,
    )

    paths = get_paths(args.root)

    if args.judges:
        judge_models = [m.strip() for m in args.judges.split(",") if m.strip()]
        eval_df = evaluate_ideas_panel(
            ideas_jsonl=paths.artifacts / "ideas_full.jsonl",
            out_tsv=paths.artifacts / "idea_eval.tsv",
            judge_models=judge_models,
        )
    else:
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


# ---------- export-ideas ----------

def cmd_export_ideas(args):
    import json as _json
    from gap2idea.pipeline.export import (
        BUNDLED_TEMPLATES,
        LatexCompilationError,
        LatexCompilerNotFound,
        compile_latex_to_pdf,
        idea_to_latex,
        write_idea_latex,
        write_ideas_pdf,
    )

    paths = get_paths(args.root)
    ideas = read_tsv(paths.artifacts / "ideas.tsv")
    if ideas.empty:
        raise SystemExit("No ideas in artifacts/ideas.tsv. Generate some first.")

    out_dir = paths.artifacts / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_cache_dir = paths.artifacts / "paper_drafts"
    if args.full_paper:
        paper_cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve template choice
    template_source: str | None = None
    if args.template_file:
        path = Path(args.template_file)
        if not path.exists():
            raise SystemExit(f"--template-file not found: {path}")
        template_source = path.read_text(encoding="utf-8")
        log.info("Using custom template from %s", path)
    elif args.template not in BUNDLED_TEMPLATES:
        raise SystemExit(
            f"--template must be one of {list(BUNDLED_TEMPLATES)} (or use --template-file)."
        )

    def _stem(i: int, row) -> str:
        safe = "".join(c if c.isalnum() else "_" for c in str(row.get("title", "")))[:40]
        return f"{i:03d}_{safe}" if safe else f"idea_{i:03d}"

    def _full_paper_for(stem: str, idea: dict) -> dict | None:
        """Return a full_paper dict if --full-paper is on. On-disk cache at
        artifacts/paper_drafts/<stem>.json so we never re-call the LLM."""
        if not args.full_paper:
            return None
        cache = paper_cache_dir / f"{stem}.json"
        if cache.exists() and not args.refresh_drafts:
            return _json.loads(cache.read_text(encoding="utf-8"))
        from gap2idea.pipeline.paper_drafter import draft_full_paper
        prior_art = None
        if args.with_prior_art:
            from gap2idea.pipeline.semantic_scholar import S2Client
            try:
                q = f"{idea.get('title','')}. {idea.get('research_question','')}"
                prior_art = S2Client().search(q, limit=8)
            except Exception as e:
                log.warning("S2 prior-art lookup failed for %s: %s", stem, e)
                prior_art = []
        data = draft_full_paper(idea, prior_art=prior_art, model=args.drafter_model)
        cache.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("Cached paper draft -> %s", cache)
        return data

    if args.format == "library-pdf":
        # Single consolidated reportlab-based PDF summary (no LaTeX needed).
        out = out_dir / "idea_library.pdf"
        write_ideas_pdf(ideas, out)
        log.info("Wrote %s", out)
        return

    if args.format == "latex":
        for i, row in ideas.iterrows():
            stem = _stem(i, row)
            idea_dict = row.to_dict()
            full_paper = _full_paper_for(stem, idea_dict)
            out_path = out_dir / f"{stem}.tex"
            write_idea_latex(
                idea_dict, out_path,
                template=args.template, template_source=template_source,
                full_paper=full_paper,
            )
        log.info(
            "Wrote %d .tex files to %s  (template=%s%s)",
            len(ideas), out_dir,
            "custom" if template_source else args.template,
            ", full-paper drafts" if args.full_paper else "",
        )
        return

    if args.format == "rendered-pdf":
        # Compile every idea's LaTeX to PDF (tectonic / pdflatex required).
        ok = fail = 0
        for i, row in ideas.iterrows():
            stem = _stem(i, row)
            idea_dict = row.to_dict()
            full_paper = _full_paper_for(stem, idea_dict)
            tex = idea_to_latex(
                idea_dict,
                template=args.template, template_source=template_source,
                full_paper=full_paper,
            )
            (out_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
            try:
                pdf = compile_latex_to_pdf(tex)
                (out_dir / f"{stem}.pdf").write_bytes(pdf)
                ok += 1
            except LatexCompilerNotFound as e:
                raise SystemExit(str(e))
            except LatexCompilationError as e:
                log.error("[%d] compile failed for %s: %s", i, stem, str(e)[:200])
                fail += 1
        log.info("Rendered %d PDFs (failed %d) into %s", ok, fail, out_dir)
        return

    raise SystemExit(f"unknown --format: {args.format}")


# ---------- serve-mcp ----------

def cmd_serve_mcp(args):
    from gap2idea.mcp_server import run_stdio
    run_stdio(args.root)


# ---------- bench-clustering (from origin/experments/clustering_quality) ----------

def cmd_bench_clustering(args):
    from gap2idea.pipeline.clustering_bench import run_benchmark

    paths = get_paths(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (paths.data / "clustering_bench")
    clusterers = [c.strip() for c in args.clusterers.split(",") if c.strip()]
    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]
    run_benchmark(
        gaps_tsv=Path(args.gaps_tsv),
        out_dir=out_dir,
        clusterers=clusterers,
        embedders=embedders,
        n_bootstrap=args.n_bootstrap,
    )


def cmd_bench_clustering_plots(args):
    from gap2idea.pipeline.clustering_bench_plots import make_all_plots

    paths = get_paths(args.root)
    bench_dir = Path(args.bench_dir) if args.bench_dir else (paths.data / "clustering_bench")
    make_all_plots(bench_dir)


# ---------- bench-extraction (from origin/experments/extraction_quality) ----------

def cmd_bench_extraction(args):
    from gap2idea.pipeline.extraction_bench import run_benchmark

    paths = get_paths(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (paths.data / "bench")
    run_benchmark(
        n=args.n,
        out_dir=out_dir,
        skip_llm=args.skip_llm,
        tarball_path=args.tarball,
        max_scan=args.max_scan,
        use_pdf=args.use_pdf,
        oracle=args.oracle,
    )


def cmd_bench_plots(args):
    from gap2idea.pipeline.extraction_bench_plots import make_all_plots

    paths = get_paths(args.root)
    bench_dir = Path(args.bench_dir) if args.bench_dir else (paths.data / "bench")
    make_all_plots(bench_dir)


def cmd_bench_ablation_plots(args):
    from gap2idea.pipeline.extraction_bench_ablation_plots import make_all_plots

    paths = get_paths(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (paths.data / "bench_ablation_plots")
    make_all_plots(out_dir=out_dir)


# ---------- bench-clustering ----------

def cmd_bench_clustering(args):
    from gap2idea.pipeline.clustering_bench import run_benchmark

    paths = get_paths(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (paths.data / "clustering_bench")
    clusterers = [c.strip() for c in args.clusterers.split(",") if c.strip()]
    embedders = [e.strip() for e in args.embedders.split(",") if e.strip()]
    run_benchmark(
        gaps_tsv=Path(args.gaps_tsv),
        out_dir=out_dir,
        clusterers=clusterers,
        embedders=embedders,
        n_bootstrap=args.n_bootstrap,
    )


def cmd_bench_clustering_plots(args):
    from gap2idea.pipeline.clustering_bench_plots import make_all_plots

    paths = get_paths(args.root)
    bench_dir = Path(args.bench_dir) if args.bench_dir else (paths.data / "clustering_bench")
    gaps_tsv = Path(args.gaps_tsv) if args.gaps_tsv else None
    showcase = None
    if args.showcase:
        c, e = args.showcase.split(":", 1)
        showcase = (c.strip(), e.strip())
    make_all_plots(bench_dir, gaps_tsv=gaps_tsv, showcase=showcase,
                   grid_embedder=args.grid_embedder)


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
        "--mode", choices=["bridge", "within", "method-gap", "orchestrated"], default="bridge",
        help=(
            "bridge: pair gap-clusters in the bridge-score sweet spot (default). "
            "within: synthesise one idea per cluster from its gaps. "
            "method-gap: apply retrieved methods (data/methods.tsv) to each gap-cluster. "
            "orchestrated: full multi-agent pipeline (synthesiser + critic-revise + judge panel)."
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
    # Orchestrated-mode extras
    gi.add_argument("--orchestrate-mode", choices=["bridge", "within", "method-gap"], default="within",
                    help="(orchestrated only) underlying generation mode wrapped by the agentic loop")
    gi.add_argument("--critic-model", default="anthropic/claude-sonnet-4",
                    help="(orchestrated only) model used by the critic agent")
    gi.add_argument("--judges", default="",
                    help="(orchestrated only) comma-separated judge models. Empty = use default panel.")
    gi.add_argument("--max-critic-iter", type=int, default=2,
                    help="(orchestrated only) max critique-revise iterations per idea")
    # Multi-agent experimental sanity stage (PR-3)
    gi.add_argument("--sanity", action=argparse.BooleanOptionalAction, default=True,
                    help="(orchestrated only) run the multi-agent experimental "
                         "sanity stage between critic loop and judge panel. "
                         "Default: --sanity. Use --no-sanity to skip.")
    gi.add_argument("--sanity-budget", choices=["smoke", "benchmark", "full"], default="benchmark",
                    help="(orchestrated + --sanity) max experiment tier the "
                         "Scale-Estimator agent may choose. smoke=tier 1, "
                         "benchmark=tier 2 (default), full=tier 3.")
    gi.set_defaults(func=cmd_generate_ideas)

    # evaluate-ideas
    ev = sub.add_parser("evaluate-ideas", help="LLM-as-judge rubric scoring (single judge or panel)")
    ev.add_argument("--judge-model", default="anthropic/claude-sonnet-4",
                    help="Single judge model. Used when --judges is empty.")
    ev.add_argument("--judges", default="",
                    help="Comma-separated panel of judge models for multi-judge consensus, "
                         "e.g. 'anthropic/claude-sonnet-4,openai/gpt-4o,google/gemini-2.5-flash'. "
                         "When set, --judge-model is ignored and panel aggregation is used.")
    ev.set_defaults(func=cmd_evaluate_ideas)

    # export-ideas
    ex = sub.add_parser(
        "export-ideas",
        help="Render the idea library to LaTeX, server-rendered PDFs, or a consolidated summary PDF.",
    )
    ex.add_argument(
        "--format", choices=["latex", "rendered-pdf", "library-pdf"], default="library-pdf",
        help=(
            "latex:         one .tex per idea using the chosen template. "
            "rendered-pdf:  compile each .tex to PDF via tectonic/pdflatex. "
            "library-pdf:   single consolidated reportlab-based summary (no LaTeX needed)."
        ),
    )
    ex.add_argument(
        "--template", default="standard",
        help="Bundled template name: minimal | standard | ieee.",
    )
    ex.add_argument(
        "--template-file", default="",
        help="Path to a custom .tex.j2 template (Jinja2). Overrides --template.",
    )
    ex.add_argument(
        "--full-paper", action="store_true",
        help="Before rendering, run the paper-drafter LLM on each idea to expand "
             "it into a full paper plan (abstract, related work, method "
             "subsections, experimental plan with TODO markers, discussion, "
             "conclusion). Result is cached at artifacts/paper_drafts/<stem>.json.",
    )
    ex.add_argument(
        "--refresh-drafts", action="store_true",
        help="With --full-paper, ignore the on-disk cache and re-call the LLM.",
    )
    ex.add_argument(
        "--with-prior-art", action="store_true",
        help="With --full-paper, run a Semantic Scholar search per idea so the "
             "drafter can populate the Related Work section with real papers.",
    )
    ex.add_argument(
        "--drafter-model", default="openai/gpt-4.1-mini",
        help="OpenRouter model used by the paper drafter when --full-paper is set.",
    )
    ex.set_defaults(func=cmd_export_ideas)

    # serve-mcp
    mc = sub.add_parser("serve-mcp", help="Run the Model Context Protocol server (stdio transport) "
                                          "so Claude Desktop / Cursor / etc. can query the corpus")
    mc.set_defaults(func=cmd_serve_mcp)

    # bench-clustering (from origin/experments/clustering_quality)
    bc = sub.add_parser("bench-clustering",
                        help="Benchmark clustering quality (clusterer x embedder grid)")
    bc.add_argument("--gaps-tsv", default="data/bench/gaps.tsv",
                    help="Path to a gaps.tsv produced by extract-gaps")
    bc.add_argument("--out-dir", default=None,
                    help="Default: <root>/data/clustering_bench")
    bc.add_argument("--clusterers", default="kmeans,agglomerative,hdbscan,bertopic")
    bc.add_argument("--embedders",
                    default="all-MiniLM-L6-v2,all-mpnet-base-v2,"
                            "intfloat/e5-base-v2,BAAI/bge-small-en-v1.5")
    bc.add_argument("--n-bootstrap", type=int, default=10)
    bc.set_defaults(func=cmd_bench_clustering)

    bcp = sub.add_parser("bench-clustering-plots",
                         help="Regenerate plots from clustering_bench/metrics.tsv")
    bcp.add_argument("--bench-dir", default=None)
    bcp.set_defaults(func=cmd_bench_clustering_plots)

    # bench-extraction (from origin/experments/extraction_quality)
    be = sub.add_parser(
        "bench-extraction",
        help="Benchmark our extraction pipeline against unarXive's labeled sections",
    )
    be.add_argument("--n", type=int, default=10)
    be.add_argument("--out-dir", default=None, help="Default: <root>/data/bench")
    be.add_argument("--tarball", default="data/bench/raw/unarxive_open_subset.tar.xz",
                    help="Path to the unarXive open-subset .tar.xz from Zenodo record 7752615")
    be.add_argument("--max-scan", type=int, default=20000,
                    help="Cap on records scanned before giving up")
    be.add_argument("--skip-llm", action="store_true",
                    help="Evaluate the regex stage only; skip openai_gaps (no API key needed)")
    be.add_argument("--use-pdf", action="store_true",
                    help="Download arxiv PDFs for the sampled papers and feed the "
                         "style-aware block extractor (PyMuPDF). When omitted, the "
                         "bench feeds unarXive's already-parsed plain text.")
    be.add_argument("--oracle", action="store_true",
                    help="Also feed the gold section text directly to openai_gaps "
                         "and report pipeline_vs_oracle gap-to-gap metrics.")
    be.set_defaults(func=cmd_bench_extraction)

    # bench-plots (extraction benchmark plots)
    bp = sub.add_parser("bench-plots",
                        help="Regenerate plots from an existing data/bench/metrics.tsv")
    bp.add_argument("--bench-dir", default=None)
    bp.set_defaults(func=cmd_bench_plots)

    # bench-ablation-plots
    bap = sub.add_parser("bench-ablation-plots",
                         help="Cross-variant ablation plots (v1/v2a/v2b + oracle)")
    bap.add_argument("--out-dir", default=None,
                     help="Default: <root>/data/bench_ablation_plots")
    bap.set_defaults(func=cmd_bench_ablation_plots)

    # bench-clustering
    bc = sub.add_parser("bench-clustering",
                        help="Benchmark clustering quality (clusterer x embedder grid)")
    bc.add_argument("--gaps-tsv", default="data/bench/gaps.tsv")
    bc.add_argument("--out-dir", default=None)
    bc.add_argument("--clusterers",
                    default="kmeans,agglomerative,hdbscan,hdbscan_umap,bertopic")
    bc.add_argument("--embedders",
                    default="all-MiniLM-L6-v2,all-mpnet-base-v2,"
                            "intfloat/e5-base-v2,BAAI/bge-small-en-v1.5")
    bc.add_argument("--n-bootstrap", type=int, default=10)
    bc.set_defaults(func=cmd_bench_clustering)

    bcp = sub.add_parser("bench-clustering-plots",
                         help="Regenerate clustering plots (optionally with 2-D cluster scatters)")
    bcp.add_argument("--bench-dir", default=None)
    bcp.add_argument("--gaps-tsv", default=None,
                     help="If set, also draw 2-D cluster scatters using these gap sentences.")
    bcp.add_argument("--showcase", default=None,
                     help="`<clusterer>:<embedder>` to highlight as the single big plot. "
                          "Default: agglomerative:BAAI/bge-small-en-v1.5")
    bcp.add_argument("--grid-embedder", default=None,
                     help="Embedder to use for the all-clusterers comparison grid. "
                          "Default: same as showcase embedder.")
    bcp.set_defaults(func=cmd_bench_clustering_plots)

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
