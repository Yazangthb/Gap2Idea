"""Benchmark our extraction pipeline against unarXive's author-labeled sections.

Design (Pipeline-vs-gold, N=10 papers):
  1. Stream the unarXive 2023 open-subset tarball (Zenodo record 7752615) and
     sample N papers that have an author-titled "future work" / "limitations"
     section. We stream tar members one-by-one so we never extract the full
     4.8GB to disk.
  2. The text of that section is the GOLD reference. The concatenation of all
     `body_text` entries is the paper's FULL TEXT.
  3. Feed FULL TEXT through our existing pipeline:
        - `sections.py`        -> predicted section span
        - `openai_gaps.py`     -> predicted gap sentences
  4. Metrics:
        - section_rouge_{1,2,L}    predicted_section_text vs gold_section_text
        - gap_recovery_at_tau      fraction of gap sentences with cosine >= tau
                                   to *some* sentence in the GOLD section
        - hallucination_at_tau     fraction of gap sentences with cosine <  tau
                                   to *any* sentence in the FULL PAPER

Only the regex stage is deterministic; the LLM stage requires
OPENROUTER_API_KEY. Pass `--skip-llm` to evaluate the regex stage alone.
"""
from __future__ import annotations

import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)

GOLD_TITLE_RE = re.compile(
    r"\b(future\s+work|future\s+direction|limitation|open\s+problem|open\s+question)s?\b",
    re.IGNORECASE,
)

DEFAULT_UNARXIVE_TARBALL = "data/bench/raw/unarxive_open_subset.tar.xz"
MIN_GOLD_CHARS = 200
MIN_FULL_CHARS = 4000


# ----------------------------------------------------------------------
# 1. Load & sample from unarXive
# ----------------------------------------------------------------------

@dataclass
class BenchPaper:
    paper_id: str
    full_text: str
    gold_section_text: str
    gold_section_titles: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.paper_id,
            "full_text": self.full_text,
            "gold_section_text": self.gold_section_text,
            "gold_section_titles": self.gold_section_titles,
        }


def _extract_gold_and_full(record: dict) -> BenchPaper | None:
    """Pull (full_text, gold_section_text) out of one unarXive record."""
    body = record.get("body_text") or []
    if not body:
        return None

    gold_chunks: list[str] = []
    full_chunks: list[str] = []
    gold_titles: list[str] = []
    seen_titles: set[str] = set()
    for item in body:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        full_chunks.append(text)
        title = (item.get("section") or "").strip()
        if title and GOLD_TITLE_RE.search(title):
            gold_chunks.append(text)
            if title not in seen_titles:
                gold_titles.append(title)
                seen_titles.add(title)

    if not gold_chunks:
        return None

    gold_text = "\n\n".join(gold_chunks).strip()
    full_text = "\n\n".join(full_chunks).strip()
    if len(gold_text) < MIN_GOLD_CHARS or len(full_text) < MIN_FULL_CHARS:
        return None

    return BenchPaper(
        paper_id=str(record.get("paper_id") or record.get("id") or "unknown"),
        full_text=full_text,
        gold_section_text=gold_text,
        gold_section_titles=gold_titles,
    )


def _iter_unarxive_records(tarball_path: Path) -> Iterator[dict]:
    """Yield one record dict at a time from the unarXive tarball.

    The open subset is a `.tar.xz` whose members are JSONL files (typically
    one per arXiv category-month bucket). Each line is one paper JSON.
    We also handle plain `.json` members (one paper per file) defensively.
    """
    with tarfile.open(tarball_path, mode="r:xz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.lower()
            if not (name.endswith(".jsonl") or name.endswith(".json")):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                if name.endswith(".jsonl"):
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
                else:
                    try:
                        yield json.loads(f.read())
                    except json.JSONDecodeError:
                        continue
            finally:
                f.close()


def sample_unarxive_papers(
    n: int = 10,
    tarball_path: Path | str = DEFAULT_UNARXIVE_TARBALL,
    max_scan: int = 20000,
) -> list[BenchPaper]:
    """Read unarXive tarball until we collect `n` papers with FW/Limitations."""
    tarball_path = Path(tarball_path)
    if not tarball_path.exists():
        raise FileNotFoundError(
            f"unarXive tarball not found at {tarball_path}. "
            "Download with: curl -L -o data/bench/raw/unarxive_open_subset.tar.xz "
            "https://zenodo.org/records/7752615/files/unarXive_230324_open_subset.tar.xz"
        )

    log.info("Scanning %s (up to %d records for %d hits)", tarball_path, max_scan, n)
    collected: list[BenchPaper] = []
    scanned = 0
    for record in _iter_unarxive_records(tarball_path):
        scanned += 1
        if scanned > max_scan:
            break
        try:
            bp = _extract_gold_and_full(record)
        except Exception as e:  # noqa: BLE001 - tolerate per-record schema drift
            log.debug("skip record %d (%s)", scanned, e)
            continue
        if bp is None:
            continue
        collected.append(bp)
        log.info("  [%d/%d] paper %s  gold_titles=%s  full=%dch  gold=%dch",
                 len(collected), n, bp.paper_id, bp.gold_section_titles,
                 len(bp.full_text), len(bp.gold_section_text))
        if len(collected) >= n:
            break

    if len(collected) < n:
        log.warning("Only found %d/%d papers with gold sections after scanning %d records",
                    len(collected), n, scanned)
    return collected


def save_bench_inputs(papers: list[BenchPaper], out_dir: Path) -> tuple[Path, Path]:
    """Save the sampled papers as:
       - bench_papers.jsonl       (id, full_text, gold_section_text, gold_section_titles)
       - paper_texts.jsonl        (id, text)  -- the format `extract_all_sections` expects
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "bench_papers.jsonl"
    texts_path = out_dir / "paper_texts.jsonl"
    with bench_path.open("w", encoding="utf-8") as fb, texts_path.open("w", encoding="utf-8") as ft:
        for bp in papers:
            fb.write(json.dumps(bp.to_dict(), ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": bp.paper_id, "text": bp.full_text}, ensure_ascii=False) + "\n")
    log.info("Wrote %s and %s", bench_path, texts_path)
    return bench_path, texts_path


def save_bench_inputs_from_pdfs(
    papers: list[BenchPaper], out_dir: Path, pdf_dir: Path,
) -> tuple[Path, Path]:
    """PDF variant: for each paper download arxiv.org/pdf/<id>.pdf and run
    PyMuPDF's style-aware block extractor. Writes:
       - bench_papers.jsonl              (unchanged — gold still comes from unarXive)
       - paper_texts.jsonl               (id, text, blocks)
    """
    import requests

    from gap2idea.pipeline.pdf_text import blocks_to_text, extract_pdf_blocks

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    bench_path = out_dir / "bench_papers.jsonl"
    texts_path = out_dir / "paper_texts.jsonl"

    with bench_path.open("w", encoding="utf-8") as fb, texts_path.open("w", encoding="utf-8") as ft:
        for bp in papers:
            # Persist the gold (text-side) record
            fb.write(json.dumps(bp.to_dict(), ensure_ascii=False) + "\n")

            pdf_path = pdf_dir / f"{bp.paper_id.replace('/', '_')}.pdf"
            if not pdf_path.exists():
                url = f"https://arxiv.org/pdf/{bp.paper_id}.pdf"
                log.info("  downloading %s", url)
                try:
                    r = requests.get(url, timeout=60)
                    r.raise_for_status()
                    pdf_path.write_bytes(r.content)
                except Exception as e:  # noqa: BLE001
                    log.warning("  download failed for %s: %s — falling back to unarXive text",
                                bp.paper_id, e)
                    ft.write(json.dumps({"id": bp.paper_id, "text": bp.full_text},
                                        ensure_ascii=False) + "\n")
                    continue

            blocks = extract_pdf_blocks(pdf_path)
            if not blocks:
                log.warning("  PyMuPDF returned no blocks for %s — falling back", bp.paper_id)
                ft.write(json.dumps({"id": bp.paper_id, "text": bp.full_text},
                                    ensure_ascii=False) + "\n")
                continue
            text = blocks_to_text(blocks)
            n_head = sum(1 for b in blocks if b.get("role") == "heading")
            log.info("  %s: %d blocks (%d headings, %d chars)",
                     bp.paper_id, len(blocks), n_head, len(text))
            ft.write(json.dumps({"id": bp.paper_id, "text": text, "blocks": blocks},
                                ensure_ascii=False) + "\n")
    log.info("Wrote %s and %s", bench_path, texts_path)
    return bench_path, texts_path


# ----------------------------------------------------------------------
# 2. Run our pipeline
# ----------------------------------------------------------------------

def run_pipeline(out_dir: Path, paper_texts_jsonl: Path, skip_llm: bool) -> tuple[Path, Path | None]:
    """Invoke our existing pipeline stages and return (sections_jsonl, gaps_tsv|None)."""
    from gap2idea.pipeline.sections import extract_all_sections

    sections_path = out_dir / "sections_extracted.jsonl"
    extract_all_sections(paper_texts_jsonl, sections_path)

    gaps_path: Path | None = None
    if not skip_llm:
        from gap2idea.pipeline.openai_gaps import extract_gaps
        gaps_path = out_dir / "gaps.tsv"
        extract_gaps(sections_path, gaps_path, resume=True)
    return sections_path, gaps_path


def generate_oracle_gaps(papers: list[BenchPaper], out_dir: Path) -> Path:
    """Feed the gold section text from unarXive straight into openai_gaps.

    Builds a synthetic sections.jsonl where each row's `section_text` is the
    paper's gold section (skipping Stage 1 entirely), then runs the same
    `extract_gaps` LLM call. The output `oracle_gaps.tsv` has the same schema
    as `gaps.tsv` so downstream code can treat it uniformly.
    """
    from gap2idea.pipeline.openai_gaps import extract_gaps

    oracle_sections_path = out_dir / "oracle_sections.jsonl"
    oracle_gaps_path = out_dir / "oracle_gaps.tsv"
    with oracle_sections_path.open("w", encoding="utf-8") as f:
        for bp in papers:
            f.write(json.dumps({
                "id": bp.paper_id,
                "section_type": "gold",
                "heading": ", ".join(bp.gold_section_titles) or "gold",
                "section_text": bp.gold_section_text,
            }, ensure_ascii=False) + "\n")
    log.info("Generating oracle gaps from gold sections (%d papers)…", len(papers))
    # resume=True so a kill mid-run keeps work and a restart skips done papers.
    extract_gaps(oracle_sections_path, oracle_gaps_path, resume=True)
    return oracle_gaps_path


# ----------------------------------------------------------------------
# 3. Metrics
# ----------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter (NLTK punkt would be heavier than needed)."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])", text)
    return [p.strip() for p in parts if len(p.strip()) >= 10]


def _rouge(pred: str, ref: str) -> dict[str, float]:
    from rouge_score import rouge_scorer  # type: ignore
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    s = scorer.score(ref, pred)
    return {f"rouge_{k}_f": v.fmeasure for k, v in s.items()}


def _embed(model, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)


def _max_cosine(query_vecs: np.ndarray, key_vecs: np.ndarray) -> np.ndarray:
    if query_vecs.shape[0] == 0 or key_vecs.shape[0] == 0:
        return np.zeros((query_vecs.shape[0],), dtype=np.float32)
    sims = query_vecs @ key_vecs.T
    return sims.max(axis=1)


def compute_metrics(
    papers: list[BenchPaper],
    sections_jsonl: Path,
    gaps_tsv: Path | None,
    taus: tuple[float, ...] = (0.5, 0.6, 0.7),
    oracle_gaps_tsv: Path | None = None,
) -> pd.DataFrame:
    """Per-paper metric table (long format).

    When `oracle_gaps_tsv` is provided, also report:
      pipeline_vs_oracle.recovery_at_τ   — fraction of pipeline gaps whose
                                            max cosine to any oracle gap ≥ τ
      pipeline_vs_oracle.coverage_at_τ   — fraction of oracle gaps that some
                                            pipeline gap matches at ≥ τ
    """
    from sentence_transformers import SentenceTransformer  # type: ignore

    sections_df = pd.read_json(sections_jsonl, lines=True, dtype=False)
    sections_df["id"] = sections_df["id"].astype(str)

    gaps_df = None
    if gaps_tsv is not None and gaps_tsv.exists():
        gaps_df = pd.read_csv(gaps_tsv, sep="\t", dtype={"id": str})

    oracle_df = None
    if oracle_gaps_tsv is not None and oracle_gaps_tsv.exists():
        oracle_df = pd.read_csv(oracle_gaps_tsv, sep="\t", dtype={"id": str})

    log.info("Loading sentence-transformer (all-MiniLM-L6-v2)…")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    rows: list[dict] = []
    for bp in papers:
        pid = bp.paper_id
        gold_text = bp.gold_section_text
        gold_sents = _split_sentences(gold_text)
        full_sents = _split_sentences(bp.full_text)
        gold_vecs = _embed(embedder, gold_sents)
        full_vecs = _embed(embedder, full_sents)

        # --- regex stage: predicted section span vs gold ---
        sub = sections_df[sections_df["id"] == pid]
        pred_section_text = "\n\n".join(sub["section_text"].astype(str).tolist()) if not sub.empty else ""
        section_kinds = sorted(sub["section_type"].unique().tolist()) if not sub.empty else []
        r = _rouge(pred_section_text, gold_text) if pred_section_text else {
            "rouge_rouge1_f": 0.0, "rouge_rouge2_f": 0.0, "rouge_rougeL_f": 0.0,
        }
        rows.append({
            "id": pid, "stage": "regex_section", "metric": "rouge1_f",
            "value": r["rouge_rouge1_f"], "extra": ",".join(section_kinds),
        })
        rows.append({
            "id": pid, "stage": "regex_section", "metric": "rouge2_f",
            "value": r["rouge_rouge2_f"], "extra": ",".join(section_kinds),
        })
        rows.append({
            "id": pid, "stage": "regex_section", "metric": "rougeL_f",
            "value": r["rouge_rougeL_f"], "extra": ",".join(section_kinds),
        })
        rows.append({
            "id": pid, "stage": "regex_section", "metric": "pred_chars",
            "value": float(len(pred_section_text)), "extra": "",
        })

        # --- LLM stage: gap sentences vs gold & full ---
        if gaps_df is not None:
            gap_sents = gaps_df.loc[gaps_df["id"] == pid, "gap_sentence"].astype(str).tolist()
            gap_vecs = _embed(embedder, gap_sents)

            sim_to_gold = _max_cosine(gap_vecs, gold_vecs)
            sim_to_full = _max_cosine(gap_vecs, full_vecs)

            rows.append({
                "id": pid, "stage": "llm_gap", "metric": "n_gaps",
                "value": float(len(gap_sents)), "extra": "",
            })
            rows.append({
                "id": pid, "stage": "llm_gap", "metric": "mean_sim_to_gold",
                "value": float(sim_to_gold.mean()) if len(sim_to_gold) else 0.0, "extra": "",
            })
            rows.append({
                "id": pid, "stage": "llm_gap", "metric": "mean_sim_to_full",
                "value": float(sim_to_full.mean()) if len(sim_to_full) else 0.0, "extra": "",
            })
            for tau in taus:
                recov = float((sim_to_gold >= tau).mean()) if len(sim_to_gold) else 0.0
                halluc = float((sim_to_full < tau).mean()) if len(sim_to_full) else 0.0
                rows.append({
                    "id": pid, "stage": "llm_gap", "metric": f"recovery_at_{tau}",
                    "value": recov, "extra": "",
                })
                rows.append({
                    "id": pid, "stage": "llm_gap", "metric": f"hallucination_at_{tau}",
                    "value": halluc, "extra": "",
                })

            # --- pipeline_vs_oracle: gap-to-gap comparison ---
            if oracle_df is not None:
                oracle_sents = oracle_df.loc[oracle_df["id"] == pid, "gap_sentence"].astype(str).tolist()
                oracle_vecs = _embed(embedder, oracle_sents)
                pipe_to_oracle = _max_cosine(gap_vecs, oracle_vecs)         # per pipeline gap
                oracle_to_pipe = _max_cosine(oracle_vecs, gap_vecs)         # per oracle gap

                rows.append({
                    "id": pid, "stage": "pipeline_vs_oracle", "metric": "n_oracle_gaps",
                    "value": float(len(oracle_sents)), "extra": "",
                })
                rows.append({
                    "id": pid, "stage": "pipeline_vs_oracle", "metric": "mean_sim_pipe_to_oracle",
                    "value": float(pipe_to_oracle.mean()) if len(pipe_to_oracle) else 0.0,
                    "extra": "",
                })
                rows.append({
                    "id": pid, "stage": "pipeline_vs_oracle", "metric": "mean_sim_oracle_to_pipe",
                    "value": float(oracle_to_pipe.mean()) if len(oracle_to_pipe) else 0.0,
                    "extra": "",
                })
                for tau in taus:
                    rec_p = float((pipe_to_oracle >= tau).mean()) if len(pipe_to_oracle) else 0.0
                    cov_o = float((oracle_to_pipe >= tau).mean()) if len(oracle_to_pipe) else 0.0
                    rows.append({
                        "id": pid, "stage": "pipeline_vs_oracle",
                        "metric": f"recovery_at_{tau}", "value": rec_p, "extra": "",
                    })
                    rows.append({
                        "id": pid, "stage": "pipeline_vs_oracle",
                        "metric": f"coverage_at_{tau}", "value": cov_o, "extra": "",
                    })

    return pd.DataFrame(rows)


def write_report(metrics: pd.DataFrame, papers: list[BenchPaper], out_md: Path) -> None:
    by = metrics.groupby(["stage", "metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    lines: list[str] = []
    lines.append(f"# Extraction-quality benchmark (N={len(papers)} papers, unarXive gold sections)\n")
    lines.append("Reference: author-titled `future work` / `limitations` sections from unarXive 2023.\n")
    lines.append("\n## Sampled papers\n")
    for bp in papers:
        lines.append(f"- `{bp.paper_id}`  titles: {bp.gold_section_titles}  "
                     f"(gold {len(bp.gold_section_text)} chars, full {len(bp.full_text)} chars)")
    lines.append("\n## Aggregate metrics (mean ± std)\n")
    lines.append("| stage | metric | mean | std | n |")
    lines.append("|---|---|---:|---:|---:|")
    for _, r in by.iterrows():
        lines.append(f"| {r['stage']} | {r['metric']} | {r['mean']:.3f} | {r['std']:.3f} | {int(r['count'])} |")
    lines.append("\n## Interpretation\n")
    lines.append("- `regex_section.rouge*_f`: how much of the gold span our regex section parser recovers (lexical overlap).")
    lines.append("- `llm_gap.recovery_at_τ`: fraction of LLM-extracted gap sentences whose max cosine to a gold-section sentence ≥ τ — high = the LLM points at real future-work content.")
    lines.append("- `llm_gap.hallucination_at_τ`: fraction whose max cosine to *any* sentence in the source paper < τ — high = the LLM is inventing content not in the paper.")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", out_md)


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------

def run_benchmark(
    n: int,
    out_dir: Path,
    skip_llm: bool = False,
    tarball_path: Path | str = DEFAULT_UNARXIVE_TARBALL,
    max_scan: int = 20000,
    use_pdf: bool = False,
    pdf_dir: Path | None = None,
    oracle: bool = False,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. sample
    papers = sample_unarxive_papers(n=n, tarball_path=tarball_path, max_scan=max_scan)
    if not papers:
        raise RuntimeError("No qualifying papers found in unarXive stream")
    if use_pdf:
        pdf_dir = pdf_dir or (out_dir / "pdfs")
        save_bench_inputs_from_pdfs(papers, out_dir, pdf_dir)
    else:
        save_bench_inputs(papers, out_dir)

    # 2. run pipeline
    sections_path, gaps_path = run_pipeline(out_dir, out_dir / "paper_texts.jsonl", skip_llm=skip_llm)

    # 2b. optional oracle: feed gold section straight to the LLM
    oracle_gaps_path: Path | None = None
    if oracle and not skip_llm:
        oracle_gaps_path = generate_oracle_gaps(papers, out_dir)

    # 3. metrics
    metrics = compute_metrics(papers, sections_path, gaps_path,
                              oracle_gaps_tsv=oracle_gaps_path)
    metrics_path = out_dir / "metrics.tsv"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    log.info("Wrote %s", metrics_path)

    # 4. report
    write_report(metrics, papers, out_dir / "REPORT.md")

    # 5. plots
    try:
        from gap2idea.pipeline.extraction_bench_plots import make_all_plots
        make_all_plots(out_dir)
    except Exception as e:  # noqa: BLE001
        log.warning("Plot generation failed: %s", e)

    return metrics_path
