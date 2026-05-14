"""Select a paper corpus from arXiv and download the PDFs.

Two sources are supported:

1. **Semantic Scholar search** — query-driven. Best for topic-focused theses.
   Returns S2 paper records; we keep only those with an `ArXiv` external ID.

2. **arXiv metadata snapshot** — bulk file (arxiv-metadata-oai-snapshot.json).
   Best when you want category-wide random sampling. Filterable by category
   and minimum year.

After selection, PDFs are downloaded in parallel from arxiv.org/pdf/{id}.pdf.
"""
from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from gap2idea.pipeline.semantic_scholar import S2Client
from gap2idea.utils import get_logger

log = get_logger(__name__)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
MIN_PDF_BYTES = 10_000


# ---------- selection: Semantic Scholar ----------

def select_from_semantic_scholar(
    query: str,
    n: int = 50,
    fields: str | None = None,
) -> pd.DataFrame:
    """Search S2 for `query`, keep papers with an arXiv ID, return a DataFrame.

    Columns: id (arxiv_id), title, year, venue, citation_count, abstract.
    """
    client = S2Client()
    # S2 search caps `limit` at 100 per call; we may need to paginate later.
    limit = min(n * 3, 100)  # over-fetch since not every hit has an arxiv id
    log.info("S2 search '%s' (over-fetch %d to keep arXiv-only)", query, limit)
    hits = client.search(query, limit=limit, fields=fields) if fields else client.search(query, limit=limit)

    rows = []
    for h in hits:
        ext = h.get("externalIds") or {}
        arx = ext.get("ArXiv")
        if not arx:
            continue
        rows.append(
            {
                "id": arx,
                "title": h.get("title") or "",
                "year": h.get("year") or "",
                "venue": h.get("venue") or "",
                "citation_count": h.get("citationCount") or 0,
                "abstract": h.get("abstract") or "",
            }
        )
        if len(rows) >= n:
            break
    df = pd.DataFrame(rows).drop_duplicates(subset="id").reset_index(drop=True)
    log.info("Selected %d papers from S2", len(df))
    return df


# ---------- selection: local arXiv snapshot ----------

def select_from_snapshot(
    snapshot_path: Path,
    target_cats: set[str] = frozenset({"cs.LG", "stat.ML"}),
    min_year: int = 2021,
    n: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Filter a local arxiv-metadata-oai-snapshot.json by category + year, then
    random-sample `n` papers. Returns columns id, title, year, categories.
    """
    log.info("Streaming snapshot %s (cats=%s, min_year=%d)", snapshot_path, target_cats, min_year)
    keepers: list[dict] = []
    with open(snapshot_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cats = set((rec.get("categories") or "").split())
            if not (cats & target_cats):
                continue
            versions = rec.get("versions") or []
            year = None
            if versions:
                created = versions[-1].get("created", "")
                # e.g. "Tue, 12 Mar 2024 18:00:00 GMT"
                for tok in created.split():
                    if tok.isdigit() and len(tok) == 4:
                        year = int(tok)
                        break
            if year is None or year < min_year:
                continue
            keepers.append(
                {
                    "id": rec.get("id"),
                    "title": (rec.get("title") or "").strip(),
                    "year": year,
                    "categories": rec.get("categories"),
                    "abstract": (rec.get("abstract") or "").strip(),
                }
            )
    log.info("Snapshot kept %d papers post-filter", len(keepers))
    rng = random.Random(seed)
    if len(keepers) > n:
        keepers = rng.sample(keepers, n)
    return pd.DataFrame(keepers).reset_index(drop=True)


# ---------- download ----------

def _download_one(arxiv_id: str, out_dir: Path, session: requests.Session) -> tuple[str, bool, str]:
    out = out_dir / f"{arxiv_id}.pdf"
    if out.exists() and out.stat().st_size > MIN_PDF_BYTES:
        return arxiv_id, True, "cached"
    try:
        r = session.get(ARXIV_PDF_URL.format(arxiv_id=arxiv_id), timeout=60)
        if r.status_code != 200:
            return arxiv_id, False, f"http {r.status_code}"
        if len(r.content) < MIN_PDF_BYTES:
            return arxiv_id, False, "too small"
        out.write_bytes(r.content)
        return arxiv_id, True, "downloaded"
    except Exception as e:
        return arxiv_id, False, str(e)


def download_pdfs(
    arxiv_ids: list[str],
    pdfs_dir: Path,
    max_workers: int = 8,
    sleep_between_batches: float = 0.0,
) -> pd.DataFrame:
    pdfs_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "gap2idea-research/0.1"})

    log.info("Downloading %d PDFs to %s", len(arxiv_ids), pdfs_dir)
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_download_one, aid, pdfs_dir, session) for aid in arxiv_ids]
        for i, fut in enumerate(as_completed(futs), 1):
            aid, ok, status = fut.result()
            rows.append({"id": aid, "ok": ok, "status": status})
            if i % 25 == 0:
                log.info("  %d/%d done", i, len(arxiv_ids))
            if sleep_between_batches and i % 10 == 0:
                time.sleep(sleep_between_batches)
    df = pd.DataFrame(rows)
    log.info("PDF downloads: %d ok / %d", int(df["ok"].sum()), len(df))
    return df
