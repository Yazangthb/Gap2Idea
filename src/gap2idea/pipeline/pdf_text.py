"""PDF -> plain text extraction via PyMuPDF.

Used as the first stage after PDFs have been downloaded. Reads every page
(unlike the notebook's tail-only 6-page heuristic) so the section splitter
can find Limitations / Future Work no matter where they live.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # pymupdf
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)

MIN_TEXT_CHARS = 500


def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> str:
    """Extract concatenated text from a single PDF. Returns '' on failure."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.warning("Failed to open %s: %s", pdf_path.name, e)
        return ""
    parts: list[str] = []
    try:
        n_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
        for i in range(n_pages):
            try:
                page = doc.load_page(i)
                parts.append(page.get_text("text"))
            except Exception as e:  # noqa: BLE001
                log.warning("Page %d of %s failed: %s", i, pdf_path.name, e)
    finally:
        doc.close()
    return "\n".join(parts)


def extract_all(
    pdfs_dir: Path,
    out_jsonl: Path,
    max_workers: int = 8,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """Run extraction over every .pdf in `pdfs_dir`. Writes JSONL (one row
    per paper) and returns a DataFrame with columns id, text, n_chars."""
    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    log.info("Extracting text from %d PDFs in %s", len(pdfs), pdfs_dir)
    rows: list[dict] = []

    def _work(p: Path) -> dict:
        text = extract_pdf_text(p, max_pages=max_pages)
        return {"id": p.stem, "text": text, "n_chars": len(text)}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, p) for p in pdfs]
        for fut in as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)  # arxiv IDs are strings even if numeric-looking
    df = df.sort_values("id").reset_index(drop=True)
    before = len(df)
    df = df[df["n_chars"] >= MIN_TEXT_CHARS].reset_index(drop=True)
    log.info("Kept %d/%d papers (>= %d chars)", len(df), before, MIN_TEXT_CHARS)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    log.info("Wrote %s", out_jsonl)
    return df
