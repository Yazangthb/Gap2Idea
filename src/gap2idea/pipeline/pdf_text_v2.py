"""PDF -> text extraction v2: column-aware reading order.

The v1 extractor (pdf_text.py) sorts spans by (page, y, x), which interleaves
two-column papers — left-column body text gets merged with right-column
reference lists at the same y. This produces "author name bleed" into the
extracted text.

v2 fixes this by clustering spans into columns per page via x-coordinate
k-means (k=1 or 2), then sorting within each column by y, then concatenating
columns left-to-right.

Same output schema as v1 (text + blocks), so it drops into the existing
pipeline. v1 is preserved unchanged for paper-archival.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)

MIN_TEXT_CHARS = 500
HEADING_SIZE_RATIO = 1.12
HEADING_MAX_WORDS = 14
BOLD_FLAG = 16


def _detect_columns(spans_on_page: list[dict], page_width: float) -> list[tuple[float, float]]:
    """Detect column x-ranges for one page. Returns list of (x_start, x_end).

    Uses a simple gap-detection heuristic: build a histogram of x-centers,
    look for a wide gap in the middle of the page (>= 8% of page width).
    """
    if not spans_on_page:
        return [(0, page_width)]
    centers = sorted((s["bbox"][0] + s["bbox"][2]) / 2 for s in spans_on_page)
    # Try the obvious binary split: is there a clear gap near the middle?
    mid = page_width / 2
    left_max = max((c for c in centers if c < mid), default=None)
    right_min = min((c for c in centers if c >= mid), default=None)
    if left_max is None or right_min is None:
        return [(0, page_width)]
    gap = right_min - left_max
    if gap < page_width * 0.04:
        return [(0, page_width)]
    # Two columns
    split_x = (left_max + right_min) / 2
    return [(0, split_x), (split_x, page_width)]


def _column_assign(span: dict, columns: list[tuple[float, float]]) -> int:
    """Return the index of the column whose x-range the span's center falls in."""
    cx = (span["bbox"][0] + span["bbox"][2]) / 2
    for i, (x0, x1) in enumerate(columns):
        if x0 <= cx < x1:
            return i
    return len(columns) - 1


def _iter_spans_column_aware(doc: fitz.Document, max_pages: int | None) -> list[dict]:
    """Spans in true reading order: page-by-page, column-by-column, top-to-bottom."""
    spans: list[dict] = []
    n_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    for i in range(n_pages):
        try:
            page = doc.load_page(i)
            page_dict = page.get_text("dict")
            page_width = float(page.rect.width)
        except Exception as e:
            log.warning("Page %d failed: %s", i, e)
            continue
        page_spans: list[dict] = []
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    text = (sp.get("text") or "").strip()
                    if not text:
                        continue
                    page_spans.append({
                        "text": text,
                        "size": float(sp.get("size", 0.0)),
                        "bold": bool(sp.get("flags", 0) & BOLD_FLAG),
                        "page": i,
                        "bbox": sp.get("bbox", (0, 0, 0, 0)),
                    })
        if not page_spans:
            continue
        cols = _detect_columns(page_spans, page_width)
        for s in page_spans:
            s["_col"] = _column_assign(s, cols)
        # Sort: column index, then y, then x
        page_spans.sort(key=lambda s: (s["_col"], round(s["bbox"][1] / 5) * 5, s["bbox"][0]))
        spans.extend(page_spans)
    return spans


def _body_size(spans: list[dict]) -> float:
    if not spans:
        return 0.0
    counts: dict[float, int] = {}
    for s in spans:
        bucket = round(s["size"] * 2) / 2
        counts[bucket] = counts.get(bucket, 0) + len(s["text"])
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _merge_lines(spans: list[dict], body_size: float) -> list[dict]:
    """Group same-line spans WITHIN the same column."""
    if not spans:
        return []
    lines: list[dict] = []
    cur_text: list[str] = []
    cur_size = spans[0]["size"]
    cur_bold = spans[0]["bold"]
    cur_y = round(spans[0]["bbox"][1])
    cur_page = spans[0]["page"]
    cur_col = spans[0].get("_col", 0)
    cur_x_end = spans[0]["bbox"][2]

    def _flush():
        if not cur_text:
            return
        text = " ".join(cur_text).strip()
        words = text.split()
        n = len(words)
        bare = text
        m = re.match(r"^(?:\d+(?:\.\d+)*[.\s]\s*|[IVXLC]+\.\s*)", bare)
        if m:
            bare = bare[m.end():]
        letters = [c for c in bare if c.isalpha()]
        caps_frac = (sum(c.isupper() for c in letters) / max(1, len(letters)))
        big = cur_size >= body_size * HEADING_SIZE_RATIO
        bold_caps = cur_bold and n <= HEADING_MAX_WORDS and caps_frac >= 0.7
        bold_big = cur_bold and n <= HEADING_MAX_WORDS and cur_size >= body_size - 0.5
        is_heading = (big or bold_caps or bold_big) and 1 <= n <= HEADING_MAX_WORDS
        lines.append({
            "role": "heading" if is_heading else "body",
            "text": text,
            "size": cur_size,
            "bold": cur_bold,
            "page": cur_page,
        })

    for s in spans:
        y = round(s["bbox"][1])
        x_start = s["bbox"][0]
        col = s.get("_col", 0)
        same_line = (
            s["page"] == cur_page
            and col == cur_col           # stay within the same column
            and abs(y - cur_y) <= 2
            and (x_start - cur_x_end) <= 80
            and (x_start - cur_x_end) >= -5
        )
        if same_line:
            cur_text.append(s["text"])
            cur_size = max(cur_size, s["size"])
            cur_bold = cur_bold or s["bold"]
            cur_x_end = s["bbox"][2]
        else:
            _flush()
            cur_text = [s["text"]]
            cur_size = s["size"]
            cur_bold = s["bold"]
            cur_y = y
            cur_page = s["page"]
            cur_col = col
            cur_x_end = s["bbox"][2]
    _flush()
    return lines


def extract_pdf_blocks_v2(pdf_path: Path, max_pages: int | None = None) -> list[dict]:
    """v2: column-aware blocks."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.warning("Failed to open %s: %s", pdf_path.name, e)
        return []
    try:
        spans = _iter_spans_column_aware(doc, max_pages)
    finally:
        doc.close()
    body_size = _body_size(spans)
    if body_size <= 0:
        return []
    return _merge_lines(spans, body_size)


def blocks_to_text_v2(blocks: list[dict]) -> str:
    return "\n".join(b["text"] for b in blocks)


def extract_all_v2(
    pdfs_dir: Path,
    out_jsonl: Path,
    max_workers: int = 8,
    max_pages: int | None = None,
) -> pd.DataFrame:
    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    log.info("Extracting %d PDFs (v2 column-aware) from %s", len(pdfs), pdfs_dir)
    rows: list[dict] = []

    def _work(p: Path) -> dict:
        blocks = extract_pdf_blocks_v2(p, max_pages=max_pages)
        text = blocks_to_text_v2(blocks)
        return {
            "id": p.stem, "text": text, "n_chars": len(text),
            "blocks": blocks,
            "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
            "extractor": "pdf_text_v2",
        }

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, p) for p in pdfs]
        for fut in as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    df = df.sort_values("id").reset_index(drop=True)
    before = len(df)
    df = df[df["n_chars"] >= MIN_TEXT_CHARS].reset_index(drop=True)
    log.info("Kept %d/%d papers (>= %d chars)", len(df), before, MIN_TEXT_CHARS)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    log.info("Wrote %s", out_jsonl)
    return df
