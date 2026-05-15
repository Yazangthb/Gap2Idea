"""PDF -> text extraction via PyMuPDF.

Two output modes:
  - `extract_pdf_text(pdf)`   plain text (the original behaviour)
  - `extract_pdf_blocks(pdf)` style-aware blocks  {role: heading|body, text, ...}

Style-aware mode walks every text span on every page and tags spans whose
font size exceeds the body-text median as candidate headings. This recovers
section titles like "Open Problems" / "Future Work" that the regex section
parser misses when the heading has no numbering / no ALL-CAPS prefix.

`extract_all` emits both: a `text` field (compatibility, used by everything
that already exists) and a `blocks` field (consumed by `sections.py` when
present).
"""
from __future__ import annotations

import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fitz  # pymupdf
import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)

MIN_TEXT_CHARS = 500

# Style-aware heading thresholds — empirically robust on arXiv 2-column layouts.
HEADING_SIZE_RATIO = 1.12   # span size >= body_size * 1.12 → candidate heading
HEADING_MAX_WORDS = 14       # headings are short; "Future Work and Limitations" fits
BOLD_FLAG = 16               # bit 4 of fitz span flags = bold/serif-bold


# ----------------------------------------------------------------------
# Plain-text mode (compat with the existing pipeline)
# ----------------------------------------------------------------------

def extract_pdf_text(pdf_path: Path, max_pages: int | None = None) -> str:
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


# ----------------------------------------------------------------------
# Style-aware mode
# ----------------------------------------------------------------------

def _iter_spans(doc: fitz.Document, max_pages: int | None) -> list[dict]:
    """Flatten every span across all (or first `max_pages`) pages.

    Each span carries: text, size, bold, page, bbox (sorted top-down, left-right).
    """
    spans: list[dict] = []
    n_pages = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    for i in range(n_pages):
        try:
            page = doc.load_page(i)
            page_dict = page.get_text("dict")
        except Exception as e:  # noqa: BLE001
            log.warning("Page %d failed: %s", i, e)
            continue
        for block in page_dict.get("blocks", []):
            if block.get("type", 0) != 0:  # 0 = text block, 1 = image
                continue
            for line in block.get("lines", []):
                for sp in line.get("spans", []):
                    text = (sp.get("text") or "").strip()
                    if not text:
                        continue
                    spans.append({
                        "text": text,
                        "size": float(sp.get("size", 0.0)),
                        "bold": bool(sp.get("flags", 0) & BOLD_FLAG),
                        "page": i,
                        "bbox": sp.get("bbox", (0, 0, 0, 0)),
                    })
    # Top-down, left-right ordering (handles two-column papers reasonably)
    spans.sort(key=lambda s: (s["page"], round(s["bbox"][1] / 5) * 5, s["bbox"][0]))
    return spans


def _body_size(spans: list[dict]) -> float:
    """Modal size weighted by character count — robust to outliers."""
    if not spans:
        return 0.0
    counts: dict[float, int] = {}
    for s in spans:
        # round to nearest 0.5pt to coalesce near-identical sizes
        bucket = round(s["size"] * 2) / 2
        counts[bucket] = counts.get(bucket, 0) + len(s["text"])
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _merge_lines(spans: list[dict], body_size: float) -> list[dict]:
    """Group consecutive same-line spans into one. Tag headings.

    Two spans are on the *same line* iff they share a page, their y values
    differ by <= 2pt, AND the new span's x-start is within a reading-flow
    distance of the previous span's x-end (≤ 80pt). The last condition
    keeps us from merging "VI. CONCLUSION" in the left column with body
    text at the same y in the right column of a 2-column layout.
    """
    if not spans:
        return []
    lines: list[dict] = []
    cur_text: list[str] = []
    cur_size = spans[0]["size"]
    cur_bold = spans[0]["bold"]
    cur_y = round(spans[0]["bbox"][1])
    cur_page = spans[0]["page"]
    cur_x_end = spans[0]["bbox"][2]

    def _flush():
        if not cur_text:
            return
        text = " ".join(cur_text).strip()
        words = text.split()
        n = len(words)
        # Strip leading section numbers/roman numerals before checking case
        bare = text
        m = re.match(r"^(?:\d+(?:\.\d+)*[.\s]\s*|[IVXLC]+\.\s*)", bare)
        if m:
            bare = bare[m.end():]
        letters = [c for c in bare if c.isalpha()]
        caps_frac = (sum(c.isupper() for c in letters) / max(1, len(letters)))
        big = cur_size >= body_size * HEADING_SIZE_RATIO
        # IEEE-style: bold + short + mostly upper-case, size irrelevant
        bold_caps = cur_bold and n <= HEADING_MAX_WORDS and caps_frac >= 0.7
        # ACL/NeurIPS-style: bold + short + same-or-bigger size as body
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
        same_line = (
            s["page"] == cur_page
            and abs(y - cur_y) <= 2
            and (x_start - cur_x_end) <= 80   # next span flows from the last one
            and (x_start - cur_x_end) >= -5   # no big backwards jump (column wrap)
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
            cur_x_end = s["bbox"][2]
    _flush()
    return lines


def extract_pdf_blocks(pdf_path: Path, max_pages: int | None = None) -> list[dict]:
    """Return style-aware blocks (one per line). Empty list on failure."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        log.warning("Failed to open %s: %s", pdf_path.name, e)
        return []
    try:
        spans = _iter_spans(doc, max_pages)
    finally:
        doc.close()
    body_size = _body_size(spans)
    if body_size <= 0:
        return []
    blocks = _merge_lines(spans, body_size)
    return blocks


def blocks_to_text(blocks: list[dict]) -> str:
    """Reconstruct flat text from blocks (preserves heading lines as their own lines)."""
    return "\n".join(b["text"] for b in blocks)


# ----------------------------------------------------------------------
# Batch driver
# ----------------------------------------------------------------------

def extract_all(
    pdfs_dir: Path,
    out_jsonl: Path,
    max_workers: int = 8,
    max_pages: int | None = None,
    include_blocks: bool = True,
) -> pd.DataFrame:
    """Run extraction over every .pdf in `pdfs_dir`.

    When `include_blocks=True` each row also gets a `blocks` field with the
    style-aware structure. The text field is preserved for backwards compat.
    """
    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    log.info("Extracting %d PDFs from %s (blocks=%s)", len(pdfs), pdfs_dir, include_blocks)
    rows: list[dict] = []

    def _work(p: Path) -> dict:
        if include_blocks:
            blocks = extract_pdf_blocks(p, max_pages=max_pages)
            text = blocks_to_text(blocks) if blocks else extract_pdf_text(p, max_pages=max_pages)
            return {
                "id": p.stem, "text": text, "n_chars": len(text),
                "blocks": blocks,
                "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
            }
        text = extract_pdf_text(p, max_pages=max_pages)
        return {"id": p.stem, "text": text, "n_chars": len(text)}

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
