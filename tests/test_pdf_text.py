"""Smoke test for PDF text extraction on a real local PDF.

Skipped if `data/pdfs/` has no .pdf files (so tests still pass on a fresh
checkout with no corpus).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gap2idea.pipeline.pdf_text import extract_pdf_text


def _first_pdf() -> Path | None:
    root = Path(__file__).parent.parent / "data" / "pdfs"
    pdfs = sorted(root.glob("*.pdf"))
    return pdfs[0] if pdfs else None


@pytest.mark.skipif(_first_pdf() is None, reason="no PDFs in data/pdfs/")
def test_extract_first_local_pdf_yields_text():
    pdf = _first_pdf()
    text = extract_pdf_text(pdf, max_pages=3)
    assert isinstance(text, str)
    # 3 pages of an arXiv paper is always > 500 chars
    assert len(text) > 500


def test_extract_missing_file_returns_empty():
    out = extract_pdf_text(Path("does-not-exist.pdf"))
    assert out == ""
