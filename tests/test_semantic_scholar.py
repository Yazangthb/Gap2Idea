"""Unit tests for non-network helpers in semantic_scholar.py."""
from __future__ import annotations

from gap2idea.pipeline.semantic_scholar import flatten_paper


def test_flatten_paper_handles_full_record():
    record = {
        "paperId": "abc123",
        "title": "Some Title",
        "year": 2024,
        "venue": "NeurIPS",
        "abstract": "...",
        "citationCount": 42,
        "openAccessPdf": {"url": "https://example.com/x.pdf"},
        "externalIds": {"ArXiv": "1234.56789"},
        "url": "https://semanticscholar.org/p/abc123",
    }
    row = flatten_paper(record, fallback_id="zzz")
    # `id` is the LOOKUP key (so it joins cleanly with our gaps tables).
    assert row["id"] == "zzz"
    # S2's reported arXiv id is kept separately for audit.
    assert row["s2_arxiv_id"] == "1234.56789"
    assert row["paperId"] == "abc123"
    assert row["title"] == "Some Title"
    assert row["year"] == 2024
    assert row["citation_count"] == 42
    assert row["openaccess_pdf"] == "https://example.com/x.pdf"


def test_flatten_paper_id_preserved_even_when_s2_diverges():
    """Regression: S2 sometimes returns paper B for a query asking for A.
    The TSV `id` must remain the query, not the response."""
    record = {"paperId": "p", "externalIds": {"ArXiv": "2007.00118"}, "title": "T"}
    row = flatten_paper(record, fallback_id="2007.00128")
    assert row["id"] == "2007.00128"
    assert row["s2_arxiv_id"] == "2007.00118"


def test_flatten_paper_uses_fallback_when_no_external():
    row = flatten_paper({"paperId": "x", "title": "T"}, fallback_id="2401.00001")
    assert row["id"] == "2401.00001"


def test_flatten_paper_handles_none():
    row = flatten_paper(None, fallback_id="2401.00002")
    assert row["id"] == "2401.00002"
    assert row["title"] == ""
    assert row["citation_count"] == ""
    assert row["s2_arxiv_id"] == ""
