"""Unit tests for openai_methods helpers (no API)."""
from __future__ import annotations

from gap2idea.pipeline.openai_methods import _flatten, _paper_text_snippet


def test_paper_text_snippet_truncates():
    txt = "abc " * 2000  # ~8000 chars
    out = _paper_text_snippet(txt, max_chars=2500)
    assert len(out) <= 2500


def test_paper_text_snippet_handles_empty():
    assert _paper_text_snippet("") == ""
    assert _paper_text_snippet(None) == ""


def test_flatten_strips_newlines_and_keeps_required_fields():
    rec = {
        "paper_id": "p1",
        "items": [
            {"method_type": "algorithm",
             "method_sentence": "We propose\na new\nGNN.",
             "paragraph_text": "Para\ntext.",
             "confidence": 0.92},
            {"method_type": "framework",
             "method_sentence": "We release a benchmark.",
             "paragraph_text": "Para 2",
             "confidence": 0.6},
        ],
    }
    rows = _flatten(rec, "p1")
    assert len(rows) == 2
    assert rows[0]["id"] == "p1"
    assert "\n" not in rows[0]["method_sentence"]
    assert rows[0]["confidence"] == 0.92
    assert {"id", "method_type", "method_sentence", "paragraph_text", "confidence"} <= set(rows[0])


def test_flatten_empty_items():
    assert _flatten({"paper_id": "p1", "items": []}, "p1") == []
