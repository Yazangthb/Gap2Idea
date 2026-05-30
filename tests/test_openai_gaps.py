"""Unit tests for openai_gaps: section_type propagation and flattening."""
from __future__ import annotations

import pandas as pd

from gap2idea.pipeline.openai_gaps import _flatten, _section_text_for_paper


# ---------- _section_text_for_paper ----------

def _sections_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_section_text_returns_dominant_section_type():
    """The dominant section_type is the highest-priority one present
    (limitations > future_work > discussion > fallback > tail)."""
    sections = _sections_df([
        {"id": "p1", "section_type": "discussion",  "section_text": "d " * 100},
        {"id": "p1", "section_type": "future_work", "section_text": "fw " * 100},
        {"id": "p1", "section_type": "limitations", "section_text": "lim " * 100},
    ])
    text, section_type = _section_text_for_paper(sections, "p1")
    assert section_type == "limitations"
    assert text.startswith("lim ")  # priority-sorted; limitations first


def test_section_text_picks_future_work_when_no_limitations():
    sections = _sections_df([
        {"id": "p2", "section_type": "discussion",  "section_text": "d " * 100},
        {"id": "p2", "section_type": "future_work", "section_text": "fw " * 100},
    ])
    text, section_type = _section_text_for_paper(sections, "p2")
    assert section_type == "future_work"
    assert text.startswith("fw ")


def test_section_text_returns_empty_when_no_match():
    sections = _sections_df([
        {"id": "p1", "section_type": "limitations", "section_text": "lim " * 100},
    ])
    text, section_type = _section_text_for_paper(sections, "missing")
    assert text == ""
    assert section_type == ""


def test_section_text_handles_unknown_section_type():
    """Unknown section_type (priority fallback) still resolves to *some* type."""
    sections = _sections_df([
        {"id": "p3", "section_type": "tail",     "section_text": "t " * 100},
        {"id": "p3", "section_type": "fallback", "section_text": "fb " * 100},
    ])
    text, section_type = _section_text_for_paper(sections, "p3")
    # fallback (priority 3) ranks above tail (priority 4)
    assert section_type == "fallback"


def test_section_text_handles_arxiv_float_ids():
    """arxiv IDs like '2106.05969' get auto-parsed as floats by pd.read_json.
    The function must compare as strings."""
    sections = _sections_df([
        {"id": 2106.05969, "section_type": "limitations", "section_text": "lim " * 100},
    ])
    text, section_type = _section_text_for_paper(sections, "2106.05969")
    assert section_type == "limitations"
    assert text.startswith("lim ")


# ---------- _flatten ----------

def test_flatten_propagates_section_type():
    rec = {
        "paper_id": "p1",
        "items": [
            {"type": "future_work", "gap_sentence": "We will explore X.",
             "paragraph_text": "Future work: explore X next.", "confidence": 0.9},
            {"type": "limitation", "gap_sentence": "Our method assumes Y.",
             "paragraph_text": "Limitations: assumes Y.", "confidence": 0.8},
        ],
    }
    rows = _flatten(rec, "p1", section_type="future_work")
    assert len(rows) == 2
    for r in rows:
        assert r["section_type"] == "future_work"
        assert r["id"] == "p1"


def test_flatten_default_section_type_is_empty():
    """Backward compat: callers that don't pass section_type get empty string,
    not KeyError."""
    rec = {
        "paper_id": "p1",
        "items": [
            {"type": "limitation", "gap_sentence": "Method assumes X.",
             "paragraph_text": "ctx", "confidence": 0.7},
        ],
    }
    rows = _flatten(rec, "p1")
    assert rows[0]["section_type"] == ""


def test_flatten_empty_items_returns_empty_list():
    assert _flatten({"paper_id": "p1", "items": []}, "p1", section_type="limitations") == []
