"""Unit tests for section extraction."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gap2idea.pipeline.openai_gaps import _section_text_for_paper
from gap2idea.pipeline.sections import (
    _cut_before_references,
    _structured_sections,
    _tail_fallback,
    _window_fallback,
    extract_all_sections,
    extract_sections_for_paper,
)


def _para(prefix: str = "Body") -> str:
    """Returns a paragraph long enough to clear MIN_BODY_CHARS=200."""
    return (prefix + " sentence here. ") * 30


def test_cut_before_references():
    txt = "Intro body.\n\nReferences\n\n[1] Foo"
    cut = _cut_before_references(txt)
    assert "References" not in cut
    assert "Intro body" in cut


def test_structured_finds_limitations_heading():
    text = (
        "1 Introduction\n" + _para("intro") + "\n"
        "4 Limitations\n" + _para("lim") + "\n"
        "5 Conclusion\n" + _para("conc") + "\n"
    )
    sections = _structured_sections(text)
    types = {s["section_type"] for s in sections}
    assert "limitations" in types
    assert "discussion" in types  # "Conclusion" is captured as discussion


def test_structured_finds_future_work_roman_numbering():
    text = (
        "I. Introduction\n" + _para("intro") + "\n"
        "II. Future Work\n" + _para("future") + "\n"
    )
    sections = _structured_sections(text)
    assert any(s["section_type"] == "future_work" for s in sections)


def test_window_fallback_when_no_heading():
    body = (
        "We present results. In future work we will tackle "
        + ("more challenges. " * 200)
    )
    win = _window_fallback(body)
    assert win is not None
    assert win["section_type"] == "fallback"
    assert "future work" in win["section_text"].lower()


def test_tail_fallback_only_when_long_enough():
    # Too short
    short = "Hello " * 50
    assert _tail_fallback(short) is None
    long = "word " * 500
    assert _tail_fallback(long) is not None


def test_extract_for_paper_caps_at_two_sections():
    text = (
        "1 Introduction\n" + _para("intro") + "\n"
        "2 Limitations\n" + _para("lim") + "\n"
        "3 Future Work\n" + _para("fw") + "\n"
        "4 Discussion\n" + _para("disc") + "\n"
        "References\n"
    )
    out = extract_sections_for_paper("paper-x", text)
    assert len(out) <= 2
    # priority order: limitations + future_work first
    types = [s["section_type"] for s in out]
    assert types == ["limitations", "future_work"] or types[0] == "limitations"
    for s in out:
        assert s["id"] == "paper-x"


def test_extract_for_paper_falls_through_to_window():
    text = "Abstract. " + _para("abstract") + "\nWe discuss future work in section 5 " + _para("body")
    out = extract_sections_for_paper("paper-y", text)
    assert len(out) >= 1
    assert out[0]["section_type"] in {"fallback", "tail", "discussion"}


def test_section_text_lookup_handles_numeric_looking_ids(tmp_path: Path):
    """Regression: arxiv IDs like '2106.05969' get parsed as floats by
    pd.read_json. The lookup must still work."""
    texts = pd.DataFrame([
        {"id": "2106.05969", "text": "Intro. " + _para("intro")
                                     + "\n1 Limitations\n" + _para("lim")
                                     + "\nReferences\n[1] foo"},
    ])
    jsonl = tmp_path / "texts.jsonl"
    texts.to_json(jsonl, orient="records", lines=True, force_ascii=False)
    sec_path = tmp_path / "sections.jsonl"
    sec_df = extract_all_sections(jsonl, sec_path)
    assert "2106.05969" in set(sec_df["id"].astype(str).tolist())

    # Read JSONL back the way openai_gaps would, and verify lookup works.
    on_disk = pd.read_json(sec_path, lines=True, dtype=False)
    on_disk["id"] = on_disk["id"].astype(str)
    text = _section_text_for_paper(on_disk, "2106.05969")
    assert len(text) > 100
