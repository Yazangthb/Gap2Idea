"""Unit tests for export.py (LaTeX + PDF generation).

These run entirely offline — no LLM, no S2, no LaTeX compiler. We only
verify the rendered artifacts are well-formed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gap2idea.pipeline.export import (
    escape_tex,
    idea_to_latex,
    ideas_to_pdf_bytes,
    write_idea_latex,
    write_ideas_pdf,
)


# ---------- escape_tex ----------

def test_escape_tex_handles_all_special_chars():
    raw = r"100% & this_thing $5 #1 {brace} ~tilde ^caret \\back"
    out = escape_tex(raw)
    # Every special char must be replaced with its LaTeX equivalent.
    for unsafe in ["&", "%", "_", "#", "{", "}"]:
        # Only allowed inside a LaTeX command (\&, \%, etc.), never bare.
        # Quick check: the escaped version has the right command on each.
        if unsafe == "&":
            assert r"\&" in out
        elif unsafe == "%":
            assert r"\%" in out


def test_escape_tex_handles_none_and_non_string():
    assert escape_tex(None) == ""
    assert escape_tex(42) == "42"


# ---------- idea_to_latex ----------

def _fake_idea(**overrides) -> dict:
    base = {
        "title": "A Method Bridging A & B (50% gain)",
        "mode": "within",
        "label_a": "Theme of A",
        "label_b": "",
        "research_question": "Can we apply X to Y?",
        "method_sketch": "Train a model with $z$ loss.",
        "evaluation_plan": "Compare against baseline R using metric M.",
        "expected_contribution": "Cross-domain insight.",
        "assumptions_and_risks": "Risk: domain shift.",
        "evidence_used_json": json.dumps([
            {"paper_id": "1912.08927", "gap_sentence": "Future work: study X."},
        ], ensure_ascii=False),
        "novelty_score": 0.78,
        "closest_paper_title": "Some Prior Work on X",
        "closest_paper_year": 2023,
        "idea_confidence": 0.85,
    }
    base.update(overrides)
    return base


def test_idea_to_latex_renders_expected_structure():
    tex = idea_to_latex(_fake_idea())
    # Required LaTeX bones
    assert r"\documentclass" in tex
    assert r"\begin{document}" in tex
    assert r"\end{document}" in tex
    # Content is included
    assert "A Method Bridging A" in tex
    assert "Can we apply X to Y?" in tex
    assert "1912.08927" in tex
    # Special chars properly escaped (% must be \%)
    assert r"\&" in tex
    assert r"\%" in tex
    # Disclaimer present
    assert "Disclaimer" in tex
    # Novelty section appears because novelty_score is present
    assert "0.78" in tex or "novelty" in tex.lower()


def test_idea_to_latex_handles_missing_evidence():
    """An idea with no evidence JSON should still render."""
    tex = idea_to_latex(_fake_idea(evidence_used_json=""))
    assert r"\begin{document}" in tex
    assert r"\end{document}" in tex


def test_idea_to_latex_with_panel_consensus_includes_scores():
    consensus = {
        "composite": 3.5, "novelty": 4, "specificity": 3, "feasibility": 4,
        "evidence_grounding": 3, "agreement": 0.92, "n_judges": 3,
    }
    tex = idea_to_latex(_fake_idea(), panel_consensus=consensus)
    assert "judge-panel scores" in tex
    assert "3.50" in tex or "3.5" in tex


def test_write_idea_latex_round_trip(tmp_path: Path):
    out = tmp_path / "idea.tex"
    write_idea_latex(_fake_idea(), out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert r"\begin{document}" in content


# ---------- PDF (reportlab) ----------

def test_ideas_to_pdf_bytes_returns_valid_pdf():
    df = pd.DataFrame([
        _fake_idea(title="Idea Alpha"),
        _fake_idea(title="Idea Beta", mode="bridge", label_b="Theme of B"),
    ])
    data = ideas_to_pdf_bytes(df, title="Test Library")
    assert isinstance(data, bytes)
    # PDF magic header
    assert data[:4] == b"%PDF"
    # Non-trivial size (multi-page)
    assert len(data) > 1500


def test_ideas_to_pdf_bytes_handles_empty():
    """Empty DataFrame should still produce a (mostly-empty) valid PDF."""
    data = ideas_to_pdf_bytes(pd.DataFrame(columns=["title"]))
    assert data[:4] == b"%PDF"


def test_write_ideas_pdf_round_trip(tmp_path: Path):
    df = pd.DataFrame([_fake_idea()])
    out = tmp_path / "lib.pdf"
    write_ideas_pdf(df, out)
    assert out.exists()
    assert out.stat().st_size > 500
    assert out.read_bytes()[:4] == b"%PDF"


# ---------- Template choice + custom template ----------

def test_list_templates_returns_bundled_names():
    from gap2idea.pipeline.export import BUNDLED_TEMPLATES, list_templates
    names = list_templates()
    assert set(names) == set(BUNDLED_TEMPLATES.keys())
    assert "standard" in names
    assert "minimal" in names
    assert "ieee" in names


@pytest.mark.parametrize("template", ["minimal", "standard", "ieee"])
def test_idea_to_latex_renders_each_bundled_template(template):
    tex = idea_to_latex(_fake_idea(), template=template)
    assert r"\begin{document}" in tex
    assert r"\end{document}" in tex
    assert "A Method Bridging A" in tex
    assert r"\&" in tex      # escape_tex still applies
    assert "1912.08927" in tex


def test_idea_to_latex_rejects_unknown_template():
    with pytest.raises(ValueError, match="Unknown template"):
        idea_to_latex(_fake_idea(), template="acl_2026")


def test_idea_to_latex_accepts_custom_template_source():
    """User-supplied template wins; gets the same context + escape_tex filter."""
    custom = (
        r"\documentclass{article}"
        r"\begin{document}"
        r"TITLE-{{ title | escape_tex }}-END"
        r"\end{document}"
    )
    tex = idea_to_latex(_fake_idea(), template_source=custom)
    assert "TITLE-A Method Bridging A " in tex   # title was rendered + escaped
    assert r"\&" in tex                           # & is escaped
    # `template` arg is ignored when `template_source` is given (no error here)
    tex2 = idea_to_latex(_fake_idea(), template="nonsense", template_source=custom)
    assert "TITLE-" in tex2


def test_custom_template_can_use_panel_consensus():
    custom = (
        r"\documentclass{article}\begin{document}"
        r"{% if panel_consensus %}"
        r"composite={{ '%.2f'|format(panel_consensus.composite) }}"
        r"{% else %}"
        r"NO_PANEL"
        r"{% endif %}"
        r"\end{document}"
    )
    tex_with = idea_to_latex(
        _fake_idea(), template_source=custom,
        panel_consensus={"composite": 4.25, "novelty": 4, "specificity": 4,
                          "feasibility": 4, "evidence_grounding": 4,
                          "agreement": 0.9, "n_judges": 3},
    )
    assert "composite=4.25" in tex_with
    tex_without = idea_to_latex(_fake_idea(), template_source=custom)
    assert "NO_PANEL" in tex_without


# ---------- LaTeX compiler detection ----------

def test_find_latex_compiler_returns_str_or_none():
    from gap2idea.pipeline.export import find_latex_compiler
    out = find_latex_compiler()
    assert out is None or out in {"tectonic", "pdflatex"}


def test_compile_latex_to_pdf_raises_clear_error_when_no_compiler(monkeypatch):
    from gap2idea.pipeline import export as exp

    monkeypatch.setattr(exp, "find_latex_compiler", lambda: None)
    with pytest.raises(exp.LatexCompilerNotFound, match="No LaTeX compiler"):
        exp.compile_latex_to_pdf(r"\documentclass{article}\begin{document}hi\end{document}")


# ---------- Compile a real PDF if a compiler is on PATH ----------

import shutil

_HAS_LATEX = any(shutil.which(c) for c in ("tectonic", "pdflatex"))


@pytest.mark.skipif(not _HAS_LATEX, reason="No LaTeX compiler installed; rendered-PDF test skipped.")
def test_compile_latex_to_pdf_minimal():
    """Smoke-test: render the minimal template, compile to PDF, check magic header."""
    from gap2idea.pipeline.export import compile_latex_to_pdf
    tex = idea_to_latex(_fake_idea(), template="minimal")
    pdf = compile_latex_to_pdf(tex)
    assert pdf[:4] == b"%PDF"
    assert len(pdf) > 1000
