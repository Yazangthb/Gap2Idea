"""Unit tests for paper_drafter.py.

All LLM calls are mocked. We're testing:
  • the drafter post-processing (related_work filtering against known IDs)
  • that the schema-compliant fake response flows through unchanged
  • the export.draft_and_render_paper helper
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from gap2idea.pipeline import paper_drafter


# ---------- fixtures ----------

@pytest.fixture
def fake_idea():
    import json as _json
    return {
        "title": "Bridging A and B for robust X",
        "research_question": "Can we apply X to Y to obtain Z?",
        "method_sketch": "Train an encoder with loss L.",
        "evaluation_plan": "Evaluate on dataset D vs baseline B.",
        "expected_contribution": "First evidence that X helps Y.",
        "assumptions_and_risks": "Assumes X is transferable.",
        "label_a": "Theme A",
        "label_b": "Theme B",
        "evidence_used_json": _json.dumps([
            {"paper_id": "1912.08927", "gap_sentence": "future work: study X"},
            {"paper_id": "2007.00128", "gap_sentence": "limitation: lack of Y data"},
        ]),
    }


@pytest.fixture
def fake_paper_draft():
    """A schema-compliant draft. All fields populated."""
    return {
        "abstract": "We propose to apply X to Y, motivated by recurring gaps.",
        "introduction": {
            "motivation": "Motivation paragraph 1. Motivation paragraph 2.",
            "contributions": ["A new method", "An evaluation protocol", "An open-source release"],
            "paper_structure": "Section 2 reviews ... Section 3 introduces ...",
        },
        "related_work": [
            {
                "paper_id": "1912.08927", "title": "Earlier study of X",
                "relevance": "Closest prior work on X.",
                "how_we_differ": "We extend to Y.",
            },
            {
                "paper_id": "INVALID_99999", "title": "Made-up",
                "relevance": "Should be filtered out — invalid id.",
                "how_we_differ": "n/a",
            },
        ],
        "method": {
            "overview": "Overview.", "approach": "Approach.",
            "architecture_or_algorithm": "Algorithm.", "training_setup": "Train with Adam.",
        },
        "experimental_setup": {
            "datasets": ["GLUE", "CIFAR-10"],
            "baselines": ["Vanilla baseline", "Prior-art baseline"],
            "metrics": ["Accuracy", "F1"],
            "implementation_notes": "Standard PyTorch on a single A100.",
            "human_work_required": "Implementation, ablations, manual error analysis.",
        },
        "expected_results": "We expect improvements on the long-tail subset.",
        "discussion": {
            "limitations": "Single-domain study.",
            "ethical_considerations": "No PII involved.",
            "future_work": "Extension to multilingual.",
        },
        "conclusion": "We described a plan for combining X and Y.",
    }


# ---------- drafter ----------

def test_draft_full_paper_returns_schema_compliant_dict(fake_idea, fake_paper_draft):
    """Happy path: LLM returns a valid draft, drafter passes it through."""
    with patch.object(paper_drafter, "_call_drafter", return_value=fake_paper_draft), \
         patch.object(paper_drafter, "get_llm_client", return_value=object()):
        out = paper_drafter.draft_full_paper(fake_idea)

    assert set(out.keys()) >= {
        "abstract", "introduction", "related_work", "method",
        "experimental_setup", "expected_results", "discussion", "conclusion",
    }
    assert out["abstract"]
    assert len(out["introduction"]["contributions"]) == 3
    assert out["experimental_setup"]["human_work_required"]


def test_draft_full_paper_filters_invalid_related_work_ids(fake_idea, fake_paper_draft):
    """The post-processing should drop related_work entries citing unknown IDs."""
    with patch.object(paper_drafter, "_call_drafter", return_value=fake_paper_draft), \
         patch.object(paper_drafter, "get_llm_client", return_value=object()):
        out = paper_drafter.draft_full_paper(fake_idea)

    # Input draft has two related-work entries: one valid (1912.08927 is in
    # evidence_used_json), one invalid (INVALID_99999). The invalid one must
    # be dropped by the post-processing step.
    ids = {rw["paper_id"] for rw in out["related_work"]}
    assert "1912.08927" in ids
    assert "INVALID_99999" not in ids


def test_draft_full_paper_accepts_explicit_evidence(fake_idea, fake_paper_draft):
    """If `evidence` is passed, it's used directly (overrides evidence_used_json)."""
    explicit_ev = [{"paper_id": "INVALID_99999", "gap_sentence": "x"}]
    with patch.object(paper_drafter, "_call_drafter", return_value=fake_paper_draft), \
         patch.object(paper_drafter, "get_llm_client", return_value=object()):
        out = paper_drafter.draft_full_paper(fake_idea, evidence=explicit_ev)

    # With INVALID_99999 now in evidence, that entry should survive; the
    # 1912.08927 entry should now be the one dropped.
    ids = {rw["paper_id"] for rw in out["related_work"]}
    assert "INVALID_99999" in ids
    assert "1912.08927" not in ids


def test_draft_full_paper_accepts_prior_art(fake_idea, fake_paper_draft):
    """prior_art IDs should also count as known for related_work filtering."""
    prior_art = [{"paperId": "INVALID_99999", "title": "T"}]
    with patch.object(paper_drafter, "_call_drafter", return_value=fake_paper_draft), \
         patch.object(paper_drafter, "get_llm_client", return_value=object()):
        out = paper_drafter.draft_full_paper(fake_idea, prior_art=prior_art)
    ids = {rw["paper_id"] for rw in out["related_work"]}
    assert "INVALID_99999" in ids


# ---------- template renders full_paper structure ----------

def test_standard_template_renders_full_paper(fake_idea, fake_paper_draft):
    """When full_paper is supplied, the LaTeX includes abstract + related work
    + human-work-required TODO boxes."""
    from gap2idea.pipeline.export import idea_to_latex
    tex = idea_to_latex(fake_idea, template="standard", full_paper=fake_paper_draft)
    # Abstract section appears
    assert r"\begin{abstract}" in tex
    assert "We propose to apply X to Y" in tex
    # Related work section
    assert "Related Work" in tex
    assert "Earlier study of X" in tex
    # TODO box for human work
    assert "Human work required" in tex
    assert "manual error analysis" in tex.lower()
    # Method subsections
    assert r"\subsection{Approach}" in tex
    # Experimental setup includes datasets/baselines/metrics
    assert "GLUE" in tex
    assert "CIFAR-10" in tex
    assert "Accuracy" in tex
    # Expected results section is present
    assert "Expected Results" in tex
    # Skeleton-mode prose should NOT appear
    assert "Author note: expand with literature framing" not in tex


def test_minimal_template_renders_full_paper(fake_idea, fake_paper_draft):
    from gap2idea.pipeline.export import idea_to_latex
    tex = idea_to_latex(fake_idea, template="minimal", full_paper=fake_paper_draft)
    assert r"\begin{abstract}" in tex
    assert "We propose to apply X to Y" in tex
    assert "Human work required" in tex
    # Doesn't depend on tcolorbox (minimal uses \fbox{\parbox{}})
    assert r"\usepackage{tcolorbox}" not in tex


def test_ieee_template_renders_full_paper(fake_idea, fake_paper_draft):
    from gap2idea.pipeline.export import idea_to_latex
    tex = idea_to_latex(fake_idea, template="ieee", full_paper=fake_paper_draft)
    assert r"\begin{abstract}" in tex
    assert "Earlier study of X" in tex
    # IEEE template uses multicol
    assert r"\begin{multicols}" in tex


def test_skeleton_mode_still_works_when_no_full_paper(fake_idea):
    """No full_paper supplied -> falls back to the skeleton view."""
    from gap2idea.pipeline.export import idea_to_latex
    tex = idea_to_latex(fake_idea, template="standard")
    assert r"\begin{abstract}" in tex
    # Skeleton mode renders the author-note text
    assert "Author note:" in tex
    # No related-work section
    assert "Related Work" not in tex


# ---------- draft_and_render_paper helper ----------

def test_draft_and_render_paper_returns_tex_and_data(fake_idea, fake_paper_draft):
    from gap2idea.pipeline import export as exp

    with patch.object(paper_drafter, "_call_drafter", return_value=fake_paper_draft), \
         patch.object(paper_drafter, "get_llm_client", return_value=object()):
        tex, fp = exp.draft_and_render_paper(fake_idea, template="standard")

    assert isinstance(tex, str) and r"\begin{document}" in tex
    assert isinstance(fp, dict)
    assert fp["abstract"] == fake_paper_draft["abstract"]
    # TeX must reflect the full_paper data
    assert "Earlier study of X" in tex


# ---------- defence in depth: empty related_work passes ----------

def test_drafter_tolerates_empty_related_work(fake_idea, fake_paper_draft):
    fake_paper_draft = {**fake_paper_draft, "related_work": []}
    with patch.object(paper_drafter, "_call_drafter", return_value=fake_paper_draft), \
         patch.object(paper_drafter, "get_llm_client", return_value=object()):
        out = paper_drafter.draft_full_paper(fake_idea)
    assert out["related_work"] == []

    from gap2idea.pipeline.export import idea_to_latex
    tex = idea_to_latex(fake_idea, template="standard", full_paper=out)
    # Should render the fallback message
    assert "Related work could not be drawn" in tex
