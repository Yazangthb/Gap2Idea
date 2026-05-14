"""Unit tests for non-API helpers in evaluation.py."""
from __future__ import annotations

from gap2idea.pipeline.evaluation import _evidence_overlap, _normalise_judge_scores


def test_evidence_overlap_perfect():
    rec = {
        "evidence_a": [{"paper_id": "p1", "gap_sentence": "alpha"}],
        "evidence_b": [{"paper_id": "p2", "gap_sentence": "beta"}],
        "idea": {
            "evidence_used": [
                {"paper_id": "p1", "gap_sentence": "alpha"},
                {"paper_id": "p2", "gap_sentence": "beta"},
            ]
        },
    }
    assert _evidence_overlap(rec) == 1.0


def test_evidence_overlap_partial():
    rec = {
        "evidence_a": [{"paper_id": "p1", "gap_sentence": "alpha"}],
        "evidence_b": [{"paper_id": "p2", "gap_sentence": "beta"}],
        "idea": {
            "evidence_used": [
                {"paper_id": "p1", "gap_sentence": "alpha"},
                {"paper_id": "p9", "gap_sentence": "hallucinated"},
            ]
        },
    }
    assert _evidence_overlap(rec) == 0.5


def test_evidence_overlap_no_evidence_used_is_zero():
    rec = {"evidence_a": [], "evidence_b": [], "idea": {"evidence_used": []}}
    assert _evidence_overlap(rec) == 0.0


def test_evidence_overlap_strips_whitespace():
    rec = {
        "evidence_a": [{"paper_id": "p1", "gap_sentence": "alpha"}],
        "evidence_b": [],
        "idea": {"evidence_used": [{"paper_id": "p1", "gap_sentence": "  alpha  "}]},
    }
    assert _evidence_overlap(rec) == 1.0


# ---------- _normalise_judge_scores ----------

def test_normalise_complete_response():
    raw = {
        "novelty": 4, "novelty_rationale": "yes",
        "specificity": 3, "specificity_rationale": "ok",
        "feasibility": 5, "feasibility_rationale": "fine",
        "evidence_grounding": 2, "evidence_grounding_rationale": "weak",
        "overall_critique": "Solid.",
    }
    out = _normalise_judge_scores(raw)
    assert out["novelty"] == 4
    assert out["evidence_grounding_rationale"] == "weak"
    assert out["overall_critique"] == "Solid."


def test_normalise_missing_rationales():
    """Regression: Claude via OpenRouter occasionally drops the *_rationale
    fields. We must not KeyError; default to empty string."""
    raw = {"novelty": 3, "specificity": 3, "feasibility": 3, "evidence_grounding": 3}
    out = _normalise_judge_scores(raw)
    assert out["novelty"] == 3
    assert out["novelty_rationale"] == ""
    assert out["overall_critique"] == ""


def test_normalise_accepts_alternate_key_spellings():
    raw = {
        "novelty": 4, "novelty_reason": "alt key",
        "specificity": 3, "specificity_explanation": "alt key 2",
        "feasibility": 4, "feasibility_rationale": "canonical",
        "evidence_grounding": 2,
        "critique": "alt overall",
    }
    out = _normalise_judge_scores(raw)
    assert out["novelty_rationale"] == "alt key"
    assert out["specificity_rationale"] == "alt key 2"
    assert out["feasibility_rationale"] == "canonical"
    assert out["overall_critique"] == "alt overall"


def test_normalise_coerces_string_scores():
    raw = {"novelty": "4", "specificity": "3.0", "feasibility": "bad", "evidence_grounding": 2}
    out = _normalise_judge_scores(raw)
    assert out["novelty"] == 4
    assert out["specificity"] == 3
    assert out["feasibility"] == 0  # default
    assert out["evidence_grounding"] == 2
