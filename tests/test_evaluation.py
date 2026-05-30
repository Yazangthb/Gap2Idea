"""Unit tests for non-API helpers in evaluation.py."""
from __future__ import annotations

from gap2idea.pipeline.evaluation import (
    _evidence_overlap,
    _falsifiability_gate,
    _normalise_judge_scores,
)


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


# ---------- _falsifiability_gate (PR-2) ----------

def _idea(pred: str = "", base: str = "RoBERTa-base") -> dict:
    return {
        "title": "X",
        "research_question": "Y",
        "falsifiable_prediction": pred,
        "named_baseline": base,
    }


def test_falsifiability_gate_pass_percentage():
    assert _falsifiability_gate(
        _idea("We expect at least 5% F1 improvement over the baseline.", "RoBERTa-base")
    ) is True


def test_falsifiability_gate_pass_inequality():
    assert _falsifiability_gate(
        _idea("Accuracy must exceed 0.85 on the MNLI dev set.", "BERT-large")
    ) is True


def test_falsifiability_gate_pass_pp_unit():
    assert _falsifiability_gate(
        _idea("At least 3pp gain over baseline on Spider.", "T5-base")
    ) is True


def test_falsifiability_gate_pass_geq_symbol():
    assert _falsifiability_gate(
        _idea("ROUGE-L >= 0.4 vs. the comparison method.", "BART")
    ) is True


def test_falsifiability_gate_fail_missing_baseline():
    assert _falsifiability_gate(
        _idea("We expect 5% improvement.", "")
    ) is False


def test_falsifiability_gate_fail_tbd_baseline():
    for bad in ["TBD", "tbd", "none", "N/A", "no baseline", "No-Baseline", "unspecified"]:
        assert _falsifiability_gate(_idea("5% gain.", bad)) is False, f"failed for {bad!r}"


def test_falsifiability_gate_fail_missing_prediction():
    assert _falsifiability_gate(_idea("", "RoBERTa-base")) is False


def test_falsifiability_gate_fail_no_quantitative_threshold():
    """A non-numeric prediction must not pass the gate even if a baseline is named."""
    assert _falsifiability_gate(
        _idea("The method will significantly outperform the baseline.", "BERT-base")
    ) is False


def test_falsifiability_gate_handles_missing_fields():
    """Old idea records lacking the fields entirely must return False, not crash."""
    assert _falsifiability_gate({"title": "X"}) is False
    assert _falsifiability_gate({}) is False
    assert _falsifiability_gate({"falsifiable_prediction": None, "named_baseline": None}) is False
