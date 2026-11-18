import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "bench"))
import baseline_report
from baseline_evidence import digest
from export_funnel_comparison import export


def fixture():
    rows = [{"pid": "p", "idx": 0, "label": "limitation"}, {"pid": "p", "idx": 1, "label": "none"}]
    candidates = [["p", 0], ["p", 1]]
    verdicts = [{"key": ["p", 0], "keep": True}, {"key": ["p", 1], "keep": False}]
    manifest = {"name": "test", "dataset": "sal", "split": "test", "mode": "balanced",
                "model_identity": {"repo": "qwen", "revision": "r1"},
                "evaluation_data_sha256": digest(rows), "candidate_keys_sha256": digest(candidates)}
    return manifest, candidates, verdicts, rows


def test_export_requires_complete_candidate_verdicts_and_matching_data():
    manifest, candidates, verdicts, rows = fixture()
    out = export(manifest, candidates, verdicts, rows, ["A", "C"])
    assert [x["input_type"] for x in out] == ["A", "C"]
    assert out[0]["kept"] == [["p", 0]]
    with pytest.raises(ValueError, match="cover all"):
        export(manifest, candidates, verdicts[:1], rows, ["A"])
    with pytest.raises(ValueError, match="Evaluation data"):
        export(manifest, candidates, verdicts, rows[:1], ["A"])
    with pytest.raises(ValueError, match="Duplicate, foreign"):
        export(manifest, candidates, verdicts + verdicts[:1], rows, ["A"])


def test_report_only_compares_matching_identity_and_complete_baseline(tmp_path, monkeypatch):
    manifest, candidates, verdicts, rows = fixture()
    comparisons = export(manifest, candidates, verdicts, rows, ["A"])
    comp_path = tmp_path / "comparisons.json"
    comp_path.write_text(json.dumps(comparisons), encoding="utf-8")
    folder = tmp_path / "model" / "test"
    folder.mkdir(parents=True)
    (folder / "sal_test_A_neutral.jsonl").write_text("", encoding="utf-8")
    baseline_manifest = {"dataset": "sal", "data_sha256": digest(rows), "prompt": "neutral", "setup": "A",
                         "model": "qwen", "model_identity": manifest["model_identity"]}
    (folder / "sal_test_A_neutral.manifest.json").write_text(json.dumps(baseline_manifest), encoding="utf-8")
    monkeypatch.setattr(baseline_report.b, "load_rows", lambda *_: rows)
    monkeypatch.setattr(baseline_report.b, "boot", lambda *args: {"f1_diff": 0.25, "ci95": [0.1, 0.4]})
    def score(complete):
        return {("p", 1)}, {"PRF": [0., 0., 0.], "ci95": {"f1": [0., 0.]},
                            "coverage": {"all_calls_recorded": complete, "answered_sentences": 2,
                                         "requested_sentences": 2, "failed_calls": 0},
                            "input_tokens": 1, "output_tokens": 1, "seconds": 1}
    monkeypatch.setattr(baseline_report.b, "score_run", lambda *_: score(False))
    assert baseline_report.report(tmp_path, comp_path)[0]["funnel_minus_baseline"] == []
    assert "INCOMPLETE" in (tmp_path / "handover" / "RESULTS.md").read_text(encoding="utf-8")
    monkeypatch.setattr(baseline_report.b, "score_run", lambda *_: score(True))
    assert baseline_report.report(tmp_path, comp_path)[0]["funnel_minus_baseline"][0]["f1_diff"] == 0.25
    baseline_manifest["model_identity"]["revision"] = "different"
    (folder / "sal_test_A_neutral.manifest.json").write_text(json.dumps(baseline_manifest), encoding="utf-8")
    assert baseline_report.report(tmp_path, comp_path)[0]["funnel_minus_baseline"] == []
