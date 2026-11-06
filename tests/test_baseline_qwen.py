"""Task 5a baseline (scripts/bench/baseline_qwen.py): prompts, call plan, run loop and scoring.

No Qwen calls: the run loop is driven by a fake client.
"""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "bench"))
import baseline_qwen as b  # noqa: E402


@pytest.fixture(scope="module")
def ns():
    return b.funnel_prompts()


@pytest.fixture(scope="module")
def rows():
    return {d: b.load_rows(d, "test") for d in b.DATASETS}


def decision(text):
    """The decision rules and strictness clause: everything between the input text and the output text."""
    return text[text.index("Decision"):text.index("Return JSON")]


@pytest.mark.parametrize("dataset", list(b.DATASETS))
def test_every_prompt_builds_with_the_new_io_only(dataset, ns):
    for prompt in b.DATASETS[dataset]["prompts"]:
        text = b.build_prompt(dataset, prompt, ns)
        assert '{"positives":[<id>, ...]}' in text
        assert '"results"' not in text and '"keep"' not in text
        assert '"[id] sentence"' in text


@pytest.mark.parametrize("prompt", b.RUBRIC_NAMES)
def test_decision_rules_match_the_funnel(prompt, ns):
    funnel = ns["RUBRICS"][prompt]
    assert decision(b.build_prompt("gold", prompt, ns)) == decision(funnel)
    assert decision(b.build_prompt("sal", prompt, ns)) == decision(ns["limitation_scope"](funnel))
    assert decision(b.build_prompt("fws", prompt, ns)) == decision(ns["future_scope"](funnel))


def test_fws_prompt_keeps_the_funnel_definition_and_examples(ns):
    from gap2idea.pipeline.gap_llm_filter import SHOTS_VALIDATE_FWS, SYSTEM_VALIDATE_FWS
    text = b.build_prompt("fws", "validate_fws", ns)
    assert text.startswith(SYSTEM_VALIDATE_FWS)
    assert all(s in text for s, _ in SHOTS_VALIDATE_FWS)


def test_test_sets_match_the_task_sheet(rows):
    want = {"sal": (40, 7968, 205), "gold": (28, 10388, 527), "fws": (209, 1506, 220)}
    for d, (papers, sents, pos) in want.items():
        r = rows[d]
        assert (len(b.papers_of(r)), len(r), sum(x["label"] != "none" for x in r)) == (papers, sents, pos)


def test_call_counts(rows):
    want = {("sal", "A"): 40, ("sal", "B"): 101, ("sal", "C"): 399, ("gold", "A"): 28, ("gold", "B"): 117,
            ("gold", "C"): 520, ("fws", "A"): 209, ("fws", "C"): 76}
    for (d, s), n in want.items():
        assert len(list(b.units(rows[d], s))) == n


@pytest.mark.parametrize("setup", list(b.SETUPS))
def test_units_cover_every_sentence_once(setup, rows):
    r = rows["sal"]
    keys = [tuple(k) for u in b.units(r, setup) for k in u["keys"]]
    assert keys == [(x["pid"], x["idx"]) for x in r]
    for u in b.units(r, setup):
        lines = u["user"].splitlines()
        assert len(lines) == len(u["keys"]) and lines[0].startswith("[0] ")
        if setup != "C":
            assert len({k[0] for k in u["keys"]}) == 1


def test_run_plan_has_20_runs():
    assert len(b.RUN_PLAN) == 20 and len(set(b.RUN_PLAN)) == 20


class FakeQwen:
    """Fails the first call three times, sends one out-of-range id, otherwise keeps sentences with 'limitation'."""

    def __init__(self):
        self.calls = 0
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kw):
        self.calls += 1
        if self.calls <= 3:
            raise TimeoutError("simulated")
        lines = kw["messages"][1]["content"].splitlines()
        pos = [i for i, line in enumerate(lines) if "limitation" in line.lower()]
        if self.calls == 4:
            pos.append(999)
        msg = types.SimpleNamespace(content=json.dumps({"positives": pos}))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(finish_reason="stop", message=msg)],
                                     usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=5))


def test_run_retries_records_failures_and_resumes(tmp_path, monkeypatch):
    fake = FakeQwen()
    monkeypatch.setattr(b, "OUTDIR", tmp_path)
    monkeypatch.setattr(b, "qwen_client", lambda: fake)
    args = types.SimpleNamespace(dataset="sal", setup="B", prompt="strict", split="dev", limit_papers=2)
    b.run(args)
    recs = [json.loads(x) for x in (tmp_path / "sal_dev_B_strict.jsonl").read_text(encoding="utf-8").splitlines()]
    assert recs[0]["status"] == "failed" and recs[0]["attempts"] == 3 and "TimeoutError" in recs[0]["error"]
    assert recs[1]["status"] == "ok" and recs[1]["invalid_ids"] == [999]
    assert all(r["status"] == "ok" for r in recs[1:])
    for r in recs[1:]:
        assert r["kept"] == [r["keys"][i] for i in r["positives"]]
    calls = fake.calls
    b.run(args)
    assert fake.calls == calls  # everything already saved


def test_test_split_needs_a_committed_plan(monkeypatch):
    monkeypatch.setattr(b, "plan_is_committed", lambda: False)
    args = types.SimpleNamespace(dataset="sal", setup="A", prompt="strict", split="test", limit_papers=0)
    with pytest.raises(SystemExit, match="Commit"):
        b.run(args)


def test_scoring_and_paired_bootstrap():
    rows = [dict(pid=p, idx=i, sentence="", label="limitation" if i == 0 else "none") for p in "abcd" for i in range(3)]
    kept = {("a", 0), ("b", 0), ("c", 1)}
    tab = b.per_paper(rows, kept)
    assert tab["a"] == [1, 0, 1] and tab["c"] == [0, 1, 1]
    assert b.prf(2, 1, 4) == pytest.approx((2 / 3, 0.5, 4 / 7))
    same = b.boot(rows, kept, kept, n=200)
    assert same == {"f1_diff": 0.0, "ci95": [0.0, 0.0]}
    ci = b.boot(rows, kept, n=200)
    assert set(ci) == {"precision", "recall", "f1"}


def test_results_table_renders():
    res = {"PRF": [0.5, 0.8, 0.615], "ci95": {"precision": [0.4, 0.6], "recall": [0.7, 0.9], "f1": [0.5, 0.7]},
           "complete": True, "failed_calls": 1, "failed_sentences": 20, "empty_answers": 3, "input_tokens": 12345,
           "seconds": 90, "vs_funnel": {"f": {"f1_diff": -0.05, "ci95": [-0.1, 0.0]}},
           "recall_by_type": {"limitation": 0.5, "future_work": 1.0, "research_gap": 0.25}}
    table = b.results_table({"gold": {"funnel": {"f": {"PRF": [0.5, 0.85, 0.64]}}, "gold_test_C_keep_biased": res}})
    assert "| arXiv gold | C | keep_biased | 50.0% (40.0–60.0) |" in table
    assert "-0.050 (-0.100 to +0.000) vs `f`" in table and "1 (20 sentences)" in table
    assert "| C | keep_biased | 50.0% | 100.0% | 25.0% |" in table
