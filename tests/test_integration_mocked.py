"""End-to-end orchestration tests for the API-calling stages, with the
OpenAI / Semantic Scholar clients monkeypatched. Verifies post-processing,
TSV/JSONL serialisation, evidence-overlap math and report generation
without burning API quota.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ---------- shared fixtures ----------

@pytest.fixture
def fake_gaps() -> pd.DataFrame:
    rows = []
    for i, pid in enumerate(["p1", "p2", "p3"]):
        rows.append({"id": pid, "gap_sentence": f"future work on topic A {i} " * 5,
                     "paragraph_text": "para A", "confidence": 0.9 - i * 0.05,
                     "gap_type": "future_work", "cluster_id": 0})
    for i, pid in enumerate(["q1", "q2", "q3"]):
        rows.append({"id": pid, "gap_sentence": f"limitation on topic B {i} " * 5,
                     "paragraph_text": "para B", "confidence": 0.95 - i * 0.05,
                     "gap_type": "limitation", "cluster_id": 1})
    return pd.DataFrame(rows)


@pytest.fixture
def fake_pairs() -> pd.DataFrame:
    return pd.DataFrame([{
        "cluster_a": 0, "cluster_b": 1, "label_a": "A", "label_b": "B",
        "cosine_sim": 0.45, "paper_overlap": 0.0, "type_complementarity": 1.0,
        "bridge_score": 0.95,
    }])


@pytest.fixture
def fake_labels() -> pd.DataFrame:
    return pd.DataFrame([
        {"cluster_id": 0, "theme_label": "Theme A", "keywords": "a, b"},
        {"cluster_id": 1, "theme_label": "Theme B", "keywords": "c, d"},
    ])


@pytest.fixture
def fake_llm_idea() -> dict:
    return {
        "pair": {"cluster_a": 0, "cluster_b": 1},
        "idea": {
            "title": "A Method Bridging A and B",
            "research_question": "Can A inform B?",
            "method_sketch": "Train an encoder on A, transfer to B; evaluate with metric M on benchmark X vs baseline R.",
            "evaluation_plan": "Compare against baseline R on metric M.",
            "expected_contribution": "Cross-domain insight.",
            "assumptions_and_risks": "Assumes A and B share structure; risks include domain shift.",
            "evidence_used": [
                {"paper_id": "p1", "gap_sentence": "future work on topic A 0 " * 5},
                {"paper_id": "q1", "gap_sentence": "limitation on topic B 0 " * 5},
            ],
            "confidence": 0.7,
        },
    }


# ---------- generate-ideas batch path ----------

def test_generate_ideas_batch_end_to_end(tmp_path: Path, fake_gaps, fake_pairs, fake_labels, fake_llm_idea):
    from gap2idea.pipeline import openai_ideas

    with patch.object(openai_ideas, "_call_llm", return_value=fake_llm_idea), \
         patch.object(openai_ideas, "novelty_check", return_value={
             "max_similarity": 0.30, "mean_similarity": 0.20,
             "novelty_score": 0.70, "closest_paper": {
                 "paperId": "X", "title": "Existing Paper", "year": 2023, "similarity": 0.30,
             }, "n_hits": 5,
         }), \
         patch.object(openai_ideas, "get_llm_client", return_value=object()):

        out_tsv = tmp_path / "ideas.tsv"
        out_jsonl = tmp_path / "ideas_full.jsonl"

        df = openai_ideas.generate_ideas_batch(
            gaps=fake_gaps, pairs=fake_pairs, cluster_labels=fake_labels,
            out_tsv=out_tsv, out_jsonl=out_jsonl,
            n_pairs=1, check_novelty=True,
        )

    # TSV row
    assert len(df) == 1
    row = df.iloc[0]
    assert row["title"] == "A Method Bridging A and B"
    assert row["cluster_a"] == 0 and row["cluster_b"] == 1
    assert row["bridge_score"] == pytest.approx(0.95)
    assert row["novelty_score"] == pytest.approx(0.70)
    assert row["closest_paper_title"] == "Existing Paper"

    # JSONL provenance file
    records = [json.loads(l) for l in out_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(records) == 1
    rec = records[0]
    assert "prompt" in rec and "system" in rec
    assert "evidence_a" in rec and len(rec["evidence_a"]) > 0
    assert "evidence_b" in rec and len(rec["evidence_b"]) > 0
    assert rec["idea"]["title"] == "A Method Bridging A and B"
    assert rec["novelty"]["novelty_score"] == pytest.approx(0.70)


def test_generate_ideas_batch_no_novelty_check(tmp_path: Path, fake_gaps, fake_pairs, fake_labels, fake_llm_idea):
    """Verify the no-novelty path still runs cleanly."""
    from gap2idea.pipeline import openai_ideas

    with patch.object(openai_ideas, "_call_llm", return_value=fake_llm_idea), \
         patch.object(openai_ideas, "get_llm_client", return_value=object()):
        df = openai_ideas.generate_ideas_batch(
            gaps=fake_gaps, pairs=fake_pairs, cluster_labels=fake_labels,
            out_tsv=tmp_path / "ideas.tsv",
            out_jsonl=tmp_path / "ideas_full.jsonl",
            n_pairs=1, check_novelty=False,
        )
    assert len(df) == 1
    assert pd.isna(df.iloc[0]["novelty_score"]) or df.iloc[0]["novelty_score"] in (None, "")


# ---------- evaluate-ideas batch path ----------

def test_evaluate_ideas_end_to_end(tmp_path: Path):
    from gap2idea.pipeline import evaluation

    # build a fake ideas_full.jsonl
    rec = {
        "pair": {"cluster_a": 0, "cluster_b": 1, "label_a": "A", "label_b": "B"},
        "evidence_a": [{"paper_id": "p1", "gap_sentence": "gap-a"}],
        "evidence_b": [{"paper_id": "q1", "gap_sentence": "gap-b"}],
        "prompt": "...", "system": "...", "model": "fake",
        "idea": {
            "title": "Idea X",
            "research_question": "?", "method_sketch": "...",
            "evaluation_plan": "...", "expected_contribution": "...",
            "assumptions_and_risks": "...", "confidence": 0.7,
            "evidence_used": [
                {"paper_id": "p1", "gap_sentence": "gap-a"},
                {"paper_id": "hallucinated", "gap_sentence": "made up"},
            ],
        },
        "novelty": {"novelty_score": 0.6, "max_similarity": 0.4},
    }
    jsonl = tmp_path / "ideas_full.jsonl"
    jsonl.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    fake_scores = {
        "novelty": 4, "novelty_rationale": "ok",
        "specificity": 3, "specificity_rationale": "ok",
        "feasibility": 4, "feasibility_rationale": "ok",
        "evidence_grounding": 5, "evidence_grounding_rationale": "ok",
        "overall_critique": "Solid.",
    }
    with patch.object(evaluation, "_call_judge", return_value=fake_scores), \
         patch.object(evaluation, "get_llm_client", return_value=object()):
        df = evaluation.evaluate_ideas(jsonl, tmp_path / "idea_eval.tsv", judge_model="fake")

    assert len(df) == 1
    row = df.iloc[0]
    # composite = mean of the 4 axes
    assert row["composite"] == pytest.approx((4 + 3 + 4 + 5) / 4.0)
    # 1 of 2 evidence_used entries was in the fed set
    assert row["evidence_overlap"] == 0.5
    # S2 novelty propagates
    assert row["s2_novelty_score"] == pytest.approx(0.6)


def test_evaluation_report_writes_markdown(tmp_path: Path):
    from gap2idea.pipeline.evaluation import write_report

    eval_df = pd.DataFrame([{
        "title": "Idea X", "novelty": 4, "specificity": 3, "feasibility": 4,
        "evidence_grounding": 5, "composite": 4.0, "evidence_overlap": 0.5,
        "s2_novelty_score": 0.6, "overall_critique": "Solid.",
    }])
    out = tmp_path / "report.md"
    write_report(eval_df, pd.DataFrame(), out)
    content = out.read_text(encoding="utf-8")
    assert "Evaluation Report" in content
    assert "Idea X" in content
    assert "composite" in content.lower()


# ---------- Option A: within-cluster batch ----------

def test_generate_ideas_within_clusters_end_to_end(
    tmp_path: Path, fake_gaps, fake_labels, fake_llm_idea,
):
    from gap2idea.pipeline import openai_ideas

    with patch.object(openai_ideas, "_call_llm_within", return_value=fake_llm_idea), \
         patch.object(openai_ideas, "novelty_check", return_value={
             "novelty_score": 0.80, "max_similarity": 0.20, "closest_paper": None, "n_hits": 0,
         }), \
         patch.object(openai_ideas, "get_llm_client", return_value=object()):
        df = openai_ideas.generate_ideas_within_clusters(
            gaps=fake_gaps, cluster_labels=fake_labels,
            out_tsv=tmp_path / "ideas.tsv",
            out_jsonl=tmp_path / "ideas_full.jsonl",
            n_clusters=2, check_novelty=True,
        )

    assert len(df) == 2  # one idea per cluster
    assert set(df["mode"]) == {"within"}
    assert df.iloc[0]["cluster_b"] is None
    assert df.iloc[0]["bridge_score"] is None
    # provenance JSONL has one record per idea
    records = [
        json.loads(l) for l in (tmp_path / "ideas_full.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert len(records) == 2
    assert all(r["mode"] == "within" for r in records)
    assert all("evidence_a" in r and "evidence_b" not in r for r in records)


# ---------- Option B: method-gap retrieval + batch ----------

def test_retrieve_methods_picks_sweet_spot():
    """The retriever should rank methods inside (sim_low, sim_high) above
    methods outside, even when the outside ones are more similar overall."""
    import numpy as np
    from gap2idea.pipeline.openai_ideas import _retrieve_methods_for_cluster

    # Gap cluster centroid points along axis 0.
    gaps = pd.DataFrame([
        {"id": "p1", "gap_sentence": "x", "cluster_id": 0},
        {"id": "p2", "gap_sentence": "y", "cluster_id": 0},
    ])
    gap_emb = np.array([[1.0, 0.0], [1.0, 0.0]])  # centroid = [1, 0]

    # Three methods at varying similarity to centroid:
    methods = pd.DataFrame([
        {"id": "m1", "method_sentence": "too similar",   "method_type": "algorithm", "confidence": 0.9,  "paragraph_text": ""},
        {"id": "m2", "method_sentence": "sweet spot",    "method_type": "algorithm", "confidence": 0.85, "paragraph_text": ""},
        {"id": "m3", "method_sentence": "too distant",   "method_type": "algorithm", "confidence": 0.8,  "paragraph_text": ""},
    ])
    method_emb = np.array([
        [1.0, 0.0],          # sim = 1.00, outside sweet spot
        [0.5, np.sqrt(0.75)],  # sim = 0.50, IN sweet spot
        [0.0, 1.0],          # sim = 0.00, outside sweet spot
    ])
    out = _retrieve_methods_for_cluster(
        cluster_id=0, gap_embeddings=gap_emb, gaps=gaps,
        methods=methods, method_embeddings=method_emb,
        top_k=1, sim_low=0.30, sim_high=0.70,
    )
    assert len(out) == 1
    assert out.iloc[0]["id"] == "m2"  # the only one in the sweet spot


def test_retrieve_methods_falls_back_when_empty_sweetspot():
    """If no method is in the sweet spot, return the closest top_k anyway."""
    import numpy as np
    from gap2idea.pipeline.openai_ideas import _retrieve_methods_for_cluster

    gaps = pd.DataFrame([{"id": "p1", "gap_sentence": "x", "cluster_id": 0}])
    gap_emb = np.array([[1.0, 0.0]])
    methods = pd.DataFrame([
        {"id": "m1", "method_sentence": "high sim", "method_type": "algorithm", "confidence": 0.9, "paragraph_text": ""},
        {"id": "m2", "method_sentence": "low sim", "method_type": "algorithm", "confidence": 0.9, "paragraph_text": ""},
    ])
    method_emb = np.array([[1.0, 0.0], [0.0, 1.0]])  # sims 1.0 and 0.0, neither in (0.3, 0.7)
    out = _retrieve_methods_for_cluster(
        cluster_id=0, gap_embeddings=gap_emb, gaps=gaps,
        methods=methods, method_embeddings=method_emb,
        top_k=2, sim_low=0.30, sim_high=0.70,
    )
    # Fallback returned closest-first; we just verify it didn't return empty.
    assert len(out) >= 1


def test_generate_ideas_method_gap_end_to_end(
    tmp_path: Path, fake_gaps, fake_labels, fake_llm_idea,
):
    import numpy as np
    from gap2idea.pipeline import openai_ideas

    n_gaps = len(fake_gaps)
    rng = np.random.default_rng(0)
    gap_emb = rng.normal(size=(n_gaps, 8))
    gap_emb /= np.linalg.norm(gap_emb, axis=1, keepdims=True)

    methods = pd.DataFrame([
        {"id": "m_paper_1", "method_sentence": "We propose a new GNN.",
         "method_type": "algorithm", "confidence": 0.9, "paragraph_text": "abstract..."},
        {"id": "m_paper_2", "method_sentence": "We release a benchmark.",
         "method_type": "benchmark", "confidence": 0.8, "paragraph_text": "abstract..."},
    ])
    method_emb = rng.normal(size=(len(methods), 8))
    method_emb /= np.linalg.norm(method_emb, axis=1, keepdims=True)

    with patch.object(openai_ideas, "_call_llm_method_gap", return_value=fake_llm_idea), \
         patch.object(openai_ideas, "novelty_check", return_value={
             "novelty_score": 0.6, "max_similarity": 0.4, "closest_paper": None, "n_hits": 0,
         }), \
         patch.object(openai_ideas, "get_llm_client", return_value=object()):
        df = openai_ideas.generate_ideas_method_gap(
            gaps=fake_gaps, gap_embeddings=gap_emb,
            methods=methods, method_embeddings=method_emb,
            cluster_labels=fake_labels,
            out_tsv=tmp_path / "ideas.tsv",
            out_jsonl=tmp_path / "ideas_full.jsonl",
            n_clusters=2, check_novelty=True,
            # Wide sim window so we don't rely on real similarity values
            sim_low=-1.0, sim_high=1.0,
        )

    assert len(df) == 2
    assert set(df["mode"]) == {"method-gap"}
    assert "mean_method_similarity" in df.columns
    assert df.iloc[0]["cluster_b"] is None
    records = [
        json.loads(l) for l in (tmp_path / "ideas_full.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    assert all("candidate_methods" in r for r in records)
    assert all(len(r["candidate_methods"]) >= 1 for r in records)


# ---------- novelty_check uses real embedder + mocked S2 ----------

def test_novelty_check_with_mocked_s2():
    from gap2idea.pipeline import openai_ideas

    fake_hits = [
        {"paperId": "X", "title": "T1", "year": 2024,
         "abstract": "We propose a method bridging A and B with metric M baseline R."},
        {"paperId": "Y", "title": "T2", "year": 2023,
         "abstract": "Unrelated work on databases."},
    ]
    idea = {
        "title": "A Method Bridging A and B",
        "research_question": "Can A inform B?",
        "method_sketch": "Train on A, transfer to B, evaluate with M vs R.",
    }

    class FakeS2:
        def search(self, query, limit=10):
            return fake_hits

    class FakeEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            # deterministic 2-d embeddings: similar texts share a token
            out = []
            for t in texts:
                v = np.array([
                    sum(1 for w in t.lower().split() if w in {"a", "b", "bridging", "method"}),
                    sum(1 for w in t.lower().split() if w in {"databases", "unrelated"}),
                ], dtype=float)
                n = np.linalg.norm(v) or 1.0
                out.append(v / n)
            return np.array(out)

    nov = openai_ideas.novelty_check(idea, FakeS2(), FakeEmbedder(), top_k=2)
    assert 0.0 <= nov["novelty_score"] <= 1.0
    assert nov["n_hits"] == 2
    assert nov["closest_paper"]["paperId"] == "X"  # the relevant abstract
