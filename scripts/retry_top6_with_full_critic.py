"""Re-run the 6 winning ideas with --max-critic-iter 3 and accept_score 4.0
so the critic actually converges (the first pass capped iter=1, so every
idea exited with verdict='revise' at score 3.0).

Sources:
  AI ideas (clusters 6, 7, 8): re-sample the same diverse evidence from
    gaps_with_clusters.tsv and replay synthesise_with_critic.
  Math ideas: re-import the BATCHES from gen_math_ideas.py and replay.

Output:
  artifacts/ideas_v2.tsv  — 6 fresh rows side-by-side with the v1 scores.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from gap2idea.pipeline.agents import synthesise_with_critic
from gap2idea.pipeline.evaluation import (
    DEFAULT_JUDGE_PANEL, _aggregate_panel, _call_judge,
    _falsifiability_gate, _normalise_judge_scores,
)
from gap2idea.pipeline.llm import get_llm_client
from gap2idea.pipeline.openai_ideas import _diverse_evidence, _evidence_payload

from gen_math_ideas import BATCHES as MATH_BATCHES


AI_CLUSTERS = [8, 7, 6]              # by descending v1 composite (3.58, 3.42, 3.33)
K_EVIDENCE = 6
MAX_CRITIC_ITER = 3
ACCEPT_SCORE = 4.0


async def judge_one(idea: dict) -> dict:
    client = get_llm_client()
    results: list[tuple[str, dict]] = []
    for m in DEFAULT_JUDGE_PANEL:
        try:
            raw = _call_judge(client, idea, model=m)
            results.append((m, _normalise_judge_scores(raw)))
        except Exception as e:
            print(f"    judge {m} failed: {e}", file=sys.stderr)
    if not results:
        return {"composite": 0.0, "agreement": 0.0, "n_judges": 0}
    return _aggregate_panel(results)


def flat_row(idea: dict, *, mode: str, cluster_a: int, cluster_b,
             label_a: str, label_b: str, result: dict, consensus: dict) -> dict:
    return {
        "mode":                  mode,
        "cluster_a":             cluster_a,
        "cluster_b":             cluster_b,
        "label_a":               label_a,
        "label_b":               label_b,
        "title":                 idea["title"],
        "research_question":     idea["research_question"],
        "method_sketch":         idea["method_sketch"],
        "evaluation_plan":       idea["evaluation_plan"],
        "expected_contribution": idea["expected_contribution"],
        "assumptions_and_risks": idea["assumptions_and_risks"],
        "falsifiable_prediction": idea.get("falsifiable_prediction", ""),
        "named_baseline":        idea.get("named_baseline", ""),
        "idea_confidence":       float(idea["confidence"]),
        "evidence_used_json":    json.dumps(idea.get("evidence_used", []), ensure_ascii=False),
        "novelty_score":         None,
        "max_similarity_to_prior": None,
        "closest_paper_title":   "",
        "closest_paper_year":    "",
        "closest_paper_id":      "",
        "n_critic_iterations":   result.get("_n_iterations", 0),
        "panel_composite":       consensus.get("composite"),
        "panel_agreement":       consensus.get("agreement"),
        "panel_n_judges":        consensus.get("n_judges"),
        "falsifiability_gate_passed": _falsifiability_gate(idea),
        # the final critic verdict + score, so we can compare against v1
        "critic_final_verdict":  result["_critique_history"][-1]["verdict"]
                                  if result.get("_critique_history") else "",
        "critic_final_score":    result["_critique_history"][-1]["score"]
                                  if result.get("_critique_history") else 0.0,
    }


async def replay_ai(gaps: pd.DataFrame, cluster_labels: pd.DataFrame,
                     cluster_a: int) -> dict | None:
    label_map = dict(zip(cluster_labels["cluster_id"].astype(int),
                          cluster_labels["theme_label"]))
    label_a = label_map.get(int(cluster_a), "")
    ev_df = _diverse_evidence(gaps, int(cluster_a), k=K_EVIDENCE)
    if ev_df.empty:
        print(f"  AI cluster {cluster_a}: no evidence", file=sys.stderr)
        return None
    ev = _evidence_payload(ev_df)
    print(f"\n[AI cluster={cluster_a}] {label_a}")
    print(f"   evidence rows: {len(ev)}")
    result = await synthesise_with_critic(
        mode="within",
        cluster_a=int(cluster_a),
        cluster_b=None,
        label_a=label_a,
        label_b="",
        gaps_df=gaps,
        fed_evidence_a=ev,
        max_iterations=MAX_CRITIC_ITER,
        accept_score=ACCEPT_SCORE,
    )
    last = result["_critique_history"][-1]
    print(f"   critic final: verdict={last['verdict']} score={last['score']:.2f} "
          f"(after {result.get('_n_iterations', 0)} revisions)")
    consensus = await judge_one(result["idea"])
    print(f"   panel composite={consensus.get('composite', 0):.2f} "
          f"agreement={consensus.get('agreement', 0):.2f}")
    return flat_row(result["idea"], mode="within",
                    cluster_a=int(cluster_a), cluster_b=None,
                    label_a=label_a, label_b="",
                    result=result, consensus=consensus)


async def replay_math(batch: dict, idx: int) -> dict | None:
    print(f"\n[math batch {idx}] {batch['label_a']} × {batch['label_b']}")
    result = await synthesise_with_critic(
        mode="bridge",
        cluster_a=-100 - idx,
        cluster_b=-200 - idx,
        label_a=batch["label_a"],
        label_b=batch["label_b"],
        gaps_df=pd.DataFrame(),
        fed_evidence_a=batch["fed_a"],
        fed_evidence_b=batch["fed_b"],
        max_iterations=MAX_CRITIC_ITER,
        accept_score=ACCEPT_SCORE,
    )
    last = result["_critique_history"][-1]
    print(f"   critic final: verdict={last['verdict']} score={last['score']:.2f} "
          f"(after {result.get('_n_iterations', 0)} revisions)")
    consensus = await judge_one(result["idea"])
    print(f"   panel composite={consensus.get('composite', 0):.2f} "
          f"agreement={consensus.get('agreement', 0):.2f}")
    return flat_row(result["idea"], mode="math-targeted",
                    cluster_a=-100 - idx, cluster_b=-200 - idx,
                    label_a=batch["label_a"], label_b=batch["label_b"],
                    result=result, consensus=consensus)


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    gaps = pd.read_csv(REPO / "artifacts" / "gaps_with_clusters.tsv",
                       sep="\t", dtype={"id": str})
    labels = pd.read_csv(REPO / "artifacts" / "cluster_labels.tsv", sep="\t")

    rows: list[dict] = []

    # AI: top 3 by v1 composite
    for cid in AI_CLUSTERS:
        try:
            r = await replay_ai(gaps, labels, cid)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  AI cluster {cid} failed: {e}", file=sys.stderr)

    # Math: the 3 batches from gen_math_ideas
    for i, batch in enumerate(MATH_BATCHES, 1):
        try:
            r = await replay_math(batch, i)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  math batch {i} failed: {e}", file=sys.stderr)

    if not rows:
        raise SystemExit("nothing produced")

    out_df = pd.DataFrame(rows)
    out_path = REPO / "artifacts" / "ideas_v2.tsv"
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"\nwrote {len(rows)} rows → {out_path}")

    # Side-by-side report
    print("\n" + "=" * 72)
    print(" v1 vs v2 composite (critic_iter 1 → 3)")
    print("=" * 72)
    for _, r in out_df.iterrows():
        cid = int(r["cluster_a"])
        topic = "math" if cid < 0 else "ai  "
        print(f"  [{topic}] cluster={cid:>4}  composite={float(r['panel_composite']):.2f}  "
              f"α={float(r['panel_agreement']):.2f}  "
              f"critic={r['critic_final_verdict']}/{float(r['critic_final_score']):.1f}  "
              f"iters={int(r['n_critic_iterations'])}  "
              f"{str(r['title'])[:60]}")


if __name__ == "__main__":
    asyncio.run(main())
