"""Multi-criteria idea evaluation for thesis-grade reporting.

Each generated idea is scored on four 1-5 Likert axes by an LLM judge:

  - novelty            : how distinct is this from prior work?
  - specificity        : are the method + evaluation concrete?
  - feasibility        : could a graduate student actually run this?
  - evidence_grounding : does the idea trace to the provided gap evidence?

We also compute a quantitative "evidence_overlap" — the fraction of the
LLM-listed `evidence_used` entries whose (paper_id, gap_sentence) pair
matches a row that was actually fed into the prompt. This catches hallucinated
evidence even when the judge gives a 5.

The judge model is configurable (default: gpt-4.1-mini). Use a *different*
model from the generator if you want to mitigate self-evaluation bias for
the final thesis numbers — pass --judge-model gpt-4o or similar.
"""
from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path

import pandas as pd
from openai import OpenAI

from gap2idea.pipeline.llm import DEFAULT_JUDGE_MODEL, get_llm_client, parse_json_response
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)

JUDGE_MODEL = DEFAULT_JUDGE_MODEL

# Default panel for cross-provider consensus. Override at the call site.
DEFAULT_JUDGE_PANEL: list[str] = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "google/gemini-2.5-flash",
]

RUBRIC = """\
You are a critical research reviewer. Score the proposed research idea on four
1-5 axes (1 = poor, 5 = excellent). Be strict. Most reasonable ideas should
land in the 2-4 range; reserve 5 for exceptional work and 1 for clearly broken.

Axes:
  1. novelty            : Does this go beyond a straightforward combination of
                          existing methods? Penalise rebranding.
  2. specificity        : Are the method, dataset, metric, and baseline named
                          concretely? Penalise hand-waving.
  3. feasibility        : Could a graduate student start in <2 weeks with
                          standard resources? Penalise unverifiable claims.
  4. evidence_grounding : Does the idea trace back to the listed evidence
                          gaps? Penalise drift away from the evidence.

Also produce a one-sentence `rationale` per axis and a final 2-3 sentence
`overall_critique` that names the single biggest improvement.
"""

EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "novelty": {"type": "integer", "minimum": 1, "maximum": 5},
        "novelty_rationale": {"type": "string"},
        "specificity": {"type": "integer", "minimum": 1, "maximum": 5},
        "specificity_rationale": {"type": "string"},
        "feasibility": {"type": "integer", "minimum": 1, "maximum": 5},
        "feasibility_rationale": {"type": "string"},
        "evidence_grounding": {"type": "integer", "minimum": 1, "maximum": 5},
        "evidence_grounding_rationale": {"type": "string"},
        "overall_critique": {"type": "string"},
    },
    "required": [
        "novelty", "novelty_rationale",
        "specificity", "specificity_rationale",
        "feasibility", "feasibility_rationale",
        "evidence_grounding", "evidence_grounding_rationale",
        "overall_critique",
    ],
    "additionalProperties": False,
}


def _evidence_overlap(idea_record: dict) -> float:
    """Fraction of evidence_used (paper_id, gap_sentence) pairs that match a
    row actually fed in evidence_a/evidence_b. 1.0 = perfectly grounded."""
    fed = set()
    for side in ("evidence_a", "evidence_b"):
        for ev in idea_record.get(side, []) or []:
            fed.add((str(ev.get("paper_id", "")), str(ev.get("gap_sentence", "")).strip()))
    used = idea_record.get("idea", {}).get("evidence_used", []) or []
    if not used:
        return 0.0
    hits = 0
    for ev in used:
        key = (str(ev.get("paper_id", "")), str(ev.get("gap_sentence", "")).strip())
        if key in fed:
            hits += 1
    return hits / len(used)


@retry(tries=3, base_delay=2.0)
def _call_judge(client: OpenAI, idea: dict, model: str) -> dict:
    # We describe the schema inline so non-OpenAI providers (which may ignore
    # `response_format=json_schema`) still produce the right fields.
    user = (
        RUBRIC
        + "\n\nReturn ONLY a JSON object with EXACTLY these keys (no extra prose):\n"
        + json.dumps(
            {
                "novelty": "<int 1-5>",
                "novelty_rationale": "<one sentence>",
                "specificity": "<int 1-5>",
                "specificity_rationale": "<one sentence>",
                "feasibility": "<int 1-5>",
                "feasibility_rationale": "<one sentence>",
                "evidence_grounding": "<int 1-5>",
                "evidence_grounding_rationale": "<one sentence>",
                "overall_critique": "<2-3 sentences>",
            },
            indent=2,
        )
        + "\n\nIdea under review:\n"
        + json.dumps(idea, ensure_ascii=False)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict research peer reviewer. Always return JSON."},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "idea_eval", "schema": EVAL_SCHEMA, "strict": True},
        },
        temperature=0.0,
    )
    return parse_json_response(resp.choices[0].message.content)


def _normalise_judge_scores(scores: dict) -> dict:
    """Coerce a judge response into all expected fields, with safe defaults.

    Non-OpenAI providers occasionally drop optional-looking rationales or
    rename keys. We accept partial responses rather than failing the whole
    evaluation batch.
    """
    def _int(x, default=0):
        try:
            return int(round(float(x)))
        except (TypeError, ValueError):
            return default

    axes = ["novelty", "specificity", "feasibility", "evidence_grounding"]
    out = {}
    for axis in axes:
        out[axis] = _int(scores.get(axis), 0)
        # Accept several rationale key spellings
        rationale = (
            scores.get(f"{axis}_rationale")
            or scores.get(f"{axis}_reason")
            or scores.get(f"{axis}_explanation")
            or ""
        )
        out[f"{axis}_rationale"] = str(rationale)
    out["overall_critique"] = str(scores.get("overall_critique") or scores.get("critique") or "")
    return out


def evaluate_ideas(
    ideas_jsonl: Path,
    out_tsv: Path,
    judge_model: str = JUDGE_MODEL,
) -> pd.DataFrame:
    """Score every idea in `ideas_jsonl` (the rich record file written by
    `generate_ideas_batch`). Returns a flat DataFrame, also written to TSV.
    """
    client = get_llm_client()
    rows: list[dict] = []
    with open(ideas_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info("Evaluating %d ideas with judge=%s", len(records), judge_model)

    for i, rec in enumerate(records, 1):
        idea = rec.get("idea", {})
        try:
            raw_scores = _call_judge(client, idea, model=judge_model)
        except Exception as e:
            log.error("Judge failed on idea %d: %s", i, e)
            continue
        scores = _normalise_judge_scores(raw_scores)
        overlap = _evidence_overlap(rec)
        composite = (
            scores["novelty"] + scores["specificity"]
            + scores["feasibility"] + scores["evidence_grounding"]
        ) / 4.0
        rows.append(
            {
                "title": idea.get("title", ""),
                "cluster_a": rec.get("pair", {}).get("cluster_a"),
                "cluster_b": rec.get("pair", {}).get("cluster_b"),
                "novelty": scores["novelty"],
                "novelty_rationale": scores["novelty_rationale"],
                "specificity": scores["specificity"],
                "specificity_rationale": scores["specificity_rationale"],
                "feasibility": scores["feasibility"],
                "feasibility_rationale": scores["feasibility_rationale"],
                "evidence_grounding": scores["evidence_grounding"],
                "evidence_grounding_rationale": scores["evidence_grounding_rationale"],
                "overall_critique": scores["overall_critique"],
                "composite": composite,
                "evidence_overlap": overlap,
                "s2_novelty_score": (rec.get("novelty") or {}).get("novelty_score"),
                "s2_max_similarity": (rec.get("novelty") or {}).get("max_similarity"),
                "judge_model": judge_model,
            }
        )
        log.info(
            "  [%d/%d] %s  nov=%d spec=%d feas=%d grnd=%d  ev_overlap=%.2f",
            i, len(records), idea.get("title", "")[:60],
            scores["novelty"], scores["specificity"],
            scores["feasibility"], scores["evidence_grounding"], overlap,
        )

    df = pd.DataFrame(rows)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote evaluation to %s", out_tsv)
    return df


def write_report(eval_df: pd.DataFrame, ideas_df: pd.DataFrame, out_md: Path) -> None:
    """Produce a markdown summary suitable for the thesis appendix."""
    if eval_df.empty:
        out_md.write_text("# Evaluation Report\n\n(No ideas evaluated.)\n", encoding="utf-8")
        return

    lines = ["# Gap2Idea — Evaluation Report", ""]
    lines.append(f"- Ideas evaluated: **{len(eval_df)}**")
    for axis in ["novelty", "specificity", "feasibility", "evidence_grounding", "composite"]:
        if axis in eval_df.columns:
            lines.append(f"- Mean **{axis}**: {eval_df[axis].astype(float).mean():.2f}")
    if "evidence_overlap" in eval_df.columns:
        lines.append(f"- Mean **evidence_overlap** (post-hoc grounding check): {eval_df['evidence_overlap'].astype(float).mean():.2f}")
    if "s2_novelty_score" in eval_df.columns and eval_df["s2_novelty_score"].notna().any():
        s = pd.to_numeric(eval_df["s2_novelty_score"], errors="coerce").dropna()
        lines.append(f"- Mean **S2 novelty_score** (1 - max similarity to prior): {s.mean():.2f}")
    lines.append("")
    lines.append("## Top 5 ideas by composite score")
    lines.append("")
    top = eval_df.sort_values("composite", ascending=False).head(5)
    for _, r in top.iterrows():
        lines.append(f"### {r['title']}")
        lines.append(
            f"- composite={r['composite']:.2f} "
            f"(novelty={r['novelty']}, specificity={r['specificity']}, "
            f"feasibility={r['feasibility']}, grounding={r['evidence_grounding']})"
        )
        lines.append(f"- evidence_overlap={r['evidence_overlap']:.2f}, "
                     f"S2 novelty={r.get('s2_novelty_score', 'n/a')}")
        lines.append(f"> {r['overall_critique']}")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote report %s", out_md)


# ===========================================================================
# Phase 3: Multi-judge panel
# ===========================================================================

def _agreement(scores_per_axis: dict[str, list[int]]) -> float:
    """Mean inter-judge agreement across axes: 1 - (mean of per-axis std / 4).
    Returns a number in [0, 1]; 1 = perfect agreement, 0 = max disagreement.
    """
    stds = []
    for axis, vs in scores_per_axis.items():
        if len(vs) >= 2:
            stds.append(statistics.pstdev(vs))
    if not stds:
        return 1.0
    return max(0.0, 1.0 - (sum(stds) / len(stds)) / 4.0)


def _aggregate_panel(panel_results: list[tuple[str, dict]]) -> dict:
    """Aggregate per-judge scores into a consensus row.

    panel_results: [(judge_model, normalised_scores), ...]
    """
    axes = ["novelty", "specificity", "feasibility", "evidence_grounding"]
    per_axis: dict[str, list[int]] = {a: [] for a in axes}
    for _, s in panel_results:
        for a in axes:
            try:
                per_axis[a].append(int(s.get(a, 0)))
            except (TypeError, ValueError):
                pass

    consensus: dict[str, float] = {}
    for a in axes:
        consensus[a] = statistics.mean(per_axis[a]) if per_axis[a] else 0.0
    consensus["composite"] = statistics.mean(consensus[a] for a in axes)
    consensus["agreement"] = _agreement(per_axis)
    consensus["n_judges"] = len(panel_results)

    # Pull a representative critique (longest)
    critiques = [(s.get("overall_critique", ""), m) for m, s in panel_results]
    critiques.sort(key=lambda x: -len(x[0]))
    consensus["consensus_critique"] = critiques[0][0] if critiques else ""

    return consensus


def evaluate_ideas_panel(
    ideas_jsonl: Path,
    out_tsv: Path,
    judge_models: list[str] | None = None,
) -> pd.DataFrame:
    """Score every idea with a panel of N judges (different providers).

    For each idea x judge we run `_call_judge` -> `_normalise_judge_scores`.
    The output TSV has consensus columns + per-judge columns (judge_N_*).
    """
    judge_models = judge_models or DEFAULT_JUDGE_PANEL
    log.info("Evaluating with %d-judge panel: %s", len(judge_models), judge_models)

    client = get_llm_client()
    with open(ideas_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    log.info("  %d ideas to score", len(records))

    rows: list[dict] = []
    for i, rec in enumerate(records, 1):
        idea = rec.get("idea", {})
        panel_results: list[tuple[str, dict]] = []
        for judge_model in judge_models:
            try:
                raw = _call_judge(client, idea, model=judge_model)
                scores = _normalise_judge_scores(raw)
                panel_results.append((judge_model, scores))
            except Exception as e:
                log.error("Judge %s failed on idea %d: %s", judge_model, i, e)
                continue

        if not panel_results:
            log.error("All judges failed on idea %d, skipping", i)
            continue

        consensus = _aggregate_panel(panel_results)
        overlap = _evidence_overlap(rec)

        row = {
            "title": idea.get("title", ""),
            "cluster_a": rec.get("pair", {}).get("cluster_a"),
            "cluster_b": rec.get("pair", {}).get("cluster_b"),
            "mode": rec.get("mode", "bridge"),
            "novelty": consensus["novelty"],
            "specificity": consensus["specificity"],
            "feasibility": consensus["feasibility"],
            "evidence_grounding": consensus["evidence_grounding"],
            "composite": consensus["composite"],
            "agreement": consensus["agreement"],
            "n_judges": consensus["n_judges"],
            "consensus_critique": consensus["consensus_critique"],
            "evidence_overlap": overlap,
            "s2_novelty_score": (rec.get("novelty") or {}).get("novelty_score"),
        }
        # Per-judge columns
        for judge_model, scores in panel_results:
            slug = judge_model.replace("/", "__")
            for axis in ["novelty", "specificity", "feasibility", "evidence_grounding"]:
                row[f"{slug}__{axis}"] = scores.get(axis)

        rows.append(row)
        log.info(
            "  [%d/%d] %s  consensus=%.2f  agreement=%.2f  judges=%d",
            i, len(records), idea.get("title", "")[:50],
            consensus["composite"], consensus["agreement"], consensus["n_judges"],
        )

    df = pd.DataFrame(rows)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote panel evaluation to %s", out_tsv)
    return df
