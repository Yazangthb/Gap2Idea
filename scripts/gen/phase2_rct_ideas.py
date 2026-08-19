"""Phase-2 prototype (limitations -> ideas) on the RCT/SAL limitation corpus.

A vertical slice of the ideation pipeline on clean, human-annotated data:
  1. normalise      -- use the corpus's own coarse taxonomy as canonical limitations
                       (at scale this is produced by embed+cluster; here it is gold)
  3. canonicalise   -- group the 952 limitation sentences by coarse type
  5. score          -- rank by BREADTH (distinct papers affected), not raw frequency
                       (frequency is near-anti-signal: the commonest limits are generic)
  6/7. generate     -- grounded RAG: per canonical limitation, one meta-research /
                       trial-methodology direction, grounded in real evidence spans

NOT covered (data can't support): entity linking, addresses-edge mining from citation
context, cross-domain transfer (single domain), retrospective validation (no temporal
citation graph). This validates the generation loop, not the graph or the scale.

    python scripts/gen/phase2_rct_ideas.py --top 6
"""
from __future__ import annotations
import argparse, ast, csv, io, sys, urllib.request, collections, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gap2idea.pipeline.llm import get_llm_client  # noqa: E402
import os  # noqa: E402

RAW = "https://raw.githubusercontent.com/MengfeiLan/SAL_Type_Classification/main/data/"


def parse_cats(s):
    s = (s or "").strip()
    if not s or s == "[0]":
        return []
    try:
        v = ast.literal_eval(s)
        return [str(x).strip() for x in (v if isinstance(v, list) else [v]) if str(x).strip()]
    except Exception:
        return [c.strip(" []'\"") for c in s.replace(";", ",").split(",") if c.strip(" []'\"")]


def load_limitations():
    rows = []
    for sp in ["train", "dev", "test"]:
        raw = urllib.request.urlopen(RAW + sp + ".csv", timeout=60).read().decode("utf-8", "replace")
        rows += list(csv.DictReader(io.StringIO(raw)))
    lim = []
    for r in rows:
        if (r.get("category_spans", "[0]") or "[0]").strip() == "[0]":
            continue
        cats = parse_cats(r.get("coarse_grained_categories"))
        if cats:
            lim.append({"pmcid": r["pmcids"], "sent": r["sentences"].strip(), "cats": cats})
    return lim


SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["recurring_gap", "why_open", "proposed_direction", "contribution_type",
                 "feasibility", "resolves"],
    "properties": {
        "recurring_gap": {"type": "string"},
        "why_open": {"type": "string"},
        "proposed_direction": {"type": "string"},
        "contribution_type": {"type": "string", "enum": ["methods", "meta-research", "tooling", "guideline", "benchmark"]},
        "feasibility": {"type": "string", "enum": ["high", "medium", "low"]},
        "resolves": {"type": "string"},
    },
}
SYS = (
    "You are a clinical-trial methodology researcher. You are given a limitation type that "
    "recurs across MANY randomized controlled trials, with real example sentences from "
    "different trials. Propose ONE concrete, publishable methodological or meta-research "
    "contribution that would address this CLASS of limitation across FUTURE trials — not "
    "fix a single trial, and not the trivial answer ('run a bigger/longer trial'). Ground "
    "every claim in the evidence. Prefer transferable methods, standards, tooling, or "
    "benchmarks. Return JSON per the schema."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--model", default="yandexgpt-5-pro")
    ap.add_argument("--out", default=str(ROOT / "data" / "phase2_rct_ideas.md"))
    args = ap.parse_args()

    lim = load_limitations()
    by_cat = collections.defaultdict(list)
    for r in lim:
        for c in r["cats"]:
            by_cat[c].append(r)
    # breadth = distinct papers; rank by that, not raw sentence count
    ranked = sorted(by_cat.items(), key=lambda kv: len({r["pmcid"] for r in kv[1]}), reverse=True)
    print(f"{len(lim)} limitation sentences across {len({r['pmcid'] for r in lim})} papers, "
          f"{len(by_cat)} canonical types\n")

    cli = get_llm_client()
    model = "gpt://%s/%s/latest" % (os.getenv("YANDEX_FOLDER_ID"), args.model)
    out = ["# Phase-2 prototype — RCT limitations -> methodological ideas\n",
           "_Canonical limitation = corpus coarse taxonomy; ranked by breadth (distinct papers); "
           "one grounded idea each via yandexgpt-5-pro._\n"]

    for cat, items in ranked[:args.top]:
        papers = len({r["pmcid"] for r in items})
        evidence = [r["sent"] for r in items][:8]
        user = (f"Limitation type: {cat}\nAffects {papers} distinct trials ({len(items)} sentences).\n"
                "Example limitation sentences from different trials:\n"
                + "\n".join(f"- {e[:220]}" for e in evidence))
        resp = cli.chat.completions.create(
            model=model, temperature=0.3, max_tokens=1200,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            response_format={"type": "json_schema", "json_schema": {"name": "idea", "schema": SCHEMA, "strict": True}})
        idea = json.loads(resp.choices[0].message.content)
        print(f"== {cat}  ({papers} papers, {len(items)} sents) ==")
        print(f"  gap:        {idea['recurring_gap']}")
        print(f"  direction:  {idea['proposed_direction'][:200]}")
        print(f"  type/feas:  {idea['contribution_type']} / {idea['feasibility']}\n")
        out += [f"\n## {cat}  ·  {papers} trials, {len(items)} limitation sentences\n",
                f"- **Recurring gap:** {idea['recurring_gap']}",
                f"- **Why still open:** {idea['why_open']}",
                f"- **Proposed direction:** {idea['proposed_direction']}",
                f"- **Contribution type:** {idea['contribution_type']}  ·  **Feasibility:** {idea['feasibility']}",
                f"- **Resolves:** {idea['resolves']}",
                "- **Grounded in:** " + " / ".join(f'"{e[:80]}…"' for e in evidence[:3])]
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
