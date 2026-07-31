"""Build a gold gap dataset by passing the FULL paper to an LLM.

Unlike the current pipeline (regex section -> LLM on the section), we send the
whole paper. This lets the model judge whether each gap is actually left OPEN or
resolved by the paper itself (the discourse-level property we found is essential).

Schema (minimal, per user choice) — per gap:
    gap_type           limitation | future_work | open_problem
    gap_sentence       VERBATIM core sentence
    paragraph_context  VERBATIM surrounding paragraph
    resolution_status  open | partially_addressed | resolved_in_paper

Derived deterministically (not trusted to the LLM):
    gap_id, verbatim_ok, location_fraction, domain

Outputs:
    data/bench_gold/gaps_full.jsonl       one record per paper
    data/bench_gold/gaps_full_flat.tsv    one row per gap (review-friendly)

Usage:
    python scripts/dataset/build_gold_dataset.py --model openai/gpt-4o
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.llm import get_llm_client, parse_json_response  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402
from gap2idea.utils import get_logger, retry  # noqa: E402

log = get_logger(__name__)

MODEL_DEFAULT = "openai/gpt-4o"
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def _provenance(gap_sentence: str, body_norm: str, body_tokens: set[str]) -> tuple[str, float, float]:
    """Return (provenance, token_recall, location_fraction).

    Source PDF text often has reading-order corruption (interleaved columns), so a
    faithful sentence may not appear as a contiguous substring. We therefore grade:
      exact  contiguous substring of source        (true verbatim)
      fuzzy  >=90% of words present in source       (faithful, de-scrambled)
      weak   <90% words present                     (possible paraphrase — review)
    """
    sent_norm = _norm(gap_sentence)
    off = body_norm.find(sent_norm) if sent_norm else -1
    gt = _WORD.findall(sent_norm)
    tr = sum(1 for t in gt if t in body_tokens) / max(1, len(gt))
    if off >= 0:
        prov = "exact"
        loc = round(off / len(body_norm), 4) if body_norm else -1.0
    else:
        prov = "fuzzy" if tr >= 0.90 else "weak"
        loc = -1.0
    return prov, round(tr, 3), loc


SYSTEM_PROMPT = (
    "You extract research GAPS from a full scientific paper. A gap is a sentence where "
    "the authors state (a) a LIMITATION of their own work, (b) a FUTURE-WORK direction, or "
    "(c) an OPEN PROBLEM.\n"
    "CRITICAL: you are given the WHOLE paper. For each gap, decide whether the paper ITSELF "
    "resolves or answers it later, and set resolution_status:\n"
    "  - open                 the paper leaves it unresolved\n"
    "  - partially_addressed  the paper makes partial progress but does not fully close it\n"
    "  - resolved_in_paper    the paper poses it rhetorically then answers/closes it (e.g. a\n"
    "                         'difficulty' or question used to set up the paper's own solution)\n"
    "Use VERBATIM text from the paper for gap_sentence and paragraph_context. "
    "Extract ALL genuine gaps (no fixed limit). "
    "Skip vague gestures with no concrete content, and skip general field/background questions "
    "that are not the authors' own gap."
)

USER_TEMPLATE = (
    "paper_id={paper_id}\n"
    "Return JSON per the schema. For each gap give gap_type, the VERBATIM gap_sentence, the "
    "VERBATIM paragraph_context containing it, and resolution_status.\n\n"
    "PAPER:\n{text}"
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["paper_id", "gaps"],
    "properties": {
        "paper_id": {"type": "string"},
        "gaps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["gap_type", "gap_sentence", "paragraph_context", "resolution_status"],
                "properties": {
                    "gap_type": {"enum": ["limitation", "future_work", "open_problem"]},
                    "gap_sentence": {"type": "string"},
                    "paragraph_context": {"type": "string"},
                    "resolution_status": {
                        "enum": ["open", "partially_addressed", "resolved_in_paper"]
                    },
                },
            },
        },
    },
}


@retry(tries=2, base_delay=3.0)
def _call(client, model: str, paper_id: str, text: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(paper_id=paper_id, text=text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "gold_gaps", "schema": SCHEMA, "strict": True},
        },
        temperature=0.0,
        timeout=240,
        max_tokens=6000,
    )
    return parse_json_response(resp.choices[0].message.content)


def _loads_tolerant(line: str) -> dict | None:
    """Parse a JSONL line; some paper_texts.jsonl rows contain raw LaTeX with
    unescaped backslash escapes (e.g. '\\underbrace' -> invalid \\u). Try a
    conservative repair (double any backslash not starting a valid JSON escape)
    and skip if still unparseable."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", line)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def load_texts(manifest: pd.DataFrame) -> dict[str, str]:
    targets = {str(r["id"]) for _, r in manifest.iterrows()}
    sources = {str(r["source"]) for _, r in manifest.iterrows()}
    texts: dict[str, str] = {}
    for src in sources:
        bad = 0
        for line in Path(src).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = _loads_tolerant(line)
            if rec is None:
                bad += 1
                continue
            rid = str(rec.get("id"))
            if rid in targets:
                texts[rid] = str(rec.get("text", ""))
        if bad:
            log.warning("  %s: skipped %d unparseable line(s)", src, bad)
    missing = targets - set(texts)
    if missing:
        log.warning("Target papers with NO loadable text: %s", sorted(missing))
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/bench_gold/papers_manifest.tsv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/bench_gold"))
    ap.add_argument("--model", default=MODEL_DEFAULT)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    texts = load_texts(manifest)
    log.info("Loaded full text for %d/%d papers", len(texts), len(manifest))

    client = get_llm_client()
    per_paper: list[dict] = []
    flat: list[dict] = []

    for _, r in manifest.iterrows():
        pid, dom = str(r["id"]), str(r["domain"])
        raw = texts.get(pid, "")
        body = _cut_before_references(raw)          # strip references
        body_norm = _norm(body)
        if len(body) < 2000:
            log.warning("  %s: text too short (%d) — skipping", pid, len(body))
            continue
        try:
            rec = _call(client, args.model, pid, body)
            gaps = rec.get("gaps", [])
        except Exception as e:
            log.error("  %s FAILED: %s", pid, e)
            gaps = []

        body_tokens = set(_WORD.findall(body_norm))
        clean_gaps = []
        for i, g in enumerate(gaps):
            sent = str(g.get("gap_sentence") or "")
            prov, tr, loc_frac = _provenance(sent, body_norm, body_tokens)
            row = {
                "gap_id": f"{pid}::g{i+1}",
                "paper_id": pid,
                "domain": dom,
                "gap_type": g.get("gap_type", ""),
                "resolution_status": g.get("resolution_status", ""),
                "provenance": prov,
                "token_recall": tr,
                "location_fraction": loc_frac,
                "gap_sentence": sent,
                "paragraph_context": str(g.get("paragraph_context") or ""),
            }
            clean_gaps.append(row)
            flat.append(row)
        per_paper.append({"paper_id": pid, "domain": dom, "n_gaps": len(clean_gaps), "gaps": clean_gaps})
        rs = pd.Series([x["resolution_status"] for x in clean_gaps]).value_counts().to_dict()
        prov_c = pd.Series([x["provenance"] for x in clean_gaps]).value_counts().to_dict()
        log.info("  %-12s %-5s gaps=%-2d  resolution=%s  provenance=%s",
                 pid, dom, len(clean_gaps), rs, prov_c)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.out_dir / "gaps_full.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in per_paper:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    flat_df = pd.DataFrame(flat)
    flat_tsv = args.out_dir / "gaps_full_flat.tsv"
    flat_df.to_csv(flat_tsv, sep="\t", index=False)
    log.info("Wrote %s (%d papers) and %s (%d gaps)", jsonl, len(per_paper), flat_tsv, len(flat))

    if not flat_df.empty:
        print("\n=== Summary ===")
        print(f"papers: {len(per_paper)}   gaps: {len(flat_df)}   "
              f"mean gaps/paper: {len(flat_df)/max(1,len(per_paper)):.1f}")
        print("\ngap_type:\n" + flat_df["gap_type"].value_counts().to_string())
        print("\nresolution_status:\n" + flat_df["resolution_status"].value_counts().to_string())
        print("\nprovenance:\n" + flat_df["provenance"].value_counts().to_string())
        print(f"token_recall mean: {flat_df['token_recall'].mean():.3f} "
              f"(words from source even if reading-order-scrambled)")
        openf = flat_df[flat_df["location_fraction"] >= 0]["location_fraction"]
        if len(openf):
            print(f"location_fraction: median={openf.median():.2f} "
                  f"(0=front, 1=end) — terminal-section prior check")


if __name__ == "__main__":
    main()
