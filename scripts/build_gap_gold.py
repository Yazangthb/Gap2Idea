"""Build a CLEAN gap-sentence gold for benchmarking the funnel.

Input  : a full paper.
Output : the VERBATIM sentences that are research gaps (authors' OWN limitation
         or a future-work direction), each with a type. Nothing else — this is a
         pure recall target for Stage A and a classification target for Stage B.

Tighter than build_gold_dataset.py: the prompt explicitly excludes the
contamination the audit surfaced — limitations of PRIOR work, contributions /
results / method descriptions mislabelled as future work, and vague gestures.

We keep a gold sentence only if its words are actually present in the source
(token_recall >= 0.80) so every target is genuinely locatable despite PDF
reading-order scrambling.

Outputs:
    data/bench_gap/gold_sentences.tsv   (paper_id, gap_id, gap_type, token_recall, gap_sentence)

Usage:
    python scripts/build_gap_gold.py --model openai/gpt-4o
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gap2idea.pipeline.llm import get_llm_client, parse_json_response  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402
from gap2idea.utils import get_logger, retry  # noqa: E402

log = get_logger(__name__)
MODEL_DEFAULT = "openai/gpt-4o"
MIN_TOKEN_RECALL = 0.80
_WS, _WORD = re.compile(r"\s+"), re.compile(r"[a-z0-9]+")


def norm(s: str) -> str:
    return _WS.sub(" ", s or "").strip()


def provenance_recall(sent: str, body_tokens: set[str]) -> float:
    gt = _WORD.findall(norm(sent).lower())
    return sum(1 for t in gt if t in body_tokens) / max(1, len(gt))


def _loads_tolerant(line: str) -> dict | None:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        repaired = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", line)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return None


def load_texts(manifest: pd.DataFrame) -> dict[str, str]:
    targets = {str(r["id"]) for _, r in manifest.iterrows()}
    out: dict[str, str] = {}
    for src in {str(r["source"]) for _, r in manifest.iterrows()}:
        for line in Path(src).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = _loads_tolerant(line)
            if rec and str(rec.get("id")) in targets:
                out[str(rec.get("id"))] = str(rec.get("text", ""))
    return out

SYSTEM_PROMPT = (
    "You extract RESEARCH GAP sentences from a full scientific paper. A gap sentence is a "
    "VERBATIM sentence in which the AUTHORS state either:\n"
    "  (a) a LIMITATION of THEIR OWN work — something their method/study does not do, an "
    "assumption it relies on, data/scope it is restricted to, or a weakness they acknowledge; or\n"
    "  (b) a FUTURE-WORK direction — something they plan to do, leave for future work, or say "
    "remains to be done.\n"
    "Extract EVERY such sentence — favour completeness (high recall).\n"
    "Do NOT extract:\n"
    "  - limitations or open problems of PRIOR work / the field in general (intro or related-work "
    "critique that the paper itself then addresses);\n"
    "  - contributions, results, or method descriptions (e.g. 'We show via Theorem 3.3 that ...', "
    "'We propose ...', 'Our method achieves ...');\n"
    "  - vague gestures with no concrete content ('more work is needed').\n"
    "Copy each sentence VERBATIM from the paper. If a gap spans two sentences, return each "
    "separately."
)
USER_TEMPLATE = "paper_id={paper_id}\nReturn JSON per schema.\n\nPAPER:\n{text}"

SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["paper_id", "gaps"],
    "properties": {
        "paper_id": {"type": "string"},
        "gaps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["gap_type", "gap_sentence", "paragraph_context"],
            "properties": {
                "gap_type": {"enum": ["limitation", "future_work"]},
                "gap_sentence": {"type": "string"},
                "paragraph_context": {"type": "string"},
            }}},
    },
}

# --- Verification pass (high-precision filter; standard extract->filter pattern) ---
VERIFY_SYSTEM = (
    "You verify candidate research-gap sentences. For EACH candidate decide is_valid: true ONLY if the "
    "sentence, read in its context, is an explicit statement of the AUTHORS' OWN work's LIMITATION or a "
    "FUTURE-WORK direction. Set is_valid=false for: cross-references ('discussed in § IV'), contributions "
    "or results ('we show', 'we bring forth bounds', 'we propose'), method/formula descriptions, "
    "definitions, and limitations of PRIOR work or the field stated as motivation. If valid but mistyped, "
    "give corrected_type."
)
VERIFY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["verdicts"],
    "properties": {"verdicts": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["idx", "is_valid", "corrected_type"],
        "properties": {"idx": {"type": "integer"}, "is_valid": {"type": "boolean"},
                       "corrected_type": {"enum": ["limitation", "future_work", "none"]}}}}},
}


@retry(tries=2, base_delay=3.0)
def _verify(client, model: str, cands: list[dict]) -> dict:
    payload = "\n".join(
        f"[{i}] type={c['gap_type']} | sentence: {c['gap_sentence']} | context: {c['paragraph_context'][:400]}"
        for i, c in enumerate(cands))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": VERIFY_SYSTEM},
                  {"role": "user", "content": "Verify each candidate.\n" + payload}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "verify", "schema": VERIFY_SCHEMA, "strict": True}},
        temperature=0.0, timeout=180, max_tokens=2000,
    )
    return parse_json_response(resp.choices[0].message.content)


@retry(tries=2, base_delay=3.0)
def _call(client, model: str, pid: str, text: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": USER_TEMPLATE.format(paper_id=pid, text=text)}],
        response_format={"type": "json_schema",
                         "json_schema": {"name": "gap_sentences", "schema": SCHEMA, "strict": True}},
        temperature=0.0, timeout=240, max_tokens=4000,
    )
    return parse_json_response(resp.choices[0].message.content)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("data/bench_gold/papers_manifest.tsv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/bench_gap"))
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--no-verify", action="store_true", help="skip the precision-filter pass")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    texts = load_texts(manifest)
    log.info("Loaded text for %d/%d papers", len(texts), len(manifest))

    client = get_llm_client()
    rows = []
    for _, r in manifest.iterrows():
        pid = str(r["id"])
        body = _cut_before_references(texts.get(pid, ""))
        if len(body) < 2000:
            log.warning("  %s too short — skip", pid)
            continue
        body_tokens = set(_WORD.findall(norm(body).lower()))
        try:
            rec = _call(client, args.model, pid, body)
            gaps = rec.get("gaps", [])
        except Exception as e:
            log.error("  %s FAILED: %s", pid, e)
            gaps = []

        # locatable candidates only
        cands = [g for g in gaps
                 if provenance_recall(str(g.get("gap_sentence") or ""), body_tokens) >= MIN_TOKEN_RECALL]

        # verification pass: drop contributions / cross-refs / prior-work limits
        valid = {i: str(c["gap_type"]) for i, c in enumerate(cands)}  # default keep
        if cands and not args.no_verify:
            try:
                vd = _verify(client, args.model, cands).get("verdicts", [])
                valid = {v["idx"]: (v["corrected_type"] if v["corrected_type"] != "none" else cands[v["idx"]]["gap_type"])
                         for v in vd if v.get("is_valid") and 0 <= v["idx"] < len(cands)}
            except Exception as e:
                log.error("  %s verify FAILED (keeping all): %s", pid, e)

        kept = 0
        for i, c in enumerate(cands):
            if i not in valid:
                continue
            sent = str(c["gap_sentence"])
            kept += 1
            rows.append({"paper_id": pid, "gap_id": f"{pid}::g{kept}",
                         "gap_type": valid[i],
                         "token_recall": round(provenance_recall(sent, body_tokens), 3),
                         "gap_sentence": sent})
        log.info("  %-12s kept=%d  (extracted=%d, locatable=%d)", pid, kept, len(gaps), len(cands))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    out = args.out_dir / "gold_sentences.tsv"
    df.to_csv(out, sep="\t", index=False)
    log.info("Wrote %s (%d gold gap sentences, %d papers)", out, len(df), df["paper_id"].nunique())
    if not df.empty:
        print("\ngap_type:\n" + df["gap_type"].value_counts().to_string())
        print("\nper-paper:\n" + df.groupby("paper_id").size().to_string())


if __name__ == "__main__":
    main()
