"""Extract research gaps from paper sections via LLM structured outputs.

Methodology (defensible for a thesis):
- We feed only Limitations / Future Work / Discussion sections (extracted by
  `sections.py`), never the full paper. This anchors the LLM on language
  authors use to describe their own gaps, rather than asking it to infer.
- We require the model to return *verbatim* sentences and paragraphs. This
  gives every downstream "gap" a concrete textual provenance.
- We bound output to <=2 items per paper to avoid the model fabricating
  filler gaps when a section has none.
- We use strict JSON-schema response format so we never need to post-parse
  free text.

The LLM is reached through OpenRouter (see `gap2idea.pipeline.llm`), so
swapping providers is a one-line change to the `model=` argument.

The output `gaps.tsv` is the entry point to `theme-mine`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
from openai import OpenAI

from gap2idea.pipeline.llm import DEFAULT_MODEL, get_llm_client, parse_json_response
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)

MODEL = DEFAULT_MODEL
MAX_INPUT_CHARS = 9000  # cap to stay well under context for cheaper model
MIN_GAP_LEN = 20
MIN_CONFIDENCE = 0.5

SYSTEM_PROMPT = (
    "You are a meticulous research-gap annotator. "
    "Extract up to 2 research gaps from the given text. "
    "Use VERBATIM sentences and paragraphs from the input only. "
    "Skip statements about contributions, results, or finished work. "
    "Skip vague gestures ('more work is needed' alone) unless paired with a concrete direction."
)

USER_TEMPLATE = (
    "id={paper_id}\n"
    "Return JSON per schema. items<=2.\n"
    "  - type in {{limitation, future_work, open_problem}}\n"
    "  - gap_sentence: one VERBATIM sentence stating a missing capability,\n"
    "    open question, or next-step direction.\n"
    "  - paragraph_text: the VERBATIM paragraph containing that sentence.\n"
    "  - confidence: 0..1; only emit items with conf >= 0.5.\n"
    "Skip everything else.\n\n"
    "TEXT:\n{text}"
)

GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_id": {"type": "string"},
        "items": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["limitation", "future_work", "open_problem"]},
                    "gap_sentence": {"type": "string"},
                    "paragraph_text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["type", "gap_sentence", "paragraph_text", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["paper_id", "items"],
    "additionalProperties": False,
}


def _section_text_for_paper(sections_df: pd.DataFrame, paper_id: str) -> str:
    """Concatenate the paper's sections, prioritising limitations/future_work."""
    priority = {"limitations": 0, "future_work": 1, "discussion": 2, "fallback": 3, "tail": 4}
    # Cast both sides to str: arxiv IDs like "2106.05969" get auto-parsed as
    # floats by pd.read_json, breaking equality lookups.
    sub = sections_df[sections_df["id"].astype(str) == str(paper_id)].copy()
    if sub.empty:
        return ""
    sub["__p"] = sub["section_type"].map(priority).fillna(9)
    sub = sub.sort_values("__p")
    joined = "\n\n".join(sub["section_text"].astype(str).tolist())
    return joined[:MAX_INPUT_CHARS]


@retry(tries=4, base_delay=2.0)
def _call_openai(client: OpenAI, paper_id: str, text: str) -> dict:
    """One structured call. Retries on transient errors."""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(paper_id=paper_id, text=text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "gap_extraction", "schema": GAP_SCHEMA, "strict": True},
        },
        temperature=0.0,
    )
    return parse_json_response(resp.choices[0].message.content)


def _flatten(record: dict, paper_id: str) -> list[dict]:
    rows = []
    for item in record.get("items", []):
        rows.append(
            {
                "id": paper_id,
                "gap_type": item["type"],
                "gap_sentence": item["gap_sentence"].replace("\n", " ").strip(),
                "paragraph_text": item["paragraph_text"].replace("\n", " ").strip(),
                "confidence": float(item["confidence"]),
            }
        )
    return rows


def extract_gaps(
    sections_jsonl: Path,
    out_tsv: Path,
    model: str = MODEL,
    resume: bool = True,
    sleep_between: float = 0.0,
) -> pd.DataFrame:
    """Extract gaps for every paper present in `sections_jsonl`.

    `resume=True` skips papers already present in `out_tsv` so reruns are cheap.
    """
    # dtype=False so arxiv IDs like "2106.05969" don't get parsed as floats.
    sections_df = pd.read_json(sections_jsonl, lines=True, dtype=False)
    sections_df["id"] = sections_df["id"].astype(str)
    paper_ids = sorted(sections_df["id"].unique())

    done_ids: set[str] = set()
    if resume and out_tsv.exists():
        try:
            done = pd.read_csv(out_tsv, sep="\t")
            done_ids = set(done["id"].astype(str).tolist())
            log.info("Resuming: %d papers already extracted", len(done_ids))
        except Exception:
            pass

    todo = [pid for pid in paper_ids if pid not in done_ids]
    log.info("Calling OpenAI (%s) for %d papers", model, len(todo))

    client = get_llm_client()
    new_rows: list[dict] = []
    for i, pid in enumerate(todo, 1):
        text = _section_text_for_paper(sections_df, pid)
        if len(text) < 200:
            log.info("  [%d/%d] %s: section text too short, skipping", i, len(todo), pid)
            continue
        try:
            rec = _call_openai(client, pid, text)
            rows = _flatten(rec, pid)
            new_rows.extend(rows)
            log.info("  [%d/%d] %s: %d gaps", i, len(todo), pid, len(rows))
        except Exception as e:
            log.error("  [%d/%d] %s FAILED: %s", i, len(todo), pid, e)
        if sleep_between:
            time.sleep(sleep_between)

    new_df = pd.DataFrame(new_rows)
    if out_tsv.exists() and resume:
        existing = pd.read_csv(out_tsv, sep="\t")
        df = pd.concat([existing, new_df], ignore_index=True)
    else:
        df = new_df

    # Post-process: dedupe + min thresholds
    df = df[df["gap_sentence"].str.len() >= MIN_GAP_LEN]
    df = df[df["confidence"] >= MIN_CONFIDENCE]
    df = df.drop_duplicates(subset=["id", "gap_sentence"]).reset_index(drop=True)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote %d gaps to %s", len(df), out_tsv)
    return df
