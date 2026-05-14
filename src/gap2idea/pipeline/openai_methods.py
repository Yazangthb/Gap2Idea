"""Extract method-claim sentences from papers via LLM structured outputs.

A "method claim" is a verbatim sentence asserting a concrete capability the
paper *introduces, implements, or shows works* — e.g.

  "We propose a graph neural network that handles dynamic edges
   via attention pooling."

Method claims are the dual of gap statements: gaps describe what's missing,
methods describe what's been done. Stage 8 (`--mode method-gap`) bridges
them — for each cluster of gaps, retrieve method claims that are
"close enough to be applicable but not yet applied" and synthesise an
idea of the form "use method M (paper P1) to address gap G (paper P2)."

Method-claim sentences live in **abstracts and introductions**, not in
limitations/future-work sections. We therefore read directly from
`data/paper_texts.jsonl` and take the first ~2500 characters of each paper
(reliably captures abstract + the "contributions" enumeration in the intro).

Output schema matches `openai_gaps` so downstream clustering can reuse the
same theme-mining machinery.
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
MAX_INPUT_CHARS = 2500  # abstract + intro contributions list
MIN_METHOD_LEN = 25
MIN_CONFIDENCE = 0.5

SYSTEM_PROMPT = (
    "You are a meticulous research-method annotator. "
    "Extract up to 3 method-claim sentences from the given text. "
    "Use VERBATIM sentences from the input only. "
    "A method-claim asserts a concrete CAPABILITY the paper introduces, "
    "implements, evaluates, or proves works -- e.g. 'We propose...', "
    "'We develop a framework that...', 'Our algorithm achieves...'. "
    "Skip: descriptions of others' methods (related-work mentions), "
    "high-level motivation, problem statements, and limitations."
)

USER_TEMPLATE = (
    "id={paper_id}\n"
    "Return JSON per schema. items<=3.\n"
    "  - method_type in {{algorithm, framework, dataset, benchmark, metric, theoretical}}\n"
    "  - method_sentence: one VERBATIM sentence asserting an introduced capability.\n"
    "  - paragraph_text: the VERBATIM paragraph containing that sentence.\n"
    "  - confidence: 0..1; only emit items with conf >= 0.5.\n"
    "Skip everything else.\n\n"
    "TEXT:\n{text}"
)

METHOD_SCHEMA = {
    "type": "object",
    "properties": {
        "paper_id": {"type": "string"},
        "items": {
            "type": "array",
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "method_type": {
                        "type": "string",
                        "enum": ["algorithm", "framework", "dataset", "benchmark", "metric", "theoretical"],
                    },
                    "method_sentence": {"type": "string"},
                    "paragraph_text": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["method_type", "method_sentence", "paragraph_text", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["paper_id", "items"],
    "additionalProperties": False,
}


def _paper_text_snippet(text: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Return the first `max_chars` of paper text — captures abstract +
    intro contributions list, where method claims actually live."""
    return (text or "").strip()[:max_chars]


@retry(tries=4, base_delay=2.0)
def _call_openai(client: OpenAI, paper_id: str, text: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(paper_id=paper_id, text=text)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "method_extraction", "schema": METHOD_SCHEMA, "strict": True},
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
                "method_type": item.get("method_type", "algorithm"),
                "method_sentence": item["method_sentence"].replace("\n", " ").strip(),
                "paragraph_text": item["paragraph_text"].replace("\n", " ").strip(),
                "confidence": float(item.get("confidence", 0.0)),
            }
        )
    return rows


def extract_methods(
    texts_jsonl: Path,
    out_tsv: Path,
    model: str = MODEL,
    resume: bool = True,
    sleep_between: float = 0.0,
) -> pd.DataFrame:
    """Extract method claims for every paper in `texts_jsonl`. Resumable."""
    texts = pd.read_json(texts_jsonl, lines=True, dtype=False)
    texts["id"] = texts["id"].astype(str)
    paper_ids = sorted(texts["id"].unique())

    done_ids: set[str] = set()
    if resume and out_tsv.exists():
        try:
            done = pd.read_csv(out_tsv, sep="\t")
            done_ids = set(done["id"].astype(str).tolist())
            log.info("Resuming: %d papers already extracted", len(done_ids))
        except Exception:
            pass

    todo = [pid for pid in paper_ids if pid not in done_ids]
    log.info("Calling LLM (%s) for %d papers' methods", model, len(todo))

    client = get_llm_client()
    new_rows: list[dict] = []
    for i, pid in enumerate(todo, 1):
        row = texts[texts["id"] == pid].iloc[0]
        text = _paper_text_snippet(str(row.get("text", "")))
        if len(text) < 300:
            log.info("  [%d/%d] %s: too short, skipping", i, len(todo), pid)
            continue
        try:
            rec = _call_openai(client, pid, text, model=model)
            rows = _flatten(rec, pid)
            new_rows.extend(rows)
            log.info("  [%d/%d] %s: %d methods", i, len(todo), pid, len(rows))
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

    df = df[df["method_sentence"].str.len() >= MIN_METHOD_LEN]
    df = df[df["confidence"] >= MIN_CONFIDENCE]
    df = df.drop_duplicates(subset=["id", "method_sentence"]).reset_index(drop=True)

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep="\t", index=False)
    log.info("Wrote %d methods to %s", len(df), out_tsv)
    return df
