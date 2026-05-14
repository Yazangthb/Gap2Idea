"""Find Limitations / Future Work / Conclusion sections in extracted paper text.

Three strategies, tried in order per paper:
  1. **Structured headings** — numbered (`4 Limitations`), Roman (`IV. Future Work`),
     or ALL-CAPS section headers. Cut at References/Bibliography.
  2. **Window fallback** — first hit of any target keyword anywhere in the doc,
     take a fixed-length token window after it.
  3. **Tail fallback** — last 900 words of the (pre-references) text; future
     work statements almost always live in the conclusion.

Output is a DataFrame with one row per (paper, section_type) pair. We keep at
most two rows per paper to bound downstream OpenAI cost.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from gap2idea.utils import get_logger

log = get_logger(__name__)

REF_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+|[IVXLC]+\.\s+)?"
    r"(?P<title>[A-Z][A-Za-z0-9 &\-/,:]{2,80})\s*$",
    re.MULTILINE,
)
LIMITATION_HEAD_RE = re.compile(r"\b(limitations?|threats to validity|caveats?)\b", re.IGNORECASE)
FUTURE_HEAD_RE = re.compile(r"\b(future work|future directions?|outlook|next steps?)\b", re.IGNORECASE)
DISCUSSION_HEAD_RE = re.compile(r"\b(discussion|conclusion(s)?)\b", re.IGNORECASE)

FALLBACK_KEYWORDS_RE = re.compile(
    r"\b(future work|future directions?|limitations?|threats to validity|"
    r"open (?:problems?|questions?)|conclusion(?:s)?\s*(?:and|&)\s*future)\b",
    re.IGNORECASE,
)

MIN_BODY_CHARS = 200
WINDOW_WORDS = 900
MAX_SECTIONS_PER_PAPER = 2


def _cut_before_references(text: str) -> str:
    m = REF_RE.search(text)
    return text[: m.start()] if m else text


def _structured_sections(text: str) -> list[dict]:
    """Walk every plausible heading and pull the body between it and the next
    heading. Keep only sections whose heading matches a target keyword."""
    headings = list(HEADING_RE.finditer(text))
    found: list[dict] = []
    for i, m in enumerate(headings):
        title = m.group("title").strip()
        nxt = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[m.end() : nxt].strip()
        if len(body) < MIN_BODY_CHARS:
            continue
        if LIMITATION_HEAD_RE.search(title):
            found.append({"section_type": "limitations", "heading": title, "section_text": body})
        elif FUTURE_HEAD_RE.search(title):
            found.append({"section_type": "future_work", "heading": title, "section_text": body})
        elif DISCUSSION_HEAD_RE.search(title):
            found.append({"section_type": "discussion", "heading": title, "section_text": body})
    # de-dup on (section_type, first 100 chars)
    seen = set()
    out = []
    for s in found:
        key = (s["section_type"], s["section_text"][:100])
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def _window_fallback(text: str) -> dict | None:
    m = FALLBACK_KEYWORDS_RE.search(text)
    if not m:
        return None
    after = text[m.start() :]
    words = after.split()
    window = " ".join(words[:WINDOW_WORDS])
    if len(window) < MIN_BODY_CHARS:
        return None
    return {"section_type": "fallback", "heading": "fallback_window", "section_text": window}


def _tail_fallback(text: str) -> dict | None:
    words = text.split()
    if len(words) < 300:
        return None
    tail = " ".join(words[-WINDOW_WORDS:])
    return {"section_type": "tail", "heading": "tail_window", "section_text": tail}


def extract_sections_for_paper(paper_id: str, full_text: str) -> list[dict]:
    pre_ref = _cut_before_references(full_text)
    sections = _structured_sections(pre_ref)

    if not sections:
        win = _window_fallback(pre_ref)
        if win:
            sections.append(win)

    if not sections:
        tail = _tail_fallback(pre_ref)
        if tail:
            sections.append(tail)

    # Prefer limitations & future_work; cap at MAX_SECTIONS_PER_PAPER
    priority = {"limitations": 0, "future_work": 1, "discussion": 2, "fallback": 3, "tail": 4}
    sections.sort(key=lambda s: priority.get(s["section_type"], 9))
    sections = sections[:MAX_SECTIONS_PER_PAPER]
    for s in sections:
        s["id"] = paper_id
    return sections


def extract_all_sections(texts_jsonl: Path, out_jsonl: Path) -> pd.DataFrame:
    # dtype=False so arxiv IDs that look like floats stay as strings.
    texts = pd.read_json(texts_jsonl, lines=True, dtype=False)
    texts["id"] = texts["id"].astype(str)
    log.info("Splitting sections for %d papers", len(texts))
    all_rows: list[dict] = []
    for _, r in texts.iterrows():
        rows = extract_sections_for_paper(str(r["id"]), str(r["text"]))
        all_rows.extend(rows)
    df = pd.DataFrame(all_rows)
    if df.empty:
        log.warning("No sections found across corpus!")
        df = pd.DataFrame(columns=["id", "section_type", "heading", "section_text"])
    else:
        log.info(
            "Found %d sections (%s)",
            len(df),
            ", ".join(f"{k}:{v}" for k, v in df["section_type"].value_counts().items()),
        )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    log.info("Wrote %s", out_jsonl)
    return df
