from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import fitz  # pymupdf


MAX_TAIL_PAGES = 6
MIN_SECTION_CHARS = 200
TARGET_SECTION_COUNT = 2
FALLBACK_MAX_WORDS = 900

REF_RE = re.compile(r"^\s*(references|bibliography)\s*$", re.IGNORECASE)
HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+|[IVXLC]+\.\s+)?([A-Z][A-Za-z0-9 &\-/,:]{2,80})\s*$"
)
LIMITATION_HEAD_RE = re.compile(r"\b(limitations?|threats to validity|caveats?)\b", re.IGNORECASE)
FUTURE_HEAD_RE = re.compile(r"\b(future work|future directions?|outlook|next steps?)\b", re.IGNORECASE)
FALLBACK_START_RE = re.compile(
    r"(conclusion(s)?\s*(and|&)\s*future work|future work|future directions|limitations?|threats to validity|discussion|conclusion(s)?)",
    re.IGNORECASE,
)


def extract_tail_text_from_pdf(pdf_path: str | Path, tail_pages: int = MAX_TAIL_PAGES) -> str:
    doc = fitz.open(str(pdf_path))
    n = len(doc)
    start = max(0, n - tail_pages)
    chunks = []
    for i in range(start, n):
        chunks.append(doc[i].get_text("text"))
    doc.close()
    return "\n".join(chunks)


def cut_before_references(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if REF_RE.match(line.strip()):
            return "\n".join(lines[:i])
    return text


def find_headings(lines: list[str]) -> list[tuple[int, str]]:
    headings = []
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if not m:
            continue
        title = m.group(1)
        if not title:
            continue
        title = title.strip()
        if len(title) < 3:
            continue
        if sum(c.islower() for c in title) > sum(c.isupper() for c in title) and " " in title:
            if title.lower() not in {"conclusion", "conclusions", "discussion", "future work", "limitations"}:
                continue
        headings.append((i, title))
    return headings


def section_spans_from_headings(lines: list[str], headings: list[tuple[int, str]]) -> list[tuple[str, int, int]]:
    spans = []
    for k, (idx, title) in enumerate(headings):
        start = idx + 1
        end = headings[k + 1][0] if k + 1 < len(headings) else len(lines)
        spans.append((title, start, end))
    return spans


def extract_target_sections_from_end(text: str, target_count: int = TARGET_SECTION_COUNT) -> list[dict]:
    text = cut_before_references(text)
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    headings = find_headings(lines)
    if not headings:
        return []

    spans = section_spans_from_headings(lines, headings)
    found: list[dict] = []
    for title, start, end in reversed(spans):
        body = "\n".join(lines[start:end]).strip()
        if len(body) < MIN_SECTION_CHARS:
            continue
        if LIMITATION_HEAD_RE.search(title):
            found.append({"section_type": "limitations", "heading": title, "text": body})
        elif FUTURE_HEAD_RE.search(title):
            found.append({"section_type": "future_work", "heading": title, "text": body})
        if len(found) >= target_count:
            break

    return list(reversed(found))


def fallback_extract_window(text: str, max_words: int = FALLBACK_MAX_WORDS) -> str | None:
    t = cut_before_references(text)
    m = FALLBACK_START_RE.search(t)
    if not m:
        return None
    window = t[m.start():].strip()
    words = window.split()
    if len(words) > max_words:
        window = " ".join(words[:max_words])
    return window.strip()


def extract_limitations_futurework(pdf_path: str | Path) -> dict:
    tail = extract_tail_text_from_pdf(pdf_path, tail_pages=MAX_TAIL_PAGES)
    sections = extract_target_sections_from_end(tail, target_count=TARGET_SECTION_COUNT)
    if sections:
        return {"sections": sections, "fallback_text": ""}

    fb = fallback_extract_window(tail, max_words=FALLBACK_MAX_WORDS)
    return {"sections": [], "fallback_text": fb or ""}


def extract_sections_from_pdfs(
    pdf_dir: str | Path,
    output_jsonl: str | Path,
    ids: list[str] | None = None,
) -> pd.DataFrame:
    pdf_dir = Path(pdf_dir)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pdf_paths = {
        p.stem.replace("_", "/"): p
        for p in pdf_dir.glob("*.pdf")
        if p.is_file() and p.stat().st_size > 0
    }
    if ids is not None:
        ids_set = set(str(i) for i in ids)
        pdf_paths = {pid: p for pid, p in pdf_paths.items() if pid in ids_set}

    rows = []
    for arxiv_id, pdf_path in tqdm(pdf_paths.items(), total=len(pdf_paths)):
        out = extract_limitations_futurework(pdf_path)
        for s in out["sections"]:
            rows.append(
                {
                    "id": arxiv_id,
                    "section_type": s["section_type"],
                    "heading": s["heading"],
                    "section_text": s["text"],
                }
            )
        if not out["sections"] and out["fallback_text"]:
            rows.append(
                {
                    "id": arxiv_id,
                    "section_type": "fallback",
                    "heading": "fallback_window",
                    "section_text": out["fallback_text"],
                }
            )

    sec_df = pd.DataFrame(rows)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in sec_df[["id", "section_type", "heading", "section_text"]].to_dict("records"):
            r["id"] = str(r.get("id", ""))
            r["section_type"] = "" if pd.isna(r.get("section_type")) else str(r.get("section_type"))
            r["heading"] = "" if pd.isna(r.get("heading")) else str(r.get("heading"))
            r["section_text"] = "" if pd.isna(r.get("section_text")) else str(r.get("section_text"))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return sec_df
