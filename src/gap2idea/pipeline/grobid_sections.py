"""Clean section structure via a running GROBID service (fallback: None -> PyMuPDF).

GROBID's font/ML-trained parser recovers the real section tree (headings +
text) that PyMuPDF's font-role heuristic misses on scrambled/2-column PDFs.
We use it to feed Stage A authoritative sections so it can *blacklist*
Introduction/Related-Work/Background regions (the source of the background-FP
error) instead of guessing from inline keywords.

Requires a GROBID service (default http://localhost:8070):
    docker run --rm -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.1

Every function degrades gracefully: if the service is down or a PDF fails,
`extract_sections` returns None and the caller falls back to PyMuPDF.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

from gap2idea.utils import get_logger

log = get_logger(__name__)

TEI = "{http://www.tei-c.org/ns/1.0}"
GROBID_URL = os.getenv("GROBID_URL", "http://localhost:8070")


def grobid_available(timeout: float = 3.0) -> bool:
    try:
        return requests.get(f"{GROBID_URL}/api/isalive", timeout=timeout).text.strip() == "true"
    except Exception:
        return False


def _div_text(div) -> str:
    return " ".join("".join(p.itertext()).strip() for p in div.findall(f"{TEI}p")).strip()


def extract_sections(pdf_path: str | Path, timeout: float = 180.0) -> list[dict] | None:
    """Return ordered [{'n', 'heading', 'text'}] for a PDF, or None on failure.

    None signals the caller to fall back to the PyMuPDF path.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None
    try:
        with pdf_path.open("rb") as f:
            r = requests.post(
                f"{GROBID_URL}/api/processFulltextDocument",
                files={"input": f},
                data={"consolidateHeader": "0", "consolidateCitations": "0", "segmentSentences": "0"},
                timeout=timeout,
            )
        r.raise_for_status()
    except Exception as e:  # noqa: BLE001
        log.warning("GROBID failed for %s: %s", pdf_path.name, e)
        return None

    try:
        root = ET.fromstring(r.text.encode("utf-8"))
    except ET.ParseError as e:
        log.warning("GROBID returned unparseable TEI for %s: %s", pdf_path.name, e)
        return None

    body = root.find(f".//{TEI}text/{TEI}body")
    if body is None:
        return None
    out: list[dict] = []
    for div in body.findall(f"{TEI}div"):
        head = div.find(f"{TEI}head")
        if head is None:
            continue
        heading = "".join(head.itertext()).strip()
        if not heading:
            continue
        out.append({"n": (head.get("n") or "").strip(), "heading": heading, "text": _div_text(div)})
    return out or None


def _grobid_text(sections: list[dict]) -> str:
    """Faithful full-paper text reconstructed from GROBID sections (clean, ordered)."""
    return "\n\n".join(
        (f"{s['heading']}\n{s['text']}" if s.get("text") else s["heading"]) for s in sections
    ).strip()


def extract_all_grobid(
    pdfs_dir,
    out_jsonl,
    max_workers: int = 4,
    fallback: bool = True,
    timeout: float = 180.0,
):
    """Ingest every PDF in ``pdfs_dir`` -> paper_texts.jsonl with GROBID's clean
    text + section tree. Per-paper PyMuPDF fallback when GROBID is down or fails.

    Output schema is a superset of ``pdf_text.extract_all``:
        id, text, n_chars, source ("grobid" | "pymupdf")
        + sections (GROBID rows)  OR  blocks, n_headings (fallback rows)
    so the funnel picks up ``sections`` when present and everything else reads
    the clean ``text``.
    """
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from gap2idea.pipeline import pdf_text as PT

    pdfs_dir, out_jsonl = Path(pdfs_dir), Path(out_jsonl)
    pdfs = sorted(pdfs_dir.glob("*.pdf"))
    up = grobid_available()
    log.info("GROBID ingest: %d PDFs (grobid_up=%s, fallback=%s)", len(pdfs), up, fallback)
    if not up and not fallback:
        raise RuntimeError(f"GROBID not reachable at {GROBID_URL} and fallback disabled")

    def _work(p: Path) -> dict:
        secs = extract_sections(p, timeout=timeout) if up else None
        if secs:
            text = _grobid_text(secs)
            return {"id": p.stem, "text": text, "n_chars": len(text),
                    "sections": secs, "source": "grobid"}
        if fallback:
            blocks = PT.extract_pdf_blocks(p)
            text = PT.blocks_to_text(blocks) if blocks else PT.extract_pdf_text(p)
            return {"id": p.stem, "text": text, "n_chars": len(text), "blocks": blocks,
                    "n_headings": sum(1 for b in blocks if b["role"] == "heading"),
                    "source": "pymupdf"}
        return {"id": p.stem, "text": "", "n_chars": 0, "source": "grobid_failed"}

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_work, p) for p in pdfs]
        for fut in as_completed(futures):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    df["id"] = df["id"].astype(str)
    df = df.sort_values("id").reset_index(drop=True)
    before = len(df)
    df = df[df["n_chars"] >= PT.MIN_TEXT_CHARS].reset_index(drop=True)
    n_grobid = int((df["source"] == "grobid").sum()) if "source" in df.columns else 0
    log.info("Kept %d/%d papers (%d via GROBID, %d via PyMuPDF fallback)",
             len(df), before, n_grobid, len(df) - n_grobid)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_jsonl, orient="records", lines=True, force_ascii=False)
    log.info("Wrote %s", out_jsonl)
    return df
