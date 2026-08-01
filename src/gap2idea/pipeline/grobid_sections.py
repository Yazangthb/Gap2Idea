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
