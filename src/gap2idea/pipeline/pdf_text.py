from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from tqdm import tqdm
import fitz  # pymupdf


def pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def pdf_filename(arxiv_id: str) -> str:
    return f"{arxiv_id.replace('/', '_')}.pdf"


def txt_filename(arxiv_id: str) -> str:
    return f"{arxiv_id.replace('/', '_')}.txt"


def download_pdf(
    arxiv_id: str,
    pdf_dir: str | Path,
    session: requests.Session,
    timeout: int = 30,
    min_pdf_bytes: int = 10_000,
) -> Path | None:
    pdf_dir = Path(pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    out_path = pdf_dir / pdf_filename(arxiv_id)

    if out_path.exists() and out_path.stat().st_size > min_pdf_bytes:
        return out_path

    try:
        resp = session.get(pdf_url(arxiv_id), timeout=timeout)
        if resp.status_code != 200:
            return None

        out_path.write_bytes(resp.content)
        if out_path.stat().st_size < min_pdf_bytes:
            return None

        return out_path
    except Exception:
        return None


def pdf_to_text(pdf_path: str | Path, max_pages: int = 12) -> str:
    doc = fitz.open(str(pdf_path))
    chunks = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        chunks.append(page.get_text("text"))
    doc.close()
    return "\n".join(chunks)


def extract_and_save_text(
    arxiv_id: str,
    pdf_dir: str | Path,
    txt_dir: str | Path,
    max_pages: int = 12,
    min_text_chars: int = 500,
) -> Path | None:
    txt_dir = Path(txt_dir)
    txt_dir.mkdir(parents=True, exist_ok=True)
    out_txt = txt_dir / txt_filename(arxiv_id)

    if out_txt.exists() and out_txt.stat().st_size > min_text_chars:
        return out_txt

    pdf_path = Path(pdf_dir) / pdf_filename(arxiv_id)
    if not pdf_path.exists():
        return None

    try:
        text = pdf_to_text(pdf_path, max_pages=max_pages)
        if len(text.strip()) < min_text_chars:
            return None

        out_txt.write_text(text, encoding="utf-8")
        return out_txt
    except Exception:
        return None


def load_ids_from_tsv(path: str | Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "id" in df.columns:
        ids = df["id"].astype(str).tolist()
    elif "paper_id" in df.columns:
        ids = df["paper_id"].astype(str).tolist()
    else:
        raise ValueError("Expected column 'id' or 'paper_id' in TSV.")
    return [i for i in ids if i]


def filter_missing_texts(
    arxiv_ids: Iterable[str],
    txt_dir: str | Path,
    min_text_chars: int = 500,
) -> list[str]:
    txt_dir = Path(txt_dir)
    todo = []
    for arxiv_id in arxiv_ids:
        txt_path = txt_dir / txt_filename(arxiv_id)
        if txt_path.exists() and txt_path.stat().st_size > min_text_chars:
            continue
        todo.append(arxiv_id)
    return todo


def download_and_extract_texts(
    arxiv_ids: Iterable[str],
    pdf_dir: str | Path,
    txt_dir: str | Path,
    download_workers: int = 16,
    extract_workers: int = 8,
    max_pages: int = 12,
    min_text_chars: int = 500,
    min_pdf_bytes: int = 10_000,
) -> dict:
    pdf_dir = Path(pdf_dir)
    txt_dir = Path(txt_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    arxiv_ids = list(arxiv_ids)
    todo_ids = filter_missing_texts(arxiv_ids, txt_dir, min_text_chars=min_text_chars)

    downloaded: dict[str, Path] = {}
    session = requests.Session()

    if todo_ids:
        with ThreadPoolExecutor(max_workers=download_workers) as ex:
            futures = {
                ex.submit(download_pdf, arxiv_id, pdf_dir, session, 30, min_pdf_bytes): arxiv_id
                for arxiv_id in todo_ids
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs"):
                arxiv_id = futures[fut]
                pdf_path = fut.result()
                if pdf_path:
                    downloaded[arxiv_id] = pdf_path

    saved_texts: dict[str, Path] = {}
    if downloaded:
        with ThreadPoolExecutor(max_workers=extract_workers) as ex:
            futures = {
                ex.submit(extract_and_save_text, arxiv_id, pdf_dir, txt_dir, max_pages, min_text_chars): arxiv_id
                for arxiv_id in downloaded.keys()
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting & saving texts"):
                arxiv_id = futures[fut]
                txt_path = fut.result()
                if txt_path:
                    saved_texts[arxiv_id] = txt_path

    return {
        "total_ids": len(arxiv_ids),
        "todo_ids": len(todo_ids),
        "downloaded_pdfs": len(downloaded),
        "saved_texts": len(saved_texts),
    }
