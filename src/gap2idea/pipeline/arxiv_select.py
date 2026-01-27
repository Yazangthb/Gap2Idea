from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalize_text(value: str) -> str:
    return str(value).replace("\n", " ").strip()


def _extract_year(versions) -> int | None:
    try:
        if not versions:
            return None
        created = versions[-1].get("created")
        if not created:
            return None
        year = pd.to_datetime(created, errors="coerce").year
        if pd.isna(year):
            return None
        return int(year)
    except Exception:
        return None


def _iter_metadata(metadata_path: str | Path) -> Iterable[dict]:
    path = Path(metadata_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def select_arxiv_subset(
    metadata_path: str | Path,
    output_path: str | Path,
    target_categories: list[str] | set[str] | None = None,
    min_year: int | None = None,
    n_papers: int | None = 250,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Filter the arXiv JSONL metadata into a smaller TSV subset.

    Returns a dataframe with columns: id, title, abstract, year, categories.
    """
    target_set = set(target_categories) if target_categories else None
    rng = random.Random(random_state)

    selected: list[dict] = []
    seen = 0

    for item in _iter_metadata(metadata_path):
        categories = str(item.get("categories", "")).strip()
        categories_list = categories.split()

        if target_set and not any(c in target_set for c in categories_list):
            continue

        year = _extract_year(item.get("versions", []))
        if min_year is not None and (year is None or year < min_year):
            continue

        record = {
            "id": str(item.get("id", "")).strip(),
            "title": _normalize_text(item.get("title", "")),
            "abstract": _normalize_text(item.get("abstract", "")),
            "year": year,
            "categories": categories,
        }
        if not record["id"]:
            continue

        if n_papers is None:
            selected.append(record)
            continue

        seen += 1
        if len(selected) < n_papers:
            selected.append(record)
            continue

        j = rng.randint(0, seen - 1)
        if j < n_papers:
            selected[j] = record

    df = pd.DataFrame(selected)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    return df
