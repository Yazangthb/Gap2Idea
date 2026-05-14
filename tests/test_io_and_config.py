"""Sanity checks for io and config helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from gap2idea.config import get_paths
from gap2idea.io import read_tsv, write_tsv


def test_get_paths_returns_subdirs():
    p = get_paths(".")
    assert p.data.name == "data"
    assert p.artifacts.name == "artifacts"
    assert p.pdfs.parent == p.data
    assert p.texts.parent == p.data


def test_tsv_round_trip(tmp_path: Path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = tmp_path / "x.tsv"
    write_tsv(df, out)
    back = read_tsv(out)
    assert list(back.columns) == ["a", "b"]
    assert back["b"].tolist() == ["x", "y"]
