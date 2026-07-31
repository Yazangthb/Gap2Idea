"""Re-run sanity + paper-drafter on the 3 AI ideas only.

The first pass short-circuited because the sanity gate reads
idea["confidence"] but the flat row stores it as idea_confidence.
gen_paper_drafts.py now mirrors the field — this driver invokes that
patched flow for just the AI rows so the math drafts (which are already
fine) aren't redone.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from gen_paper_drafts import drive_one, is_math


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    df = pd.read_csv(REPO / "artifacts" / "ideas_v2.tsv", sep="\t")
    out_dir = REPO / "artifacts" / "paper_drafts"
    out_dir.mkdir(parents=True, exist_ok=True)

    ai_rows = [r for _, r in df.iterrows() if not is_math(r.to_dict())]
    print(f"re-drafting {len(ai_rows)} AI ideas with sanity stage...")
    for row in ai_rows:
        await drive_one(row.to_dict(), out_dir)


if __name__ == "__main__":
    asyncio.run(main())
