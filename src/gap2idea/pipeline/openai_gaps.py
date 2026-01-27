from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


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

SYSTEM = "Extract up to 2 research gaps from given text. Use verbatim sentences and paragraphs from input only."


def build_user_prompt(paper_id: str, text: str) -> str:
    return (
        f"id={paper_id}\n"
        "Return JSON per schema. items<=2. "
        "type in {limitation,future_work,open_problem}. "
        "gap_sentence: one verbatim sentence stating missing/next work. "
        "paragraph_text: verbatim paragraph containing it. "
        "Skip contributions/results.\n"
        f"TEXT:\n{text}"
    )


def paper_text_from_sec_df(g: pd.DataFrame, max_chars: int = 9000) -> str:
    order = {"limitations": 0, "future_work": 1, "fallback": 2}
    gg = g.copy()
    gg["ord"] = gg["section_type"].map(lambda x: order.get(x, 9))
    gg = gg.sort_values("ord")
    chunks = []
    for _, r in gg.iterrows():
        txt = str(r["section_text"]).strip()
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks)[:max_chars]


def extract_gaps_from_sections(
    sections_jsonl: str | Path,
    out_tsv: str | Path,
    model: str = "gpt-4.1-mini",
    max_papers: int | None = 50,
    flush_every: int = 25,
) -> pd.DataFrame:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment (.env).")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    sec_df = pd.read_json(sections_jsonl, lines=True)

    out_tsv = Path(out_tsv)
    out_tsv.parent.mkdir(parents=True, exist_ok=True)

    if out_tsv.exists():
        existing = pd.read_csv(out_tsv, sep="\t")
        done_ids = set(existing["id"].astype(str).unique())
    else:
        existing = pd.DataFrame()
        done_ids = set()

    paper_ids = sec_df["id"].astype(str).unique().tolist()
    if max_papers is not None:
        paper_ids = paper_ids[:max_papers]
    paper_ids = [pid for pid in paper_ids if pid not in done_ids]

    rows = []

    def flush_rows(buf):
        if not buf:
            return
        df_new = pd.DataFrame(buf)
        header = not out_tsv.exists()
        df_new.to_csv(
            out_tsv,
            sep="\t",
            index=False,
            mode="a",
            header=header,
            encoding="utf-8",
            escapechar="\\",
        )
        buf.clear()

    for paper_id in tqdm(paper_ids, total=len(paper_ids)):
        g = sec_df[sec_df["id"].astype(str) == paper_id]
        text = paper_text_from_sec_df(g)
        if not text or len(text.strip()) < 200:
            continue

        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": build_user_prompt(paper_id, text)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "gap_extraction",
                        "schema": GAP_SCHEMA,
                        "strict": True,
                    }
                },
            )

            data = json.loads(resp.output_text)
            items = data.get("items", [])
            for it in items:
                rows.append(
                    {
                        "id": paper_id,
                        "gap_type": it["type"],
                        "gap_sentence": it["gap_sentence"],
                        "paragraph_text": it["paragraph_text"],
                        "confidence": it["confidence"],
                    }
                )

            if len(rows) >= flush_every:
                flush_rows(rows)
        except Exception:
            continue

    flush_rows(rows)
    if out_tsv.exists():
        return pd.read_csv(out_tsv, sep="\t")
    return existing
