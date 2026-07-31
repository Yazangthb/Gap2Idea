"""Build a per-sentence labeling sheet for the N=10 bench.

For each paper in data/bench:
  - Take sentences from the UNION of (gold_section_text, predicted section_text).
    Gold is the unarXive author-labeled section; predicted is what sections.py
    actually extracted. The union covers every sentence the current LLM
    extractor could plausibly have picked PLUS every sentence a perfect
    extractor should have considered.
  - Split into sentences with a scientific-text-aware regex.
  - Send each paper's sentences to a stronger LLM (silver model) in one call,
    asking for a typed label per sentence:
        none           descriptive / background / not a gap claim
        limitation     stated shortcoming of the current work
        future_work    stated intention to do X next
        open_problem   stated unresolved question, no commitment

Output TSV columns:
    paper_id, source, sent_idx, sentence,
    silver_label, silver_confidence, silver_rationale,
    gold_label, notes

`gold_label` and `notes` are left empty for the human adjudication pass.

Usage:
    python scripts/archive/build_label_sheet.py
        --bench-dir data/bench
        --out data/bench/label_sheet.tsv
        --model anthropic/claude-sonnet-4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

# Make `gap2idea` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.llm import get_llm_client, parse_json_response
from gap2idea.utils import get_logger, retry

log = get_logger(__name__)

SILVER_MODEL_DEFAULT = "openai/gpt-4o"

# ---------------------------------------------------------------------------
# Sentence splitter (scientific-text-aware)
# ---------------------------------------------------------------------------

# Common abbreviations whose trailing "." would otherwise trigger a false split.
_ABBREVIATIONS = [
    "e.g.", "i.e.", "cf.", "vs.", "etc.", "et al.", "Fig.", "Figs.",
    "Eq.", "Eqs.", "Sec.", "Secs.", "Ref.", "Refs.", "Tab.", "Tabs.",
    "Dr.", "Mr.", "Mrs.", "Prof.", "St.", "No.", "Vol.", "pp.",
    "approx.", "viz.", "resp.", "U.S.", "U.K.",
]
_ABBR_SENTINEL = "\x00"  # NULL byte — won't appear in real text

_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\{(\[\$])")
_WS_RE = re.compile(r"\s+")


def split_sentences(text: str) -> list[str]:
    """Split scientific text into sentences. Returns stripped, non-empty parts."""
    if not text:
        return []
    # Collapse whitespace, then protect abbreviations from splitting.
    norm = _WS_RE.sub(" ", text).strip()
    for abbr in _ABBREVIATIONS:
        norm = norm.replace(abbr, abbr.replace(".", _ABBR_SENTINEL))
    parts = _SPLIT_RE.split(norm)
    out = []
    for p in parts:
        s = p.replace(_ABBR_SENTINEL, ".").strip()
        # Drop trivial fragments (likely splitter noise)
        if len(s) >= 15:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Sentence union from gold + predicted
# ---------------------------------------------------------------------------

def _norm_for_dedupe(s: str) -> str:
    """Normalised form for dedupe: lowercase, collapse whitespace, strip punct edges."""
    s = _WS_RE.sub(" ", s).strip().lower()
    return s.strip(" .;,:!?\"'()[]{}")


def collect_paper_sentences(
    paper_id: str,
    gold_text: str,
    predicted_sections: list[dict],
) -> list[dict]:
    """Return deduped list of {sent_idx, sentence, source} for the paper.

    source ∈ {"gold", "predicted", "both"} so we know where each sentence came
    from. Dedupe is on normalised lowercase text.
    """
    bag: dict[str, dict] = {}  # norm_text -> row
    order: list[str] = []

    for s in split_sentences(gold_text):
        k = _norm_for_dedupe(s)
        if not k:
            continue
        if k not in bag:
            bag[k] = {"sentence": s, "source": "gold"}
            order.append(k)

    for sec in predicted_sections:
        for s in split_sentences(sec.get("section_text") or ""):
            k = _norm_for_dedupe(s)
            if not k:
                continue
            if k in bag:
                if bag[k]["source"] == "gold":
                    bag[k]["source"] = "both"
            else:
                bag[k] = {"sentence": s, "source": "predicted"}
                order.append(k)

    rows = []
    for i, k in enumerate(order):
        rows.append({
            "paper_id": paper_id,
            "sent_idx": i,
            "sentence": bag[k]["sentence"],
            "source": bag[k]["source"],
        })
    return rows


# ---------------------------------------------------------------------------
# Silver-LLM labeling pass
# ---------------------------------------------------------------------------

SILVER_SYSTEM_PROMPT = (
    "You label sentences from the Limitations / Future Work / Discussion section "
    "of a scientific paper. Each sentence gets ONE of four labels:\n"
    "  - none          descriptive, background, motivation, or contribution claim; NOT a gap statement\n"
    "  - limitation    a stated shortcoming of THIS paper's method, scope, dataset, theory, or evaluation\n"
    "  - future_work   a stated intention by the authors to do X, extend Y, or explore Z next\n"
    "  - open_problem  a stated unresolved question or unsolved problem, with NO author commitment to address it\n"
    "Be strict: if a sentence merely describes prior work, results, or context, label `none`. "
    "Citations, references to figures/equations, and methodology recaps are `none`. "
    "A sentence can be a 'limitation' even without the word 'limitation' — what matters is the rhetorical function."
)

SILVER_USER_TEMPLATE = (
    "Paper: {paper_id}\n"
    "Section context: this text comes from the paper's Limitations/Future Work/Discussion area.\n\n"
    "Label every sentence. Return JSON matching the schema; one item per sentence in input order.\n\n"
    "SENTENCES:\n{numbered}"
)

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sent_idx": {"type": "integer"},
                    "label": {
                        "type": "string",
                        "enum": ["none", "limitation", "future_work", "open_problem"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string"},
                },
                "required": ["sent_idx", "label", "confidence", "rationale"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["items"],
    "additionalProperties": False,
}


@retry(tries=2, base_delay=3.0)
def silver_label_paper(client, model: str, paper_id: str, sentences: list[dict]) -> list[dict]:
    """One LLM call per paper. Returns list of {sent_idx, label, confidence, rationale}.

    Returned items use the schema's field names (sent_idx, label, confidence,
    rationale). Non-strict-honoring models (e.g. anthropic via OpenRouter) sometimes
    return a bare array or different keys; we normalise both.
    """
    numbered = "\n".join(f"[{r['sent_idx']}] {r['sentence']}" for r in sentences)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SILVER_SYSTEM_PROMPT},
            {"role": "user", "content": SILVER_USER_TEMPLATE.format(
                paper_id=paper_id, numbered=numbered)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "sentence_labels", "schema": LABEL_SCHEMA, "strict": True},
        },
        temperature=0.0,
        timeout=180,
    )
    raw = resp.choices[0].message.content or ""
    return _normalise_label_items(raw)


# ---------------------------------------------------------------------------
# Tolerant parsing — strip ```fence```, accept bare arrays, normalise key names.
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*([\[{].*?[\]}])\s*```", re.DOTALL)
_KEY_ALIASES = {
    "sentence": "sent_idx",
    "sentence_idx": "sent_idx",
    "idx": "sent_idx",
    "id": "sent_idx",
}


def _normalise_label_items(raw: str) -> list[dict]:
    text = raw.strip()
    # Strip markdown fence if present
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)

    import json as _json
    try:
        obj = _json.loads(text)
    except _json.JSONDecodeError:
        # Last-ditch: use the project's tolerant parser
        try:
            obj = parse_json_response(raw)
        except Exception:
            return []

    if isinstance(obj, dict) and "items" in obj:
        items = obj["items"]
    elif isinstance(obj, list):
        items = obj
    else:
        return []

    normed: list[dict] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        # Remap aliased keys
        row = {_KEY_ALIASES.get(k, k): v for k, v in it.items()}
        if "sent_idx" not in row or "label" not in row:
            continue
        try:
            row["sent_idx"] = int(row["sent_idx"])
        except (TypeError, ValueError):
            continue
        row.setdefault("confidence", "")
        row.setdefault("rationale", "")
        normed.append(row)
    return normed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=Path("data/bench"))
    ap.add_argument("--out", type=Path, default=Path("data/bench/label_sheet.tsv"))
    ap.add_argument("--model", default=SILVER_MODEL_DEFAULT)
    args = ap.parse_args()

    bench_papers_path = args.bench_dir / "bench_papers.jsonl"
    sections_path = args.bench_dir / "sections_extracted.jsonl"

    if not bench_papers_path.exists():
        raise FileNotFoundError(f"Missing {bench_papers_path}")
    if not sections_path.exists():
        raise FileNotFoundError(f"Missing {sections_path}")

    papers = [json.loads(l) for l in bench_papers_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    sections = [json.loads(l) for l in sections_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    sections_by_paper: dict[str, list[dict]] = {}
    for s in sections:
        sections_by_paper.setdefault(str(s["id"]), []).append(s)

    log.info("Building sentence union for %d papers, silver model = %s", len(papers), args.model)
    client = get_llm_client()
    all_rows: list[dict] = []

    for p in papers:
        pid = str(p["id"])
        preds = sections_by_paper.get(pid, [])
        sents = collect_paper_sentences(
            paper_id=pid,
            gold_text=p.get("gold_section_text") or "",
            predicted_sections=preds,
        )
        if not sents:
            log.warning("  %s: no sentences (skipping)", pid)
            continue

        log.info("  %s: %d sentences (gold-titles=%s, predicted-types=%s)",
                 pid, len(sents),
                 p.get("gold_section_titles"),
                 [s["section_type"] for s in preds])

        try:
            silver = silver_label_paper(client, args.model, pid, sents)
        except Exception as e:
            log.error("  %s: silver labeling failed (%s) — leaving labels blank", pid, e)
            silver = []

        by_idx = {item["sent_idx"]: item for item in silver}
        for r in sents:
            lab = by_idx.get(r["sent_idx"], {})
            all_rows.append({
                "paper_id": r["paper_id"],
                "source": r["source"],
                "sent_idx": r["sent_idx"],
                "sentence": r["sentence"],
                "silver_label": lab.get("label", ""),
                "silver_confidence": lab.get("confidence", ""),
                "silver_rationale": lab.get("rationale", ""),
                "gold_label": "",  # for human adjudication
                "notes": "",
            })

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, sep="\t", index=False)
    log.info("Wrote %d rows to %s", len(df), args.out)

    if not df.empty:
        log.info("Silver label distribution:\n%s",
                 df["silver_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
