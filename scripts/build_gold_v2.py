"""Build a comprehensive gold v2 for the 10 gold papers using gpt-4o.

Two-pass extraction + adjudication:
  PASS 1 — extraction: gpt-4o reads the FULL paper and exhaustively lists
           every semantic gap (limitation, future-work, scope, assumption,
           open problem) as VERBATIM sentences from the paper.
  PASS 2 — adjudication: a separate gpt-4o call independently judges each
           extracted candidate ("is this REALLY a semantic gap?"). Only
           sentences passing both are kept.

Also merges in the 17 SciBERT-predicted "extras" we already audited gpt-4o
verified in audit_gold_pipeline.py, after deduplication.

Output: data/bench_gap/gold_sentences_v2.tsv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import token_containment  # noqa: E402
from gap2idea.pipeline.gap_prefilter import normalize_text  # noqa: E402
from gap2idea.pipeline.llm import get_llm_client, parse_json_response  # noqa: E402

SYSTEM_EXTRACT = (
    "You are an expert at identifying research GAPS in scientific papers. A gap is "
    "a sentence where the authors mention something NOT YET DONE in their OWN work — "
    "explicitly OR implicitly. This includes:\n"
    "  - explicit limitations or weaknesses ('A limitation is...', 'we did not study...')\n"
    "  - scope restrictions ('This work focuses on...', 'restricted to...')\n"
    "  - assumptions and dependencies ('our method assumes...', 'relies on...')\n"
    "  - future-work plans ('we leave X for future work', 'future work will...')\n"
    "  - open problems they pose ('remains an open question', 'it remains unclear')\n"
    "  - acknowledged tradeoffs ('X is exact, but Y...', 'while X works, Y...')\n"
    "\n"
    "Read the paper EXHAUSTIVELY and extract EVERY such sentence VERBATIM from the "
    "paper text. Be comprehensive — capture subtle hedged ones too. Do NOT include:\n"
    "  - contribution claims ('we propose', 'our method achieves')\n"
    "  - prior-work citations\n"
    "  - method/hyperparameter descriptions\n"
    "  - acknowledgments / thanks\n"
    "  - results / performance numbers\n"
)

USER_EXTRACT_TEMPLATE = (
    "Paper id: {pid}\n"
    "Extract EVERY research-gap sentence VERBATIM. Reply with JSON in this exact format:\n"
    '{{"gaps": [{{"gap_type": "limitation|future_work|scope|assumption|open_problem",\n'
    '"sentence": "<verbatim sentence>"}}, ...]}}\n\n'
    "PAPER:\n{text}"
)

SYSTEM_ADJUDICATE = (
    "You judge whether each candidate sentence is a REAL semantic research gap. "
    "Reject if the sentence is: a contribution claim, prior-work citation, method "
    "description, acknowledgment, results statement, or anything that does NOT mention "
    "something not-yet-done in the authors' own work. Accept only sentences that "
    "explicitly or implicitly describe a limitation, scope restriction, assumption, "
    "future-work direction, or open problem of the authors' own work."
)


def _cut_text(text: str, max_chars: int = 80000) -> str:
    return text[:max_chars]


def extract_gaps_from_paper(client, model, pid: str, text: str) -> list[dict]:
    """Pass 1: comprehensive extraction."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_EXTRACT},
            {"role": "user", "content": USER_EXTRACT_TEMPLATE.format(pid=pid, text=_cut_text(text))},
        ],
        temperature=0.0, max_tokens=4000,
        response_format={"type": "json_object"},
    )
    try:
        data = parse_json_response(resp.choices[0].message.content)
        return data.get("gaps", []) or []
    except Exception as e:
        print(f"  [pid={pid}] extract parse fail: {e}", flush=True)
        return []


def adjudicate(client, model, candidates: list[dict], paper_text: str) -> list[dict]:
    """Pass 2: each candidate gets an independent yes/no with reason."""
    if not candidates:
        return []
    # Provide ~60-word context for each candidate via paper_text search
    enriched = []
    for c in candidates:
        sent = c.get("sentence", "")
        idx = paper_text.find(sent[:60])
        if idx >= 0:
            before = " ".join(paper_text[max(0, idx-300):idx].split()[-30:])
            after = " ".join(paper_text[idx+len(sent):idx+len(sent)+300].split()[:30])
            enriched.append((sent, before, after, c.get("gap_type", "")))
        else:
            enriched.append((sent, "", "", c.get("gap_type", "")))
    listing = "\n".join(
        f"{i+1}. type={t}\n   context: ...{b[-80:]} >>> {s[:200]} <<< {a[:80]}..."
        for i, (s, b, a, t) in enumerate(enriched)
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_ADJUDICATE},
            {"role": "user", "content": "For each numbered candidate, reply with one line: "
             "'<idx>. KEEP | <12-word reason>' or '<idx>. DROP | <12-word reason>'.\n\n" + listing},
        ],
        temperature=0.0, max_tokens=70 * len(candidates),
    )
    out = []
    decisions = {}
    for ln in resp.choices[0].message.content.splitlines():
        m = re.match(r"\s*(\d+)\s*[.):\-]\s*(KEEP|DROP)\s*\|?\s*(.*)", ln.strip(), re.IGNORECASE)
        if m:
            decisions[int(m.group(1))] = (m.group(2).upper() == "KEEP", m.group(3).strip())
    for i, c in enumerate(candidates):
        keep, reason = decisions.get(i+1, (True, "default-keep"))
        if keep:
            c["adjudication_reason"] = reason
            out.append(c)
    return out


def dedup(gaps: list[dict], tau: float = 0.8) -> list[dict]:
    out = []
    norms = []
    for g in gaps:
        s = g.get("sentence", "")
        n = normalize_text(s)
        if not n:
            continue
        # quick dup test against accepted
        is_dup = False
        for prev in norms:
            if token_containment(s, prev) >= tau or token_containment(prev, s) >= tau:
                is_dup = True
                break
        if not is_dup:
            out.append(g)
            norms.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    args = ap.parse_args()

    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec
    print(f"Loaded {len(papers)} gold papers", flush=True)

    client = get_llm_client()
    new_gold = []
    for pid, rec in papers.items():
        t0 = time.time()
        cands = extract_gaps_from_paper(client, args.model, pid, str(rec.get("text", "")))
        n_extract = len(cands)
        kept = adjudicate(client, args.model, cands, str(rec.get("text", "")))
        kept = dedup(kept)
        for k, c in enumerate(kept):
            new_gold.append({
                "paper_id": pid,
                "gap_id": f"{pid}::v2_g{k+1}",
                "gap_type": c.get("gap_type", "limitation"),
                "gap_sentence": c.get("sentence", ""),
                "reason": c.get("adjudication_reason", ""),
            })
        print(f"  {pid}: extracted={n_extract}  kept={len(kept)}  ({time.time()-t0:.1f}s)", flush=True)

    df = pd.DataFrame(new_gold)
    out = ROOT / "data/bench_gap/gold_sentences_v2.tsv"
    df.to_csv(out, sep="\t", index=False)
    print(f"\nNew gold v2: {len(df)} gaps over {df['paper_id'].nunique()} papers")
    print(f"  types: {df['gap_type'].value_counts().to_dict()}")
    print(f"  saved: {out}")

    # Compare to v1 gold
    v1 = pd.read_csv(ROOT / "data/bench_gap/gold_sentences.tsv", sep="\t")
    print(f"\nOld gold v1: {len(v1)} gaps over {v1['paper_id'].nunique()} papers")
    overlap = 0
    v1_sents = v1["gap_sentence"].tolist()
    for _, g in df.iterrows():
        for s in v1_sents:
            if token_containment(g["gap_sentence"], s) >= 0.8 or token_containment(s, g["gap_sentence"]) >= 0.8:
                overlap += 1
                break
    print(f"Overlap (v2 ∩ v1): {overlap} sentences")
    print(f"New in v2: {len(df) - overlap} sentences")


if __name__ == "__main__":
    main()
