"""Audit semantic precision: for each kept prediction, ask gpt-4o whether the
sentence (with context) describes a real research gap. Counts TP semantically,
not via token overlap.

This reveals TRUE precision — uncoupled from PDF scramble damage to token-match.
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

from gap2idea.pipeline.llm import get_llm_client  # noqa: E402

SYS_AUDIT = (
    "For each numbered input, decide whether the highlighted sentence "
    "(between >>> <<<) — read together with its context — describes a real "
    "RESEARCH GAP. A research gap means the authors mention something NOT YET "
    "DONE in their OWN work: a limitation, weakness, assumption, scope "
    "restriction, future-work direction, or open problem.\n\n"
    "BE LENIENT WITH SCRAMBLED SENTENCES: If the sentence is fragmented or has "
    "interleaved text (PDF parsing artifact), but the semantic content of the "
    "fragment + context still describes a gap, reply YES.\n\n"
    "Reply NO only if the sentence (with context) is clearly NOT a gap (e.g. "
    "a contribution claim, result, method description, citation, gratitude, etc.).\n\n"
    "Output ONLY '<idx>. YES' or '<idx>. NO' per line, nothing else."
)


def find_context(sent, text, window=40):
    s = sent.strip()[:80]
    idx = text.find(s)
    if idx < 0:
        return "", ""
    end = idx + len(sent)
    before = " ".join(text[max(0, idx-500):idx].split()[-window:])
    after = " ".join(text[end:end+500].split()[:window])
    return before, after


def call_batched(client, model, system, items, batch=10):
    preds = [False] * len(items)
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        lines = []
        for k, (sent, b, a, _pid) in enumerate(chunk):
            ctx = f"{b} >>> {sent} <<< {a}".strip()
            lines.append(f"{k+1}. {ctx}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": "Judge:\n" + "\n".join(lines)}],
                temperature=0.0, max_tokens=10*len(chunk))
            content = resp.choices[0].message.content
            for ln in content.splitlines():
                m = re.match(r"\s*(\d+)\s*[.):\-]\s*(YES|NO)", ln.strip(), re.IGNORECASE)
                if m:
                    k = int(m.group(1)) - 1
                    if 0 <= k < len(chunk):
                        preds[i+k] = m.group(2).upper() == "YES"
        except Exception as e:
            print(f"  batch err: {e}", flush=True)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--input", default="data/scibert_prep/scibert_gold_gaps_v2slice.tsv")
    args = ap.parse_args()

    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec
    preds_df = pd.read_csv(ROOT / args.input, sep="\t", dtype=str).fillna("")
    print(f"Input predictions: {len(preds_df)}", flush=True)

    # Build with context
    items = []
    for _, row in preds_df.iterrows():
        pid = row["paper_id"]; sent = row["gap_sentence"]
        rec = papers.get(pid, {})
        b, a = find_context(sent, rec.get("text", "")) if rec else ("", "")
        items.append((sent, b, a, pid))

    client = get_llm_client()

    # === Pass 1: strict GAP/JUNK ===
    from test_strict_precision import call_batched as strict_call, SYS_STRICT, SYS_CONFIRM
    print(f"\n=== PASS 1: STRICT GAP/JUNK ===", flush=True)
    t0 = time.time()
    pass1 = strict_call(client, args.model, SYS_STRICT, items, label_yes="GAP", label_no="JUNK")
    p1_items = [it for it, k in zip(items, pass1) if k]
    print(f"  done in {time.time()-t0:.1f}s. kept {len(p1_items)}/{len(items)}", flush=True)

    # === Pass 2: confirmation ===
    print(f"\n=== PASS 2: confirmation ===", flush=True)
    t0 = time.time()
    pass2 = strict_call(client, args.model, SYS_CONFIRM, p1_items, label_yes="YES", label_no="NO")
    p2_items = [it for it, k in zip(p1_items, pass2) if k]
    print(f"  done in {time.time()-t0:.1f}s. confirmed {len(p2_items)}/{len(p1_items)}", flush=True)

    # === Pass 3: semantic audit (with scramble-leniency) ===
    print(f"\n=== PASS 3: SEMANTIC AUDIT (scramble-lenient) ===", flush=True)
    t0 = time.time()
    audit = call_batched(client, args.model, SYS_AUDIT, p2_items)
    n_real_gaps = sum(audit)
    print(f"  done in {time.time()-t0:.1f}s. semantic gaps: {n_real_gaps}/{len(p2_items)}", flush=True)

    semantic_precision = n_real_gaps / max(1, len(p2_items))
    print(f"\n=== TRUE PRECISION (semantic) ===")
    print(f"  Predictions kept: {len(p2_items)}")
    print(f"  Real gaps (semantic):    {n_real_gaps}")
    print(f"  TRUE precision:          {semantic_precision:.3f}")

    # Show false positives
    fps = [(it, ok) for it, ok in zip(p2_items, audit) if not ok]
    if fps:
        print(f"\nFalse positives (gpt-4o says NOT_GAP):")
        for it, _ in fps[:10]:
            print(f"  - {it[0][:120]}")


if __name__ == "__main__":
    main()
