"""Pipeline v3: Stage A v2 slice → gpt-4o direct GAP/JUNK classification with context.

Replaces SciBERT-FT (Stage B) with gpt-4o batched on the entire slice, using
±30-word context. Eliminates the Stage B recall ceiling.

This is a frozen version added to the registry — does NOT replace v1/v2 path.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    slice_terminal_regions, slice_with_midpaper_anchors,
    token_containment, _looks_like_sentence,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402
from gap2idea.pipeline.llm import get_llm_client  # noqa: E402

MATCH_TAU = 0.70
WINDOW = 30
BATCH = 15  # smaller because each sentence carries context

SYS = (
    "For each numbered input, classify the highlighted sentence (between >>> <<<) "
    "as GAP or JUNK using the surrounding context.\n\n"
    "GAP = the authors mention something NOT YET DONE in their own work — "
    "a limitation, weakness, assumption, scope restriction, future-work direction, "
    "or open problem. Includes subtle phrasings: 'this work focuses on X', "
    "'we did not Y', 'we restrict to Z', 'our method assumes A', 'evaluation is "
    "restricted to B', 'inference adds latency and cost', 'unfortunately requires C'.\n\n"
    "JUNK = anything else: gratitude, numbered performance results, method or "
    "hyperparameter descriptions, prior-work citations, dataset descriptions, "
    "table/figure references, scrambled fragments, contribution claims with no gap.\n\n"
    "When uncertain whether a sentence could be a real limitation, default to GAP.\n\n"
    "Output ONLY '<idx>. GAP' or '<idx>. JUNK' per line, nothing else."
)


def find_context(sent, text, window=WINDOW):
    s_short = sent.strip()[:80]
    idx = text.find(s_short)
    if idx < 0:
        return "", ""
    end = idx + len(sent)
    before = " ".join(text[max(0, idx-400):idx].split()[-window:])
    after = " ".join(text[end:end+400].split()[:window])
    return before, after


def classify_batched(client, model, items, batch=BATCH):
    """items = (sentence, before, after, paper_id) tuples. Returns list[bool]."""
    keep = [True] * len(items)
    for i in range(0, len(items), batch):
        chunk = items[i:i+batch]
        lines = []
        for k, (sent, b, a, _pid) in enumerate(chunk):
            ctx = f"{b} >>> {sent} <<< {a}".strip()
            lines.append(f"{k+1}. {ctx}")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": "Classify:\n" + "\n".join(lines)}],
                temperature=0.0, max_tokens=10*len(chunk))
            content = resp.choices[0].message.content
            ans = {}
            for ln in content.splitlines():
                m = re.match(r"\s*(\d+)\s*[.):\-]\s*(GAP|JUNK)", ln.strip(), re.IGNORECASE)
                if m:
                    ans[int(m.group(1))] = m.group(2).upper() == "GAP"
            for k in range(len(chunk)):
                keep[i+k] = ans.get(k+1, True)
        except Exception as e:
            print(f"  batch err: {e}", flush=True)
    return keep


def evaluate(preds, gold, gold_v):
    matched = set(); tp = 0
    for sent, _, _, pid in preds:
        sub = gold_v[gold_v["paper_id"] == pid]
        for _, g in sub.iterrows():
            if max(token_containment(g["gap_sentence"], sent),
                   token_containment(sent, g["gap_sentence"])) >= MATCH_TAU:
                matched.add(g["gap_id"]); tp += 1
                break
    recall = len(matched) / len(gold)
    prec = tp / max(1, len(preds))
    f1 = 2 * recall * prec / max(1e-9, recall + prec)
    return {"n": len(preds), "matched": len(matched), "R": recall, "P": prec, "F1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--gold", default="data/bench_gap/gold_sentences_v2.tsv")
    ap.add_argument("--slicer", choices=["v1", "v2"], default="v2")
    args = ap.parse_args()

    gold = pd.read_csv(ROOT / args.gold, sep="\t", dtype={"paper_id": str})
    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec
    print(f"Gold: {len(gold)} gaps / {len(papers)} papers", flush=True)
    print(f"Slicer: Stage A {args.slicer}", flush=True)

    slice_fn = slice_terminal_regions if args.slicer == "v1" else slice_with_midpaper_anchors

    # Stage A: slice + build items
    all_items = []
    slice_total = 0
    for pid, rec in papers.items():
        regs = slice_fn(rec["text"], blocks=rec.get("blocks"))
        for r in regs:
            for s in r.sentences:
                if not _looks_like_sentence(s):
                    continue
                b, a = find_context(s, rec["text"])
                all_items.append((s, b, a, pid))
        slice_total += sum(len(r.sentences) for r in regs)
    # Dedup near-duplicates (PDF re-segmentation overlaps)
    seen = set(); items = []
    for s, b, a, pid in all_items:
        key = (pid, " ".join(s.lower().split())[:60])
        if key in seen:
            continue
        seen.add(key); items.append((s, b, a, pid))
    print(f"\nSlice sentences (dedup'd): {len(items)}", flush=True)

    # LLM classification with context
    print(f"\n=== Stage BC (gpt-4o GAP/JUNK with context, batch={BATCH}) ===", flush=True)
    t0 = time.time()
    client = get_llm_client()
    keep_mask = classify_batched(client, args.model, items)
    print(f"  classified {len(items)} in {time.time()-t0:.1f}s", flush=True)
    preds = [it for it, k in zip(items, keep_mask) if k]
    print(f"  kept {len(preds)}/{len(items)}", flush=True)

    # Metrics
    m = evaluate(preds, gold, gold)
    print(f"\n=== Stage A {args.slicer} + LLM Stage BC ({args.model}) on gold v2 ===")
    print(f"  predictions:  {m['n']}  ({m['n']/len(papers):.1f}/paper)")
    print(f"  matched gold: {m['matched']}/{len(gold)}")
    print(f"  recall:       {m['R']:.3f}")
    print(f"  precision:    {m['P']:.3f}")
    print(f"  F1:           {m['F1']:.3f}")

    # Save predictions
    out_path = ROOT / f"data/scibert_prep/pipeline_v3_gold_gaps_{args.slicer}slice.tsv"
    pd.DataFrame([{"paper_id": pid, "gap_sentence": s}
                  for s, _, _, pid in preds]).to_csv(out_path, sep="\t", index=False)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
