"""Full 3-stage pipeline metrics on gold v2 (49 verified gaps / 10 papers).

Reports both Stage A versions:
  - v1 (frozen, terminal-section) — produces the existing scibert_gold_gaps.tsv
  - v2 (mid-paper anchor sweep) — needs a fresh SciBERT inference (GPU box)

For each Stage A version, reports:
  Stage A alone:   localization recall at τ=0.70 / 0.80
  Stage A + B:     SciBERT-FT classifier output
  Stage A + B + C: + context-aware GAP/JUNK LLM filter (gpt-4o)

Uses τ=0.70 token-containment matching (lenient to handle PDF scramble fragments).

    python -u scripts/eval/eval_full_pipeline_goldv2.py [--with-stage-c]
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
    slice_terminal_regions, slice_with_midpaper_anchors, token_containment,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402

MATCH_TAU = 0.70


def gold_match(sent, sub):
    for _, g in sub.iterrows():
        if max(token_containment(g["gap_sentence"], sent),
               token_containment(sent, g["gap_sentence"])) >= MATCH_TAU:
            return g["gap_id"]
    return None


def stage_a_recall(papers, gold, slice_fn, name):
    slice_total = full_total = 0
    loc_70 = loc_80 = 0
    for pid, rec in papers.items():
        regs = slice_fn(rec["text"], blocks=rec.get("blocks"))
        slice_text = " ".join(s for r in regs for s in r.sentences)
        full = split_sentences(_cut_before_references(rec["text"]))
        full_total += len(full)
        slice_total += sum(len(r.sentences) for r in regs)
        for _, g in gold[gold["paper_id"] == pid].iterrows():
            c = token_containment(g["gap_sentence"], slice_text)
            if c >= 0.7: loc_70 += 1
            if c >= 0.8: loc_80 += 1
    return {
        "name": name, "full": full_total, "slice": slice_total,
        "drop": 1 - slice_total / max(1, full_total),
        "loc@0.70": loc_70 / len(gold),
        "loc@0.80": loc_80 / len(gold),
    }


def stage_b_metrics(preds, gold, name):
    matched = set()
    tp = 0
    for _, p in preds.iterrows():
        sub = gold[gold["paper_id"] == p["paper_id"]]
        m = gold_match(p["gap_sentence"], sub)
        if m:
            matched.add(m); tp += 1
    recall = len(matched) / len(gold)
    prec = tp / max(1, len(preds))
    f1 = 2 * recall * prec / max(1e-9, recall + prec)
    return {"name": name, "n_pred": len(preds), "matched": len(matched),
            "recall": recall, "precision": prec, "F1": f1}


# === Stage C: context-aware GAP/JUNK ===
SYS_C = (
    "For each numbered input, classify the highlighted sentence (between >>> <<<) "
    "as GAP or JUNK, using surrounding context.\n"
    "GAP = authors mention something NOT YET DONE (limitation, scope, assumption, "
    "future-work, open problem).\n"
    "JUNK = anything else (gratitude, numbered result, method recipe, citation, fragment).\n"
    "Output ONLY '<idx>. GAP' or '<idx>. JUNK' per line."
)


def stage_c_filter(preds, papers, model="openai/gpt-4o", batch=10):
    from gap2idea.pipeline.llm import get_llm_client
    client = get_llm_client()
    items = []
    for _, row in preds.iterrows():
        rec = papers.get(row["paper_id"])
        before = after = ""
        if rec:
            text = rec["text"]
            sent = row["gap_sentence"]
            idx = text.find(sent.strip()[:80])
            if idx >= 0:
                end = idx + len(sent)
                before = " ".join(text[max(0, idx-400):idx].split()[-30:])
                after = " ".join(text[end:end+400].split()[:30])
        items.append((row["gap_sentence"], before, after))
    keep = [True] * len(items)
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        lines = []
        for k, (sent, b, a) in enumerate(chunk):
            ctx = f"{b} >>> {sent} <<< {a}".strip()
            lines.append(f"{k+1}. {ctx}")
        user = "Classify:\n" + "\n".join(lines)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYS_C},
                          {"role": "user", "content": user}],
                temperature=0.0, max_tokens=10 * len(chunk))
            content = resp.choices[0].message.content
            ans = {}
            for ln in content.splitlines():
                m = re.match(r"\s*(\d+)\s*[.):\-]\s*(GAP|JUNK)", ln.strip(), re.IGNORECASE)
                if m:
                    ans[int(m.group(1))] = m.group(2).upper() == "GAP"
            for k in range(len(chunk)):
                keep[i + k] = ans.get(k + 1, True)
        except Exception as e:
            print(f"  Stage C batch error: {e}", flush=True)
    return preds[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-stage-c", action="store_true", help="run gpt-4o Stage C")
    args = ap.parse_args()

    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences_v2.tsv", sep="\t", dtype={"paper_id": str})
    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec
    preds_v1 = pd.read_csv(ROOT / "data/scibert_prep/scibert_gold_gaps.tsv", sep="\t", dtype=str).fillna("")
    print(f"Gold v2: {len(gold)} gaps / {len(papers)} papers")
    print(f"Stage B (Stage A v1) preds available: {len(preds_v1)}")

    # === STAGE A ===
    print("\n" + "=" * 75)
    print("STAGE A (slice only)")
    print("=" * 75)
    a_v1 = stage_a_recall(papers, gold, slice_terminal_regions, "v1 frozen")
    a_v2 = stage_a_recall(papers, gold, slice_with_midpaper_anchors, "v2 midpaper")
    print(f"{'version':<14} {'drop':>6} {'slice/paper':>12} {'loc@0.70':>10} {'loc@0.80':>10}")
    for a in [a_v1, a_v2]:
        print(f"{a['name']:<14} {a['drop']*100:>5.1f}% {a['slice']/len(papers):>12.0f} "
              f"{a['loc@0.70']:>10.3f} {a['loc@0.80']:>10.3f}")

    # === STAGE A v1 + B ===
    print("\n" + "=" * 75)
    print("STAGE A v1 + B (SciBERT-FT, existing predictions)")
    print("=" * 75)
    b = stage_b_metrics(preds_v1, gold, "A_v1+B")
    print(f"  predictions:    {b['n_pred']}")
    print(f"  matched gold:   {b['matched']}/{len(gold)}")
    print(f"  recall:         {b['recall']:.3f}")
    print(f"  precision:      {b['precision']:.3f}")
    print(f"  F1:             {b['F1']:.3f}")

    if args.with_stage_c:
        # === STAGE A v1 + B + C ===
        print("\n" + "=" * 75)
        print("STAGE A v1 + B + C (gpt-4o GAP/JUNK with ±30-word context)")
        print("=" * 75)
        t0 = time.time()
        preds_c = stage_c_filter(preds_v1, papers, model="openai/gpt-4o")
        print(f"  Stage C kept {len(preds_c)}/{len(preds_v1)} in {time.time()-t0:.1f}s")
        c = stage_b_metrics(preds_c, gold, "A_v1+B+C")
        print(f"  matched gold:   {c['matched']}/{len(gold)}")
        print(f"  recall:         {c['recall']:.3f}")
        print(f"  precision:      {c['precision']:.3f}")
        print(f"  F1:             {c['F1']:.3f}")

    print("\n" + "=" * 75)
    print("CONSOLIDATED — Full pipeline metrics on gold v2 (49 gaps / 10 papers)")
    print("=" * 75)
    print(f"  Stage A v1 alone:        loc@0.70 = {a_v1['loc@0.70']:.3f}")
    print(f"  Stage A v2 alone:        loc@0.70 = {a_v2['loc@0.70']:.3f}  (+{a_v2['loc@0.70']-a_v1['loc@0.70']:+.3f})")
    print(f"  Stage A v1 + B:          recall={b['recall']:.3f}  precision={b['precision']:.3f}  F1={b['F1']:.3f}")
    if args.with_stage_c:
        print(f"  Stage A v1 + B + C:      recall={c['recall']:.3f}  precision={c['precision']:.3f}  F1={c['F1']:.3f}")
    print()
    print(f"  Note: Stage A v2 + B + C requires re-running SciBERT-FT on V100 with v2 slice.")
    print(f"        Run gpu_eval_v2.py for that when GPU is available.")


if __name__ == "__main__":
    main()
