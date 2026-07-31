"""Precision-first Stage C: only keeps predictions where gpt-4o is HIGHLY
CONFIDENT they are real gaps. Multi-pass verification: a sentence is kept
only if BOTH a strict-gate prompt AND a contradiction check agree it's a gap.

Goal: precision ≈ 1.0 on gold v2, even at recall cost.
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

from gap2idea.pipeline.gap_funnel import token_containment  # noqa: E402
from gap2idea.pipeline.llm import get_llm_client  # noqa: E402

MATCH_TAU = 0.70

# Strict prompt v5 — selected via fixture-based iteration on 62 labeled items.
# Final scores: 47/51 TP recall (92%), 4/5 FP caught (80%), single strawman FP residual.
SYS_STRICT = (
    "You are a strict precision filter for research-gap extraction. Each item shows a "
    "sentence between '>>>' markers with ±50 words of context.\n\n"
    "Output GAP if the sentence is one of these — about the authors' OWN work, not "
    "external systems or replaced ideas:\n"
    "  - explicit limitation, weakness, drawback, or shortcoming\n"
    "  - explicit scope restriction ('we focus on', 'restricted to')\n"
    "  - explicit assumption ('we assume', 'our method relies on')\n"
    "  - future-work direction ('future work will', 'we leave for future work', "
    "    'we plan to', 'potential extension/direction')\n"
    "  - open problem ('remains open')\n"
    "  - explicit 'we did not / cannot / do not' about their own work\n\n"
    "Output JUNK if ANY of these — even when the sentence sounds limitation-flavoured:\n"
    "  - contribution claim or positive result ('achieves', 'outperforms', 'maintains "
    "    perfect', 'notably ... [success]')\n"
    "  - method/setup description ('we propose', 'we present', 'we aim to approximate', "
    "    'in contrast we focus on')\n"
    "  - the flaw is owned by something the paper does NOT contribute and then fixes:\n"
    "       • an external system being benchmarked (e.g. evaluating GPT-X, baseline models)\n"
    "       • a baseline the authors deliberately reject ('would have huge overhead', "
    "         followed by 'therefore we do not compare with')\n"
    "       • an intermediate idea the paper then replaces (context-after: 'since "
    "         [problem], we consider Y instead', 'we therefore use', 'in response we "
    "         propose')\n"
    "  - funding, acknowledgments, citations, captions, table/figure refs\n"
    "  - hedged speculation about benefits without commitment ('could improve', "
    "    'might help')\n"
    "  - scrambled fragment\n\n"
    "Disambiguation cue: a flaw 'owned by the paper's final contribution' is GAP. "
    "A flaw 'described to motivate the paper's contribution' is JUNK. Check the "
    "context-after for the fix announcement.\n\n"
    "When in doubt → JUNK. Reply ONLY '<idx>. GAP' or '<idx>. JUNK' per line."
)

# Second-pass contradiction check.
SYS_CONFIRM = (
    "For each numbered sentence, decide: does this sentence clearly mention "
    "something the AUTHORS have NOT YET DONE in their own work (limitation, "
    "future-work, scope restriction, assumption, open problem)? "
    "Reply YES only if the answer is unambiguously yes — the sentence explicitly "
    "and clearly states the gap. If the sentence is a contribution, result, method "
    "description, citation, encouragement, speculation, or even slightly ambiguous, "
    "reply NO. Reply ONLY '<idx>. YES' or '<idx>. NO' per line."
)


def find_context(sent, text, window=50):
    s = sent.strip()[:80]
    idx = text.find(s)
    if idx < 0:
        return "", ""
    end = idx + len(sent)
    before = " ".join(text[max(0, idx-600):idx].split()[-window:])
    after = " ".join(text[end:end+600].split()[:window])
    return before, after


def call_batched(client, model, system, items, batch=15, label_yes="GAP", label_no="JUNK"):
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
                          {"role": "user", "content": "Classify:\n" + "\n".join(lines)}],
                temperature=0.0, max_tokens=10*len(chunk))
            content = resp.choices[0].message.content
            patt = rf"\s*(\d+)\s*[.):\-]\s*({label_yes}|{label_no})"
            ans = {}
            for ln in content.splitlines():
                m = re.match(patt, ln.strip(), re.IGNORECASE)
                if m:
                    ans[int(m.group(1))] = m.group(2).upper() == label_yes
            for k in range(len(chunk)):
                preds[i+k] = ans.get(k+1, False)  # default REJECT (precision-first)
        except Exception as e:
            print(f"  batch err: {e}", flush=True)
    return preds


def evaluate(preds_kept, gold):
    matched = set(); tp = 0
    for sent, _, _, pid in preds_kept:
        sub = gold[gold["paper_id"] == pid]
        for _, g in sub.iterrows():
            if max(token_containment(g["gap_sentence"], sent),
                   token_containment(sent, g["gap_sentence"])) >= MATCH_TAU:
                matched.add(g["gap_id"]); tp += 1
                break
    recall = len(matched) / len(gold)
    prec = tp / max(1, len(preds_kept))
    f1 = 2*recall*prec/max(1e-9, recall+prec)
    return {"n": len(preds_kept), "matched": len(matched),
            "recall": recall, "precision": prec, "F1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-4o")
    ap.add_argument("--input", default="data/scibert_prep/scibert_gold_gaps_v2slice.tsv",
                    help="Stage B predictions to filter")
    ap.add_argument("--no-confirm", action="store_true", help="skip second confirmation pass")
    args = ap.parse_args()

    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences_v2.tsv", sep="\t", dtype={"paper_id": str})
    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec
    preds_df = pd.read_csv(ROOT / args.input, sep="\t", dtype=str).fillna("")
    print(f"Input predictions: {len(preds_df)}")
    print(f"Gold v2: {len(gold)} gaps / {len(papers)} papers", flush=True)

    # Build context items
    items = []
    for _, row in preds_df.iterrows():
        pid = row["paper_id"]; sent = row["gap_sentence"]
        rec = papers.get(pid, {})
        b, a = find_context(sent, rec.get("text", "")) if rec else ("", "")
        items.append((sent, b, a, pid))

    client = get_llm_client()
    print(f"\n=== PASS 1: STRICT GAP/JUNK filter ({args.model}) ===", flush=True)
    t0 = time.time()
    pass1 = call_batched(client, args.model, SYS_STRICT, items, label_yes="GAP", label_no="JUNK")
    print(f"  done in {time.time()-t0:.1f}s. kept {sum(pass1)}/{len(items)}", flush=True)
    pass1_items = [it for it, k in zip(items, pass1) if k]
    m1 = evaluate(pass1_items, gold)
    print(f"  After Pass 1: n={m1['n']}  recall={m1['recall']:.3f}  precision={m1['precision']:.3f}  F1={m1['F1']:.3f}")

    if not args.no_confirm and pass1_items:
        print(f"\n=== PASS 2: confirmation pass ===", flush=True)
        t0 = time.time()
        pass2 = call_batched(client, args.model, SYS_CONFIRM, pass1_items, label_yes="YES", label_no="NO")
        print(f"  done in {time.time()-t0:.1f}s. confirmed {sum(pass2)}/{len(pass1_items)}", flush=True)
        final = [it for it, k in zip(pass1_items, pass2) if k]
        m2 = evaluate(final, gold)
        print(f"  After Pass 2: n={m2['n']}  recall={m2['recall']:.3f}  precision={m2['precision']:.3f}  F1={m2['F1']:.3f}")

        # Show the final outputs
        print(f"\n=== FINAL OUTPUT ({len(final)} predictions) ===")
        for sent, b, a, pid in final[:20]:
            ctx = f"...{b[-40:]} **{sent[:90]}** {a[:40]}..."
            print(f"  [{pid}] {ctx}")


if __name__ == "__main__":
    main()
