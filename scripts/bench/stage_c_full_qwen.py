# -*- coding: utf-8 -*-
"""Stage C on every balanced funnel-v2 candidate of the gold test split.

Qwen3.8-27B, Sonnet's keep-biased rubric with context, batch 20, no thinking, papers
in reading order. Verdicts are appended to a checkpoint after each batch, so a rerun
resumes. Scores use LLM gold; end-to-end recall is against all 664 test gaps.

Usage: stage_c_full_qwen.py [--limit N]
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

from stage_c_prompt_vs_model import BATCH_SCHEMA, ROOT, RUBRIC, parse_json_response, qwen_client

CANDS = ROOT / "data/gold/funnel_v2_candidates_balanced.json"
CKPT = ROOT / "docs/experiments/stage_c_full_qwen_verdicts.jsonl"
OUT = ROOT / "docs/experiments/stage_c_full_qwen.json"
SHUFFLED = ROOT / "docs/experiments/stage_c_prompt_vs_model_qwen.json"
MODEL, BATCH, TEST_GAPS = "qwen3.8-27b", 20, 664
EXTRA = {"seed": 0, "chat_template_kwargs": {"enable_thinking": False}}


def load_in_reading_order():
    cands = json.loads(CANDS.read_text(encoding="utf-8"))
    first_seen = {}
    for c in cands:
        first_seen.setdefault(c["id"], len(first_seen))
    return sorted(cands, key=lambda c: (first_seen[c["id"]], c["idx"]))


def judge(client, pool, start):
    """Returns {global index: keep} for one batch; ids absent from the reply are left out."""
    idx = range(start, min(start + BATCH, len(pool)))
    items = [{"i": i, "type": pool[i]["gap_type"], "context": pool[i]["paragraph_text"][:600],
              "sentence": pool[i]["gap_sentence"][:400]} for i in idx]
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=256 + 400 * len(items),
        messages=[{"role": "system", "content": RUBRIC},
                  {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "gap_verdicts", "schema": BATCH_SCHEMA, "strict": True}},
        extra_body=EXTRA)
    rows = parse_json_response(resp.choices[0].message.content).get("results", [])
    return {int(r["id"]): bool(r["keep"]) for r in rows if int(r["id"]) in idx}


def run(pool):
    done = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["i"]] = r["keep"]
    client, calls, failed, t0 = qwen_client(), 0, [], time.time()
    with CKPT.open("a", encoding="utf-8") as fh:
        for start in range(0, len(pool), BATCH):
            idx = range(start, min(start + BATCH, len(pool)))
            if all(i in done for i in idx):
                continue
            got = {}
            for _attempt in range(2):
                try:
                    got = judge(client, pool, start)
                    calls += 1
                    if len(got) == len(idx):
                        break
                except Exception as e:  # recorded and reported; never counted as a verdict
                    print(f"batch {start} failed: {type(e).__name__}: {e}", flush=True)
            if len(got) < len(idx):
                failed.append(start)
            for i, keep in sorted(got.items()):
                done[i] = keep
                fh.write(json.dumps({"i": i, "id": pool[i]["id"], "idx": pool[i]["idx"], "keep": keep}) + "\n")
            fh.flush()
            print(f"{min(start + BATCH, len(pool))}/{len(pool)} [{time.time() - t0:.0f}s]", flush=True)
    return done, calls, time.time() - t0, failed


def prf(rows):
    gaps = sum(c["label"] != "none" for c, _ in rows)
    tp = sum(k and c["label"] != "none" for c, k in rows)
    fp = sum(k and c["label"] == "none" for c, k in rows)
    return {"candidates": len(rows), "gaps": gaps, "kept": tp + fp, "tp": tp, "fp": fp,
            "precision_before": round(gaps / len(rows), 3) if rows else 0.0,
            "precision_after": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "gaps_kept": round(tp / gaps, 3) if gaps else None}


def end_to_end(tp, fp):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / TEST_GAPS
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap(by_paper, n=2000):
    """Papers are resampled because sentences within a paper are not independent."""
    rng, papers, stats = random.Random(0), list(by_paper), []
    scale = TEST_GAPS / sum(c["label"] != "none" for rows in by_paper.values() for c, _ in rows)
    for _ in range(n):
        tp = fp = gaps = 0
        for p in rng.choices(papers, k=len(papers)):
            for c, k in by_paper[p]:
                gaps += c["label"] != "none"
                tp += k and c["label"] != "none"
                fp += k and c["label"] == "none"
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (gaps * scale) if gaps else 0.0
        stats.append((prec, rec, 2 * prec * rec / (prec + rec) if prec + rec else 0.0))
    ci = lambda v: [round(sorted(v)[int(0.025 * n)], 3), round(sorted(v)[int(0.975 * n)], 3)]
    return dict(zip(("precision", "recall", "f1"), (ci([s[j] for s in stats]) for j in range(3))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pool = load_in_reading_order()
    if args.limit:
        pool = pool[:args.limit]
    done, calls, secs, failed = run(pool)
    rows = [(c, done[i]) for i, c in enumerate(pool) if i in done]
    print(f"judged {len(rows)}/{len(pool)} in {calls} calls [{secs:.0f}s]; failed batches: {failed}", flush=True)
    if args.limit or len(rows) < len(pool):
        print("partial run: not writing final results")
        print(prf(rows))
        sys.exit(0 if args.limit else 1)

    by_paper = defaultdict(list)
    for c, k in rows:
        by_paper[c["id"]].append((c, k))
    overall = prf(rows)
    p, r, f = end_to_end(overall["tp"], overall["fp"])
    pb, rb, fb = end_to_end(overall["gaps"], overall["candidates"] - overall["gaps"])
    result = {
        "config": {"model": MODEL, "batch": BATCH, **EXTRA, "calls": calls, "seconds": round(secs, 1)},
        "overall": overall,
        "end_to_end": {"before": [round(pb, 3), round(rb, 3), round(fb, 3)],
                       "after": [round(p, 3), round(r, 3), round(f, 3)], "ci95": bootstrap(by_paper)},
        "by_gold_type": {t: prf([(c, k) for c, k in rows if c["label"] == t])
                         for t in ("limitation", "future_work", "research_gap")},
        "by_domain": {d: prf([(c, k) for c, k in rows if c["id"].startswith("PMC") == (d == "biomed")])
                      for d in ("biomed", "arxiv")},
        "papers": len(by_paper),
    }

    # Same sentences, same setup: reading order here, shuffled in the 80-sentence test.
    old = json.loads(SHUFFLED.read_text(encoding="utf-8"))
    verdict_here = {(c["id"], c["idx"]): k for c, k in rows}
    pairs = [(verdict_here.get((s["id"], s["idx"])), v, s["label"])
             for s, v in zip(old["sample"], old["rubric_20"]["verdicts"])]
    pairs = [x for x in pairs if x[0] is not None]
    result["reading_order_vs_shuffled_80"] = {
        "compared": len(pairs), "agree": sum(a == b for a, b, _ in pairs),
        "gaps_kept_reading_order": sum(a for a, _, l in pairs if l != "none"),
        "gaps_kept_shuffled": sum(b for _, b, l in pairs if l != "none"),
        "fp_reading_order": sum(a for a, _, l in pairs if l == "none"),
        "fp_shuffled": sum(b for _, b, l in pairs if l == "none"),
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
