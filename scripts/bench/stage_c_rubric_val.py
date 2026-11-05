# -*- coding: utf-8 -*-
"""Stage C rubric strictness on gold VALIDATION papers only. Test papers are never loaded.

Short NLI at tau 0.354 supplies the candidates; Qwen judges them in reading order, batch 20.
Rubrics were written before any validation verdict was seen, though the stricter wording was
informed by reading false positives from the earlier test run. Half A selects, half B confirms.
"""
import json
import sys
import time

import numpy as np

from nli_variants import ROOT, halves, load_val
from stage_c_nli_qwen import TAUS, TYPES, candidates, evc
from stage_c_prompt_vs_model import BATCH_SCHEMA, RUBRIC, parse_json_response, qwen_client

SCORES = ROOT / "data/gold/nli_val_scores.npz"
OUT = ROOT / "docs/experiments/stage_c_rubric_val.json"
BATCH = 20
EXTRA = {"seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
KEEP_BIAS = ('Be KEEP-BIASED: when you are genuinely uncertain whether a sentence is a real gap, answer YES. '
             'Only answer NO when it clearly is NOT a gap by the rules above.')
if KEEP_BIAS not in RUBRIC:
    sys.exit("Rubric text changed; keep-bias clause not found")

NEUTRAL = RUBRIC.replace(KEEP_BIAS, "Judge each sentence on the rules above, without favouring YES or NO.")
STRICT = RUBRIC.replace(KEEP_BIAS, (
    "Be STRICT. Answer YES only when the sentence itself explicitly states the limitation, the proposed direction, "
    "or the unsolved problem. Also answer NO for: a general property or difficulty of a method or domain given as "
    "background or motivation; a statement of why the authors' approach is needed; an observed experimental outcome; "
    "and a passing remark that names no specific gap. When genuinely uncertain, answer NO."))
RUBRICS = {"keep_biased": RUBRIC, "neutral": NEUTRAL, "strict": STRICT}


def judge(client, pool, start, prompt):
    idx = range(start, min(start + BATCH, len(pool)))
    items = [{"i": i, "type": pool[i]["gap_type"], "context": pool[i]["paragraph_text"][:600],
              "sentence": pool[i]["gap_sentence"][:400]} for i in idx]
    resp = client.chat.completions.create(
        model="qwen3.8-27b", temperature=0.0, max_tokens=256 + 400 * len(items),
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
        response_format={"type": "json_schema", "json_schema": {
            "name": "gap_verdicts", "schema": BATCH_SCHEMA, "strict": True}},
        extra_body=EXTRA)
    rows = parse_json_response(resp.choices[0].message.content).get("results", [])
    return {int(r["id"]): bool(r["keep"]) for r in rows if int(r["id"]) in idx}


def run_one(name, prompt, pool, client):
    ckpt = ROOT / f"docs/experiments/stage_c_rubric_val_{name}.jsonl"
    done = {}
    if ckpt.exists():
        for raw in ckpt.read_text(encoding="utf-8").splitlines():
            r = json.loads(raw)
            done[r["i"]] = r["keep"]
    failed, t0 = [], time.time()
    with ckpt.open("a", encoding="utf-8") as fh:
        for start in range(0, len(pool), BATCH):
            idx = range(start, min(start + BATCH, len(pool)))
            if all(i in done for i in idx):
                continue
            got = {}
            for _attempt in range(2):
                try:
                    got = judge(client, pool, start, prompt)
                    if len(got) == len(idx):
                        break
                except Exception as e:  # reported; never counted as a verdict
                    print(f"{name} batch {start}: {type(e).__name__}: {e}", flush=True)
            if len(got) < len(idx):
                failed.append(start)
            for i, keep in sorted(got.items()):
                done[i] = keep
                fh.write(json.dumps({"i": i, "keep": keep}) + "\n")
            fh.flush()
            print(f"{name} {min(start + BATCH, len(pool))}/{len(pool)} [{time.time() - t0:.0f}s]", flush=True)
    return done, failed


def score(rows, kept, where):
    use = [r for r in rows if where(r)]
    gaps = sum(r["label"] != "none" for r in use)
    tp = sum((r["pid"], r["idx"]) in kept and r["label"] != "none" for r in use)
    fp = sum((r["pid"], r["idx"]) in kept and r["label"] == "none" for r in use)
    p, q = tp / (tp + fp) if tp + fp else 0.0, tp / gaps
    return {"kept": tp + fp, "tp": tp, "fp": fp, "gaps": gaps, "P": round(p, 3), "R": round(q, 3),
            "F1": round(2 * p * q / (p + q) if p + q else 0.0, 3)}


def by_type(rows, kept, where):
    out = {}
    for t in ("limitation", "future_work", "research_gap"):
        use = [r for r in rows if where(r) and r["label"] == t]
        out[t] = round(sum((r["pid"], r["idx"]) in kept for r in use) / len(use), 3)
    return out


def main():
    val, rows = load_val("val")
    npz = np.load(SCORES)
    typed = np.stack([evc(npz, h) for h in TYPES], 1)
    pool = candidates(rows, typed, TAUS["r90"])
    half_a = halves(val, {r["pid"]: r["dom"] for r in rows})
    a = lambda r: r["pid"] in half_a
    b = lambda r: r["pid"] not in half_a
    every = lambda r: True
    reached = {(c["id"], c["idx"]) for c in pool}
    result = {"candidates": len(pool), "before": {"A": score(rows, reached, a), "B": score(rows, reached, b)}}
    print(json.dumps(result), flush=True)
    client = qwen_client()
    for name, prompt in RUBRICS.items():
        done, failed = run_one(name, prompt, pool, client)
        if failed or len(done) < len(pool):
            sys.exit(f"{name}: incomplete, failed batches {failed}. Rerun to resume.")
        kept = {(pool[i]["id"], pool[i]["idx"]) for i, k in done.items() if k}
        gaps_in = sum(c["label"] != "none" for c in pool)
        tp = sum(k and pool[i]["label"] != "none" for i, k in done.items())
        fp = sum(k and pool[i]["label"] == "none" for i, k in done.items())
        result[name] = {"keeps_gaps": round(tp / gaps_in, 3), "removes_negatives": round(1 - fp / (len(pool) - gaps_in), 3),
                        "A": score(rows, kept, a), "A_recall_by_type": by_type(rows, kept, a),
                        "B": score(rows, kept, b), "B_recall_by_type": by_type(rows, kept, b),
                        "all": score(rows, kept, every)}
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
