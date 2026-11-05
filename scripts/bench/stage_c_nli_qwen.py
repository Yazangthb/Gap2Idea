# -*- coding: utf-8 -*-
"""Frozen test run: short typed ModernBERT NLI as Stage B, then Qwen Stage C.

Hypotheses and thresholds were fixed on validation (nli_variants.md) before any test scoring:
"This is a limitation." / "This is future work." / "This is a research gap.", maximum of the
entailment-vs-contradiction scores, tau 0.669 (R>=80) and 0.354 (R>=90).

  export FILE     write label-free test sentences for nli_score_remote.py
  stage-b         Stage B metrics only, including the reproduction check of the original 0.617
  run             Stage C on both thresholds with checkpoints, then all comparisons
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

from nli_variants import ROOT, load_val
from stage_c_full_qwen import CKPT as BGE_CKPT
from stage_c_full_qwen import judge, load_in_reading_order
from stage_c_heldout import selection_keys
from stage_c_prompt_vs_model import qwen_client

SCORES = ROOT / "data/gold/nli_test_scores.npz"
OUT = ROOT / "docs/experiments/stage_c_nli_qwen.json"
VARIANTS = json.loads((ROOT / "docs/experiments/nli_variants.json").read_text(encoding="utf-8"))
TAUS = {name: VARIANTS["typed_c_max_evc"]["thresholds"][key] for name, key in (("r80", "R>=80"), ("r90", "R>=90"))}
ORIG_TAU = 0.75  # tuned on validation in the Oct 27 run
TYPES = {"lim_c": "limitation", "fw_c": "future_work", "gap_c": "research_gap"}
BATCH, TEST_GAPS = 20, 664


def evc(npz, h):
    a = npz[f"sent__{h}"]
    return a[:, 0] / (a[:, 0] + a[:, 2] + 1e-9)


def prf(tp, fp, gaps):
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / gaps if gaps else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rounded(x):
    return [round(v, 3) for v in x]


def setup():
    _, rows = load_val("test")
    if sum(r["label"] != "none" for r in rows) != TEST_GAPS:
        sys.exit("Test gold does not match the 664 gaps used elsewhere")
    npz = np.load(SCORES)
    typed = np.stack([evc(npz, h) for h in TYPES], 1)
    return rows, typed, evc(npz, "orig")


def candidates(rows, typed, tau):
    names = list(TYPES.values())
    out = []
    for i, r in enumerate(rows):
        if not r["junk"] and typed[i].max() >= tau:
            out.append({"row": i, "id": r["pid"], "idx": r["idx"], "gap_sentence": r["sentence"],
                        "paragraph_text": r["context"], "gap_type": names[int(typed[i].argmax())], "label": r["label"]})
    return out


def stage_b():
    rows, typed, orig = setup()
    y = np.array([r["label"] != "none" for r in rows])
    res = {}
    p = orig >= ORIG_TAU
    res["orig_tau075_no_junk_filter"] = rounded(prf(int((p & y).sum()), int((p & ~y).sum()), TEST_GAPS))
    keep = np.array([not r["junk"] for r in rows])
    for name, tau in TAUS.items():
        f = typed.max(1) >= tau
        f &= keep
        res[name] = {"tau": tau, "kept": int(f.sum()), "PRF": rounded(prf(int((f & y).sum()), int((f & ~y).sum()), TEST_GAPS))}
    print(json.dumps(res, indent=1), flush=True)
    return res


def run_stage_c(name, pool, client):
    ckpt = ROOT / f"docs/experiments/stage_c_nli_{name}_verdicts.jsonl"
    done = {}
    if ckpt.exists():
        for raw in ckpt.read_text(encoding="utf-8").splitlines():
            r = json.loads(raw)
            done[r["i"]] = r["keep"]
    calls, failed, t0 = 0, [], time.time()
    with ckpt.open("a", encoding="utf-8") as fh:
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
                    print(f"{name} batch {start} failed: {type(e).__name__}: {e}", flush=True)
            if len(got) < len(idx):
                failed.append(start)
            for i, keep in sorted(got.items()):
                done[i] = keep
                fh.write(json.dumps({"i": i, "id": pool[i]["id"], "idx": pool[i]["idx"], "keep": keep}) + "\n")
            fh.flush()
            print(f"{name} {min(start + BATCH, len(pool))}/{len(pool)} [{time.time() - t0:.0f}s]", flush=True)
    return done, calls, round(time.time() - t0, 1), failed


def bge_kept():
    pool = load_in_reading_order()
    verdict = {}
    for raw in BGE_CKPT.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        verdict[r["i"]] = r["keep"]
    return {(c["id"], c["idx"]) for i, c in enumerate(pool) if verdict[i]}


def score_set(rows, kept, mask=None):
    """kept: set of (paper, idx). Returns scores overall, per type and per domain."""
    use = [r for r in rows if mask is None or (r["pid"], r["idx"]) not in mask]
    def calc(sub):
        gaps = sum(r["label"] != "none" for r in sub)
        tp = sum((r["pid"], r["idx"]) in kept and r["label"] != "none" for r in sub)
        fp = sum((r["pid"], r["idx"]) in kept and r["label"] == "none" for r in sub)
        return tp, fp, gaps
    tp, fp, gaps = calc(use)
    res = {"kept": tp + fp, "tp": tp, "fp": fp, "PRF": rounded(prf(tp, fp, gaps))}
    res["recall_by_type"] = {t: round(sum((r["pid"], r["idx"]) in kept for r in use if r["label"] == t)
                                       / sum(r["label"] == t for r in use), 3)
                             for t in ("limitation", "future_work", "research_gap")}
    res["by_domain"] = {d: rounded(prf(*calc([r for r in use if r["dom"] == d])))
                        for d in ("arxiv_cs", "biomed_sal")}
    return res


def paired(rows, a, b, n=2000):
    """F1 difference a minus b, resampling papers. Sentences within a paper are not independent."""
    by_paper = {}
    for r in rows:
        by_paper.setdefault(r["pid"], []).append(r)
    def counts(p, kept):
        tp = fp = gaps = 0
        for r in by_paper[p]:
            gap = r["label"] != "none"
            gaps += gap
            k = (r["pid"], r["idx"]) in kept
            tp += k and gap
            fp += k and not gap
        return tp, fp, gaps
    table = {p: (counts(p, a), counts(p, b)) for p in by_paper}
    papers, rng, diffs = list(by_paper), random.Random(0), []
    for _ in range(n):
        pick = rng.choices(papers, k=len(papers))
        fa = prf(*[sum(table[p][0][j] for p in pick) for j in range(3)])[2]
        fb = prf(*[sum(table[p][1][j] for p in pick) for j in range(3)])[2]
        diffs.append(fa - fb)
    diffs.sort()
    all_papers = papers
    point = prf(*[sum(table[p][0][j] for p in all_papers) for j in range(3)])[2] - \
        prf(*[sum(table[p][1][j] for p in all_papers) for j in range(3)])[2]
    return {"f1_diff": round(point, 3), "ci95": [round(diffs[int(0.025 * n)], 3), round(diffs[int(0.975 * n)], 3)]}


def run():
    rows, typed, orig = setup()
    client = qwen_client()
    seen = selection_keys()
    bge = bge_kept()
    orig_kept = {(r["pid"], r["idx"]) for i, r in enumerate(rows) if orig[i] >= ORIG_TAU}
    result = {"taus": TAUS, "stage_b": stage_b(),
              "baselines": {"bge_small_plus_qwen": score_set(rows, bge), "modernbert_orig_alone": score_set(rows, orig_kept)}}
    for name, tau in TAUS.items():
        pool = candidates(rows, typed, tau)
        done, calls, secs, failed = run_stage_c(name, pool, client)
        if failed or len(done) < len(pool):
            sys.exit(f"{name}: incomplete, failed batches {failed}. Rerun to resume; no results written.")
        kept = {(pool[i]["id"], pool[i]["idx"]) for i, k in done.items() if k}
        reached = {(c["id"], c["idx"]) for c in pool}
        gaps_in = sum(c["label"] != "none" for c in pool)
        neg_in = len(pool) - gaps_in
        tp = sum(k and pool[i]["label"] != "none" for i, k in done.items())
        fp = sum(k and pool[i]["label"] == "none" for i, k in done.items())
        result[name] = {
            "candidates": len(pool), "calls": calls, "seconds": secs,
            "stage_c_keeps_gaps": round(tp / gaps_in, 3), "stage_c_removes_negatives": round(1 - fp / neg_in, 3),
            "before": score_set(rows, reached), "after": score_set(rows, kept),
            "after_excluding_196_selection_sentences": score_set(rows, kept, seen),
            "vs_bge_small_plus_qwen": paired(rows, kept, bge),
            "vs_modernbert_alone": paired(rows, kept, orig_kept),
        }
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    result["r80_vs_r90"] = paired(rows, {(c["id"], c["idx"]) for c in []} or kept_set(rows, typed, "r80"), kept_set(rows, typed, "r90"))
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1), flush=True)


def kept_set(rows, typed, name):
    pool = candidates(rows, typed, TAUS[name])
    kept = set()
    for raw in (ROOT / f"docs/experiments/stage_c_nli_{name}_verdicts.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        if r["keep"]:
            kept.add((pool[r["i"]]["id"], pool[r["i"]]["idx"]))
    return kept


def export():
    _, rows = load_val("test")
    Path(sys.argv[2]).write_text(json.dumps([{"sentence": r["sentence"]} for r in rows], ensure_ascii=False), encoding="utf-8")
    print("test sentences", len(rows), "->", sys.argv[2])


if __name__ == "__main__":
    {"export": export, "stage-b": stage_b, "run": run}[sys.argv[1]]()
