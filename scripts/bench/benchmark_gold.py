# -*- coding: utf-8 -*-
"""Benchmark the Gap2Idea funnel on the LLM-ensemble gold (data/gold/gap_gold_final.jsonl).

The funnel (Stage A section slice -> Stage B cue rules [+ bge-small head] -> optional Stage C LLM)
targets ONLY limitation + future_work gaps; research_gap is out of funnel scope (the LLM/full-paper
path). So we score against the funnel's targets (limitation+future_work) and report research_gap
separately. Predictions are matched to gold sentences by scramble-robust token containment.

  python scripts/bench/benchmark_gold.py                 # Stage-A ceiling + rules + hybrid(bge-small)
  python scripts/bench/benchmark_gold.py --no-hybrid     # skip the encoder head (rules only)

Writes docs/experiments/gold_benchmark_results.md.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import (
    extract_gaps, slice_section_aware, token_containment, EmbeddingGapHead,
)

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
HEAD = ROOT / "data" / "pipeline_head.joblib"
OUTDOC = ROOT / "docs" / "experiments" / "gold_benchmark_results.md"
TAU = 0.7
TARGET = {"limitation", "future_work"}
# SAL abbreviated section codes -> heading words Stage A's section regex understands
SAL_SEC_MAP = {"DISCUSS": "Discussion", "CONCL": "Conclusion", "INTRO": "Introduction",
               "METHODS": "Methods", "RESULTS": "Results", "ABSTRACT": "Abstract",
               "TITLE": "Title", "SUPPL": "Supplementary", "APPENDIX": "Appendix"}


def matched(a, b, t=TAU):
    return max(token_containment(a, b), token_containment(b, a)) >= t


def sal_blocks(sents, secs):
    """Reconstruct heading/text blocks from SAL per-sentence (sent, sec)."""
    blocks, last = [], None
    for s, sec in zip(sents, secs):
        if sec != last:
            blocks.append({"role": "heading", "text": SAL_SEC_MAP.get(sec, sec.title())})
            last = sec
        blocks.append({"role": "text", "text": s})
    return blocks


def slice_sentences(text, blocks):
    return [s for r in slice_section_aware(text, blocks=blocks) for s in r.sentences]


def score(preds, gold):
    """preds: list of dicts (gap_sentence, paragraph_text). gold: list of (sent,label).
    Returns dict of counts for this paper."""
    g_tgt = [g for g in gold if g[1] in TARGET]
    g_any = gold
    # recall over target gaps
    rec_hit = 0
    tgt_by_type = defaultdict(lambda: [0, 0])
    for s, lab in g_tgt:
        hit = any(matched(s, p["gap_sentence"]) or token_containment(s, p["paragraph_text"]) >= TAU
                  for p in preds)
        rec_hit += hit
        tgt_by_type[lab][0] += hit; tgt_by_type[lab][1] += 1
    # precision over predictions
    tp_strict = tp_any = 0
    for p in preds:
        if any(matched(p["gap_sentence"], s) for s, lab in g_tgt): tp_strict += 1
        if any(matched(p["gap_sentence"], s) for s, lab in g_any): tp_any += 1
    return dict(n_pred=len(preds), n_tgt=len(g_tgt), rec_hit=rec_hit,
                tp_strict=tp_strict, tp_any=tp_any,
                by_type={k: tuple(v) for k, v in tgt_by_type.items()})


def ceiling(slice_sents, gold):
    g_tgt = [g for g in gold if g[1] in TARGET]
    hit = sum(1 for s, _ in g_tgt
              if any(token_containment(s, ss) >= TAU for ss in slice_sents))
    return hit, len(g_tgt)


def prf(tp, pred, gold_n):
    P = tp / pred if pred else 0.0
    R = tp / gold_n if gold_n else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def agg(rows):
    """Aggregate per-paper score dicts -> micro P/R/F and per-type recall."""
    n_pred = sum(r["n_pred"] for r in rows)
    n_tgt = sum(r["n_tgt"] for r in rows)
    rec = sum(r["rec_hit"] for r in rows)
    tps = sum(r["tp_strict"] for r in rows)
    tpa = sum(r["tp_any"] for r in rows)
    bt = defaultdict(lambda: [0, 0])
    for r in rows:
        for k, (h, t) in r["by_type"].items():
            bt[k][0] += h; bt[k][1] += t
    Ps, R, Fs = prf(tps, n_pred, n_tgt)
    Pa, _, Fa = prf(tpa, n_pred, n_tgt)
    return dict(n_pred=n_pred, n_tgt=n_tgt, R=rec / n_tgt if n_tgt else 0.0,
                P_strict=Ps, F_strict=Fs, P_lenient=Pa, F_lenient=Fa,
                by_type={k: (h / t if t else 0.0, t) for k, (h, t) in bt.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hybrid", action="store_true", help="skip the bge-small encoder head")
    ap.add_argument("--tau", type=float, default=TAU)
    args = ap.parse_args()
    globals()["TAU"] = args.tau

    # gold per paper
    gold = defaultdict(list); domain = {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        g = json.loads(ln)
        gold[g["paper_id"]].append((g["sentence"], g["label"]))
        domain[g["paper_id"]] = g["domain"]

    corp = {p["paper_id"]: p for p in load_corpus()}
    arxiv_text = {}
    for ln in (ROOT / "data" / "paper_texts.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); arxiv_text[str(r["id"])] = str(r["text"])

    head = None
    hybrid = not args.no_hybrid
    if hybrid:
        if not HEAD.exists():
            print(f"no head at {HEAD}; running rules only"); hybrid = False
        else:
            import joblib
            clf = joblib.load(HEAD)
            enc = EmbeddingGapHead.load_encoder("BAAI/bge-small-en-v1.5")
            head = EmbeddingGapHead(enc, clf, "BAAI/bge-small-en-v1.5")

    # build per-paper inputs once
    inp = {}
    for pid in gold:
        p = corp.get(pid)
        if not p:
            continue
        if p["domain"] == "biomed_sal":
            inp[pid] = ("", sal_blocks(p["sents"], p["secs"]))
        else:
            inp[pid] = (arxiv_text.get(pid, " ".join(p["sents"])), None)

    results = {}  # mode -> domain -> [per-paper score]
    for mode in (["ceiling", "rules"] + (["hybrid"] if hybrid else [])):
        results[mode] = defaultdict(list)
        for pid, (text, blocks) in inp.items():
            dom = domain[pid]
            if mode == "ceiling":
                sl = slice_sentences(text, blocks)
                hit, tot = ceiling(sl, gold[pid])
                results[mode][dom].append(dict(n_pred=0, n_tgt=tot, rec_hit=hit,
                                               tp_strict=0, tp_any=0, by_type={}))
            else:
                m = "rules" if mode == "rules" else "hybrid"
                preds = extract_gaps(pid, text, blocks=blocks,
                                     head=head if m == "hybrid" else None, mode=m)
                results[mode][dom].append(score(preds, gold[pid]))

    # ---- report ----
    doms = ["arxiv_cs", "biomed_sal"]
    L = ["# Funnel benchmark on the gap gold\n",
         f"Gold: **{sum(len(v) for v in gold.values())} gaps / {len(gold)} papers** "
         "(arXiv CS + biomedical SAL). Funnel targets **limitation + future_work**; "
         "**research_gap is out of funnel scope** (LLM/full-paper path) and excluded from "
         f"funnel P/R. Matching: token-containment >= {args.tau}.\n"]

    # Stage A ceiling
    L.append("## Stage A — localization recall ceiling (target gaps inside the section slice)\n")
    L.append("| domain | ceiling recall |\n|---|---|")
    for d in doms + ["ALL"]:
        rows = sum((results["ceiling"][x] for x in (doms if d == "ALL" else [d])), [])
        a = agg(rows)
        L.append(f"| {d} | {a['R']*100:.1f}% ({sum(r['rec_hit'] for r in rows)}/{a['n_tgt']}) |")

    # detector P/R/F
    for mode in (["rules"] + (["hybrid"] if hybrid else [])):
        tag = "Stage A + cue rules" if mode == "rules" else "Stage A + cue rules + bge-small head (hybrid)"
        L.append(f"\n## {mode} — {tag}\n")
        L.append("| domain | recall (target) | precision (strict / lenient) | F1 (strict) | preds |\n|---|---|---|---|---|")
        for d in doms + ["ALL"]:
            rows = sum((results[mode][x] for x in (doms if d == "ALL" else [d])), [])
            a = agg(rows)
            L.append(f"| {d} | {a['R']*100:.1f}% | {a['P_strict']*100:.1f}% / {a['P_lenient']*100:.1f}% "
                     f"| {a['F_strict']*100:.1f}% | {a['n_pred']} |")
        # per-type recall (ALL)
        aall = agg(sum((results[mode][x] for x in doms), []))
        bt = aall["by_type"]
        L.append("\nPer-type recall (all): " + ", ".join(
            f"{k} {v[0]*100:.1f}% (n={v[1]})" for k, v in sorted(bt.items())))

    L.append("\n### Notes\n"
             "- **strict precision** = prediction matches a gold *limitation/future_work*; "
             "**lenient** = matches any gold gap (incl. research_gap the funnel found but doesn't target).\n"
             "- research_gap (the majority gold type) is deliberately not a funnel target; Stage C / "
             "full-paper LLM is the path for it.\n"
             "- hybrid uses the bge-small(frozen)+logreg head (`data/pipeline_head.joblib`); the "
             "encoder pairing is assumed from the 384-dim head.\n"
             "- gold is LLM-ensemble labelled (pending human validation), so these are provisional.")

    doc = "\n".join(L) + "\n"
    OUTDOC.parent.mkdir(parents=True, exist_ok=True)
    OUTDOC.write_text(doc, encoding="utf-8")
    print(doc)
    print("wrote", OUTDOC)


if __name__ == "__main__":
    main()
