# -*- coding: utf-8 -*-
"""End-to-end funnel benchmark v2 on the gold, with the NEW recall-first Stage A.

Stage A (recall-first):  junk-filter (drop references / tables / figure captions / non-sentences
   / biomed boilerplate sections) + the gold-trained bge-small head at a LOW threshold tuned on
   VAL for a target recall. Stage B = same head's predicted type. Stage C (optional) = LLM filter.

Operates directly on the gold-aligned sentence list, so recall/precision are EXACT (no matching).
Paper-level split from data/gold/splits.json; threshold tuned on VAL, reported on TEST.

  python scripts/bench/benchmark_funnel_v2.py                 # A+B, target recall 0.95
  python scripts/bench/benchmark_funnel_v2.py --target 0.98
  python scripts/bench/benchmark_funnel_v2.py --stage-c --sc-model yandexgpt-5-lite --sc-limit 800

Writes docs/experiments/funnel_v2_results.md and data/gold/funnel_v2_candidates.json.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
import numpy as np
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import EmbeddingGapHead, _is_reference, _looks_like_sentence

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
SPLITS = ROOT / "data" / "gold" / "splits.json"
HEAD = ROOT / "data" / "gold" / "gap_head_gold.joblib"
DOC = ROOT / "docs" / "experiments" / "funnel_v2_results.md"
CAND = ROOT / "data" / "gold" / "funnel_v2_candidates.json"
# biomed sections that never carry gaps (corpus analysis) -> junk-drop
DROP_SEC = {"TITLE", "TABLE", "FIG", "ABBR", "COMP_INT", "AUTH_CONT", "ACK_FUND", "SUPPL"}
GAPCLS = ["limitation", "future_work", "research_gap"]


def is_junk(sent, sec):
    if sec in DROP_SEC:
        return True
    return _is_reference(sent) or not _looks_like_sentence(sent)


def rows_for(pids, corp, gold, dom):
    out = []
    for pid in pids:
        p = corp[pid]; secs = p.get("secs")
        for i, s in enumerate(p["sents"]):
            sec = secs[i] if secs else None
            out.append(dict(pid=pid, idx=i, sent=s, dom=dom[pid],
                            label=gold[pid].get(i, "none"), junk=is_junk(s, sec)))
    return out


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def evaluate(rows, keep_flag):
    """keep_flag: bool per row. Binary gap + per-domain + per-type + type-acc."""
    tp = fp = fn = 0
    by_dom = defaultdict(lambda: [0, 0, 0])
    by_type = defaultdict(lambda: [0, 0])
    tacc_ok = tacc_tot = 0
    for r, keep in zip(rows, keep_flag):
        gap = r["label"] != "none"
        if keep and gap:
            tp += 1; by_dom[r["dom"]][0] += 1
            by_type[r["label"]][0] += 1
            tacc_tot += 1; tacc_ok += (r.get("pred_type") == r["label"])
        elif keep and not gap:
            fp += 1; by_dom[r["dom"]][1] += 1
        elif not keep and gap:
            fn += 1; by_dom[r["dom"]][2] += 1
        if gap:
            by_type[r["label"]][1] += 1
    P, R, F = prf(tp, fp, fn)
    ret = sum(keep_flag) / len(keep_flag)
    return dict(P=P, R=R, F=F, retention=ret, tp=tp, fp=fp, fn=fn,
                by_dom={d: prf(v[0], v[1], v[2]) for d, v in by_dom.items()},
                by_type={t: (v[0] / v[1] if v[1] else 0, v[1]) for t, v in by_type.items()},
                type_acc=tacc_ok / tacc_tot if tacc_tot else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.95)
    ap.add_argument("--stage-c", action="store_true")
    ap.add_argument("--sc-model", default="yandexgpt-5-lite")
    ap.add_argument("--sc-limit", type=int, default=0, help="cap candidates judged (0=all)")
    args = ap.parse_args()

    split = json.loads(SPLITS.read_text())
    gold = defaultdict(dict); dom = {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            g = json.loads(ln); gold[g["paper_id"]][g["idx"]] = g["label"]; dom[g["paper_id"]] = g["domain"]
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in dom}

    va = rows_for(split["val"], corp, gold, dom)
    te = rows_for(split["test"], corp, gold, dom)

    b = __import__("joblib").load(HEAD)
    enc = EmbeddingGapHead.load_encoder(b["encoder_name"]); clf = b["clf"]
    none_j = list(clf.classes_).index("none")
    gap_js = {c: list(clf.classes_).index(c) for c in GAPCLS if c in clf.classes_}

    def score(rows):
        nz = [r for r in rows if not r["junk"]]
        X = enc.encode([r["sent"] for r in nz], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        proba = clf.predict_proba(X)
        for r, row in zip(nz, proba):
            r["pgap"] = float(1.0 - row[none_j])
            r["pred_type"] = max(gap_js, key=lambda c: row[gap_js[c]])
        for r in rows:
            if r["junk"]:
                r["pgap"] = -1.0; r["pred_type"] = "none"
    score(va); score(te)

    # tune tau on val for target recall over ALL gold gaps
    vg = [r for r in va if not r["junk"]]
    vg.sort(key=lambda r: -r["pgap"])
    totpos = sum(1 for r in va if r["label"] != "none")
    cum = 0; tau = 0.0
    for r in vg:
        cum += (r["label"] != "none")
        if cum / totpos >= args.target:
            tau = r["pgap"]; break
    # max-F1 tau on val (reference, balanced)
    bestF = (-1, 0.5)
    for t in np.arange(0.1, 0.9, 0.05):
        kf = [(not r["junk"]) and r["pgap"] >= t for r in va]
        f = evaluate(va, kf)["F"]
        if f > bestF[0]: bestF = (f, round(float(t), 2))
    tauF = bestF[1]

    def keepflag(rows, t): return [(not r["junk"]) and r["pgap"] >= t for r in rows]
    recallfirst = evaluate(te, keepflag(te, tau))
    balanced = evaluate(te, keepflag(te, tauF))
    # junk-filter-only ceiling
    jf = evaluate(te, [not r["junk"] for r in te])

    # save kept candidates for Stage C (recall-first AND balanced)
    kept = [r for r, k in zip(te, keepflag(te, tau)) if k]
    kept_bal = [r for r, k in zip(te, keepflag(te, tauF)) if k]
    def ctx(pid, i, w=1):
        s = corp[pid]["sents"]; lo, hi = max(0, i - w), min(len(s), i + w + 1)
        return " ".join(s[lo:hi])
    def dump(rows, path):
        path.write_text(json.dumps([{"id": r["pid"], "idx": r["idx"], "gap_sentence": r["sent"],
                                     "gap_type": r["pred_type"], "paragraph_text": ctx(r["pid"], r["idx"]),
                                     "label": r["label"]} for r in rows]), encoding="utf-8")
    dump(kept, CAND)
    dump(kept_bal, CAND.with_name("funnel_v2_candidates_balanced.json"))

    def line(m):
        return (f"R={m['R']*100:.1f}% P={m['P']*100:.1f}% F1={m['F']*100:.1f}% keep={m['retention']*100:.1f}%")

    L = ["# Funnel v2 — recall-first Stage A (junk-filter + bge-small head) on the gold\n",
         f"Exact per-sentence eval on the gold-aligned TEST sentences. Stage-A head threshold tuned on "
         f"VAL for target recall {args.target:.0%}. Gap = any type (lim/fw/research_gap).\n",
         "| config | recall | precision | F1 | retention |\n|---|---|---|---|---|",
         f"| junk-filter only (ceiling) | {jf['R']*100:.1f}% | {jf['P']*100:.1f}% | {jf['F']*100:.1f}% | {jf['retention']*100:.1f}% |",
         f"| **A+B recall-first (tau={tau:.3f})** | **{recallfirst['R']*100:.1f}%** | {recallfirst['P']*100:.1f}% | {recallfirst['F']*100:.1f}% | {recallfirst['retention']*100:.1f}% |",
         f"| A+B balanced (max-F1 tau={tauF}) | {balanced['R']*100:.1f}% | {balanced['P']*100:.1f}% | {balanced['F']*100:.1f}% | {balanced['retention']*100:.1f}% |"]
    L.append("\n**Recall-first per-domain:** " + " · ".join(
        f"{d}: R={v[1]*100:.1f}% P={v[0]*100:.1f}%" for d, v in sorted(recallfirst["by_dom"].items())))
    L.append("**Recall-first per-type recall:** " + " · ".join(
        f"{t}: {v[0]*100:.1f}% (n={v[1]})" for t, v in sorted(recallfirst["by_type"].items())))
    L.append(f"**Type accuracy (detected gaps):** {recallfirst['type_acc']*100:.1f}%")
    gaps_in_junk = sum(1 for r in te if r["junk"] and r["label"] != "none")
    L.append(f"\nGold gaps lost to the junk-filter: {gaps_in_junk} / {sum(1 for r in te if r['label']!='none')} "
             f"(= the {100-jf['R']*100:.1f}pt gap below 100% recall).")
    L.append(f"\nCandidates saved for Stage C: {len(kept)} -> `{CAND.name}`.")

    sc_summary = None
    if args.stage_c:
        from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
        pool = kept[:args.sc_limit] if args.sc_limit else kept
        print(f"Stage C: judging {len(pool)} candidates with {args.sc_model} ...", flush=True)
        filt = LLMGapFilter(backend="api", model=args.sc_model, mode="validate")
        rows_in = [{"id": r["pid"], "gap_type": r["pred_type"], "gap_sentence": r["sent"],
                    "paragraph_text": ctx(r["pid"], r["idx"])} for r in pool]
        out = filt.filter_gaps(rows_in, protect_rules=False)
        keptset = {(d["id"], d["gap_sentence"]) for d in out}
        tp = sum(1 for r in pool if (r["pid"], r["sent"]) in keptset and r["label"] != "none")
        fp = sum(1 for r in pool if (r["pid"], r["sent"]) in keptset and r["label"] == "none")
        kept_pos_before = sum(1 for r in pool if r["label"] != "none")
        P = tp / (tp + fp) if tp + fp else 0
        # recall vs ALL test gaps (pool is a subset if sc-limit)
        allpos = sum(1 for r in te if r["label"] != "none")
        R = tp / allpos
        sc_summary = dict(judged=filt.n_judged, calls=filt.n_calls, kept=len(out),
                          P=P, R=R, F=(2 * P * R / (P + R) if P + R else 0),
                          kept_pos_before=kept_pos_before)
        L.append(f"\n## Stage C ({args.sc_model}, mode=validate)\n")
        L.append(f"- judged {filt.n_judged} in {filt.n_calls} calls; kept {len(out)}/{len(pool)}.")
        L.append(f"- After Stage C: **P={P*100:.1f}% R={R*100:.1f}% F1={sc_summary['F']*100:.1f}%** "
                 f"(precision lifted from {recallfirst['P']*100:.1f}%).")

    L.append("\n### Notes\n- Exact per-sentence metrics (gold-aligned); no token matching.\n"
             "- recall-first = Stage A tuned high-recall; low precision is BY DESIGN — Stage C restores it.\n"
             "- junk-filter = free regex + biomed boilerplate sections; no GROBID/sections needed.")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nwrote", DOC, "and", CAND)


if __name__ == "__main__":
    main()
