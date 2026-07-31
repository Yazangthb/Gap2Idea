"""Benchmark the funnel against the CLEAN gap-sentence gold (paper -> gap sents).

Stage A is judged purely on RECALL: of the gold gap sentences, how many land in
the slice (token-containment, scramble-robust). It is the ceiling for the funnel,
so it must be high. Stage B is then judged on classifying the localized gaps.

    python scripts/dataset/build_gap_gold.py            # make the gold (once)
    python scripts/bench/bench_gap_recall.py --head data/gap_head_bge.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    cue_label, extract_gaps, slice_terminal_regions, token_containment, _looks_like_sentence,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402

TYPES = ["limitation", "future_work"]
LOC_TAU, MATCH_TAU = 0.80, 0.80
ROOT = Path(__file__).resolve().parents[2]


def load():
    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences.tsv", sep="\t", dtype={"paper_id": str})
    mani = pd.read_csv(ROOT / "data/bench_gold/papers_manifest.tsv", sep="\t", dtype=str)
    texts = {}
    for src in {str(s) for s in mani["source"]}:
        for line in (ROOT / src).read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("id")) in set(mani["id"]):
                    texts[str(r.get("id"))] = {"text": str(r.get("text", "")),
                                               "blocks": r.get("blocks") if isinstance(r.get("blocks"), list) else None}
    return gold, texts


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def stage_a(gold, texts):
    print("\n" + "=" * 66 + "\nSTAGE A — recall of gold gap sentences in the slice\n" + "=" * 66)
    rows, audit, sl, fu = [], [], 0, 0
    for pid, rec in texts.items():
        regions = slice_terminal_regions(rec["text"], blocks=rec["blocks"])
        st = " ".join(s for r in regions for s in r.sentences)
        sl += sum(len(r.sentences) for r in regions)
        fu += len(split_sentences(_cut_before_references(rec["text"])))
        for _, g in gold[gold["paper_id"] == pid].iterrows():
            audit.append({"gap_id": g["gap_id"], "gap_type": g["gap_type"],
                          "containment": round(token_containment(g["gap_sentence"], st), 3)})
    adf = pd.DataFrame(audit)
    for tau in (0.90, 0.80, 0.70):
        hit = adf["containment"] >= tau
        rows.append({"tau": tau, "recall_all": round(hit.mean(), 3),
                     **{f"recall_{t}": round(hit[adf.gap_type == t].mean(), 3) for t in TYPES}})
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\n  load: full={fu} sliced={sl}  drop_rate={1 - sl/fu:.3f}")
    miss = adf[adf["containment"] < LOC_TAU].sort_values("containment")
    if len(miss):
        print(f"\n  MISSES (<{LOC_TAU}):")
        for _, m in miss.iterrows():
            print(f"    {m.gap_id:>16} {m.gap_type:<11} cont={m.containment}")
    adf.to_csv(ROOT / "data/bench_gap/stage_a_audit.tsv", sep="\t", index=False)
    return adf


def stage_b(gold, texts, head, thr):
    print("\n" + "=" * 66 + "\nSTAGE B — classify slice sentences (vs gold)\n" + "=" * 66)
    sents, glab = [], []
    for pid, rec in texts.items():
        gp = gold[gold["paper_id"] == pid]
        for r in slice_terminal_regions(rec["text"], blocks=rec["blocks"]):
            for s in r.sentences:
                best_t, best = None, 0.0
                for _, g in gp.iterrows():
                    c = max(token_containment(g["gap_sentence"], s), token_containment(s, g["gap_sentence"]))
                    if c > best:
                        best, best_t = c, g["gap_type"]
                sents.append(s)
                glab.append(best_t if best >= LOC_TAU else "none")
    print(f"  slice sentences: {len(sents)}   gold-gap sentences in slice: {sum(x!='none' for x in glab)}")
    modes = ["rules"] + (["model", "hybrid"] if head else [])
    rows = []
    for mode in modes:
        pred = _classify(sents, mode, head, thr)
        for t in TYPES + ["ANY_GAP"]:
            if t == "ANY_GAP":
                tp = sum(g != "none" and p != "none" for g, p in zip(glab, pred))
                fp = sum(g == "none" and p != "none" for g, p in zip(glab, pred))
                fn = sum(g != "none" and p == "none" for g, p in zip(glab, pred))
            else:
                tp = sum(g == t and p == t for g, p in zip(glab, pred))
                fp = sum(p == t and g != t for g, p in zip(glab, pred))
                fn = sum(g == t and p != t for g, p in zip(glab, pred))
            P, R, F = _prf(tp, fp, fn)
            rows.append({"mode": mode, "type": t, "tp": tp, "fp": fp, "fn": fn,
                         "precision": P, "recall": R, "f1": F})
    print(pd.DataFrame(rows).to_string(index=False))


def _classify(sents, mode, head, thr):
    if mode == "rules":
        return [cue_label(s) or "none" for s in sents]
    preds = head.predict(sents) if head else [("none", 0.0)] * len(sents)
    if mode == "model":
        return [(l if l in TYPES and p >= thr and _looks_like_sentence(s) else "none")
                for s, (l, p) in zip(sents, preds)]
    out = []
    for s, (l, p) in zip(sents, preds):
        rt = cue_label(s)
        if rt:
            out.append(rt)
        elif l in TYPES and p >= thr and _looks_like_sentence(s):
            out.append(l)
        else:
            out.append("none")
    return out


def end_to_end(gold, texts, head, thr):
    print("\n" + "=" * 66 + "\nEND-TO-END — funnel vs gold (recall + type-acc)\n" + "=" * 66)
    modes = ["rules"] + (["model", "hybrid"] if head else [])
    rows = []
    for mode in modes:
        matched, type_ok, npred = set(), 0, 0
        preds = {pid: extract_gaps(pid, rec["text"], blocks=rec["blocks"],
                                   head=head if mode != "rules" else None, mode=mode)
                 for pid, rec in texts.items()}
        npred = sum(len(v) for v in preds.values())
        for _, g in gold.iterrows():
            for pr in preds.get(g["paper_id"], []):
                if max(token_containment(g["gap_sentence"], pr["gap_sentence"]),
                       token_containment(pr["gap_sentence"], g["gap_sentence"])) >= MATCH_TAU:
                    matched.add(g["gap_id"])
                    if pr["gap_type"] == g["gap_type"]:
                        type_ok += 1
                    break
        rows.append({"mode": mode, "n_pred": npred, "gold": len(gold),
                     "matched": len(matched), "recall": round(len(matched)/len(gold), 3),
                     "type_acc": round(type_ok/max(1, len(matched)), 3)})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    gold, texts = load()
    print(f"gold: {len(gold)} gap sentences / {gold['paper_id'].nunique()} papers   "
          f"{gold['gap_type'].value_counts().to_dict()}")
    head = None
    if a.head and a.head.exists():
        from gap2idea.pipeline.gap_funnel import EmbeddingGapHead
        head = EmbeddingGapHead.load(a.head)
    stage_a(gold, texts)
    stage_b(gold, texts, head, a.threshold)
    end_to_end(gold, texts, head, a.threshold)


if __name__ == "__main__":
    main()
