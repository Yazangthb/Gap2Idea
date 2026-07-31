"""Validate the Tier-0 lexical prefilter on the bench eval set.

Ground truth = data/bench/label_sheet.tsv, per-sentence labels.
Effective label = gold_label if present else silver_label.
A sentence is a GAP iff its effective label in {limitation, future_work, open_problem}.

Tier 0 flags a sentence as a CANDIDATE if it contains any dictionary phrase.

Reported (the two that matter for a recall-tuned prefilter are **recall** and
**survivor-rate**):
    recall        TP / (TP+FN)   gaps kept  — THE ceiling for Tier 1 (target >=0.95)
    specificity   TN / (TN+FP)   non-gaps correctly dropped (filter-rate)
    survivor_rate (TP+FP) / all  load handed to Tier 1 (lower = cheaper)
    precision     TP / (TP+FP)   expected LOW for Tier 0 — fine
    f1            harmonic mean  (for completeness)

Evaluates both the recall-first `dictionary` and the compact `dictionary_compact`
so the recall/filter-rate trade-off is visible. Also dumps missed gaps (FN).

Usage:
    python scripts/archive/eval_tier0.py --dict data/tier0_dictionary.json --bench-dir data/bench
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.gap_prefilter import GapPrefilter  # noqa: E402
from gap2idea.utils import get_logger  # noqa: E402

log = get_logger(__name__)

GAP_TYPES = {"limitation", "future_work", "open_problem"}


def _clean(v) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def effective_label(row: pd.Series) -> str:
    return _clean(row.get("gold_label")) or _clean(row.get("silver_label"))


def evaluate(pf: GapPrefilter, labels: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    tp = fp = tn = fn = 0
    missed_rows = []
    for _, r in labels.iterrows():
        is_gap = effective_label(r) in GAP_TYPES
        flagged = pf.is_candidate(str(r["sentence"]))
        if is_gap and flagged:
            tp += 1
        elif is_gap and not flagged:
            fn += 1
            missed_rows.append({
                "paper_id": r["paper_id"], "sent_idx": r["sent_idx"],
                "label": effective_label(r), "sentence": r["sentence"],
            })
        elif not is_gap and flagged:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    survivor_rate = (tp + fp) / total if total else 0.0

    metrics = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "survivor_rate": round(survivor_rate, 4),
        "n_gaps": tp + fn,
        "n_sentences": total,
    }
    return metrics, pd.DataFrame(missed_rows)


def sweep_lift(data: dict, labels: pd.DataFrame, target_recall: float = 0.95) -> pd.DataFrame:
    """Sweep a lift threshold over the mined pool and report the recall/survivor
    trade-off, so we can pick the operating point (smallest survivor-rate that
    still clears the recall target). Unigrams must clear an extra +1.0 margin
    because single tokens are inherently less specific than multi-word phrases."""
    max_n = int(data.get("max_n", 3))
    pool = data["dictionary"]
    grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    rows = []
    for t in grid:
        phrases = [
            p["phrase"] for p in pool
            if p["lift"] >= (t + 1.0 if p["n"] == 1 else t)
        ]
        if not phrases:
            continue
        pf = GapPrefilter(phrases=phrases, max_n=max_n)
        m, _ = evaluate(pf, labels)
        rows.append({
            "lift_floor": t, "dict_size": len(phrases),
            "recall": m["recall"], "specificity": m["specificity"],
            "survivor_rate": m["survivor_rate"], "precision": m["precision"],
            "f1": m["f1"], "meets_target": m["recall"] >= target_recall,
        })
    return pd.DataFrame(rows)


def hybrid_sweep(data: dict, labels: pd.DataFrame) -> pd.DataFrame:
    """Drop only phrases that are BOTH weakly discriminative (lift < lift_cut)
    AND frequent in non-gaps (df_neg > dfneg_cut) — i.e. generic fillers like
    'be'/'model'/'our'. Keeps frequent-but-discriminative cues ('future','work').
    This targets the survivor-rate killers without sacrificing cue coverage."""
    max_n = int(data.get("max_n", 3))
    pool = data["dictionary"]
    rows = []
    for lift_cut in (2.0, 2.5, 3.0):
        for dfneg_cut in (40, 60, 80, 120):
            phrases = [
                p["phrase"] for p in pool
                if not (p["lift"] < lift_cut and p["df_neg"] > dfneg_cut)
            ]
            pf = GapPrefilter(phrases=phrases, max_n=max_n)
            m, _ = evaluate(pf, labels)
            rows.append({
                "lift_cut": lift_cut, "dfneg_cut": dfneg_cut,
                "dict_size": len(phrases), "recall": m["recall"],
                "specificity": m["specificity"], "survivor_rate": m["survivor_rate"],
                "precision": m["precision"], "f1": m["f1"],
            })
    return pd.DataFrame(rows).sort_values(["recall", "specificity"], ascending=False)


def per_paper_recall(pf: GapPrefilter, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pid, sub in labels.groupby("paper_id"):
        gaps = sub[sub.apply(lambda r: effective_label(r) in GAP_TYPES, axis=1)]
        if gaps.empty:
            continue
        kept = sum(pf.is_candidate(str(s)) for s in gaps["sentence"])
        rows.append({"paper_id": pid, "n_gaps": len(gaps), "kept": kept,
                     "recall": round(kept / len(gaps), 3)})
    return pd.DataFrame(rows).sort_values("recall")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dict", type=Path, default=Path("data/tier0_dictionary.json"))
    ap.add_argument("--bench-dir", type=Path, default=Path("data/bench"))
    ap.add_argument("--out", type=Path, default=Path("data/bench/tier0_eval.tsv"))
    args = ap.parse_args()

    label_path = args.bench_dir / "label_sheet.tsv"
    labels = pd.read_csv(label_path, sep="\t", dtype={"paper_id": str, "sent_idx": int})
    log.info("Loaded %d labeled sentences (%d papers)", len(labels), labels["paper_id"].nunique())

    data = json.loads(args.dict.read_text(encoding="utf-8"))
    max_n = int(data.get("max_n", 3))

    variants = {
        "pool":    [p["phrase"] for p in data["dictionary"]],
        "compact": [p["phrase"] for p in data.get("dictionary_compact", [])],
    }

    summary = []
    missed_by_variant = {}
    for name, phrases in variants.items():
        if not phrases:
            continue
        pf = GapPrefilter(phrases=phrases, max_n=max_n)
        m, missed = evaluate(pf, labels)
        m["variant"] = name
        m["dict_size"] = len(phrases)
        summary.append(m)
        missed_by_variant[name] = missed

    sdf = pd.DataFrame(summary)[
        ["variant", "dict_size", "recall", "specificity", "survivor_rate",
         "precision", "f1", "tp", "fp", "fn", "tn", "n_gaps", "n_sentences"]
    ]
    print("\n=== Tier 0 on bench eval set ===")
    print(sdf.to_string(index=False))

    # Lift-threshold sweep: find the operating-point knee
    sweep = sweep_lift(data, labels)
    print("\n=== Lift-threshold sweep (unigrams need lift_floor+1.0) ===")
    print(sweep.to_string(index=False))
    ok = sweep[sweep["meets_target"]]
    if not ok.empty:
        best = ok.loc[ok["survivor_rate"].idxmin()]
        print(f"\n>>> Operating point: lift_floor={best['lift_floor']} "
              f"(dict={int(best['dict_size'])}) recall={best['recall']} "
              f"survivor={best['survivor_rate']} specificity={best['specificity']}")
    sweep.to_csv(args.out.with_name("tier0_sweep.tsv"), sep="\t", index=False)

    # Hybrid rule: drop only low-lift AND high-df_neg generic fillers
    hyb = hybrid_sweep(data, labels)
    print("\n=== Hybrid sweep: drop phrases that are low-lift AND high-df_neg ===")
    print(hyb.to_string(index=False))
    hyb95 = hyb[hyb["recall"] >= 0.95]
    if not hyb95.empty:
        best = hyb95.loc[hyb95["survivor_rate"].idxmin()]
        print(f"\n>>> Best hybrid @recall>=0.95: lift_cut={best['lift_cut']} "
              f"dfneg_cut={int(best['dfneg_cut'])} dict={int(best['dict_size'])} "
              f"recall={best['recall']} survivor={best['survivor_rate']} "
              f"specificity={best['specificity']}")
    hyb.to_csv(args.out.with_name("tier0_hybrid_sweep.tsv"), sep="\t", index=False)

    # Per-paper recall for the recall-first pool
    pool_pf = GapPrefilter(phrases=variants["pool"], max_n=max_n)
    ppr = per_paper_recall(pool_pf, labels)
    print("\n=== Per-paper recall (pool) ===")
    print(ppr.to_string(index=False))

    # Missed gaps for the pool
    missed = missed_by_variant.get("pool")
    if missed is not None and not missed.empty:
        print(f"\n=== Missed gaps (pool): {len(missed)} ===")
        for _, r in missed.iterrows():
            print(f"  [{r['paper_id']} #{r['sent_idx']} {r['label']}] {r['sentence'][:130]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sdf.to_csv(args.out, sep="\t", index=False)
    ppr.to_csv(args.out.with_name("tier0_per_paper.tsv"), sep="\t", index=False)
    if missed is not None:
        missed.to_csv(args.out.with_name("tier0_missed.tsv"), sep="\t", index=False)
    log.info("Wrote %s (+ per_paper, missed)", args.out)


if __name__ == "__main__":
    main()
