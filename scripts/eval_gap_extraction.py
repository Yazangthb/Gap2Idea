"""Evaluate the current LLM gap extractor against per-sentence labels.

Inputs:
  data/bench/gaps.tsv          predicted (paper_id, gap_type, gap_sentence, ...)
  data/bench/label_sheet.tsv   labeled (paper_id, sentence, silver_label, gold_label, ...)

Effective label = gold_label if non-empty else silver_label. This lets us run
the metric pass *before* full human adjudication (with `--allow-silver`) and
then re-run later once `gold_label` is filled in.

Matching: for each predicted gap_sentence, find the labeled sentence with the
highest normalised-token Jaccard. If max Jaccard >= --match-tau (default 0.6),
count it as a match to that labeled sentence.

Metrics:
  - per-type precision / recall / F1 (limitation, future_work, open_problem)
  - macro F1 across the 3 gap types
  - sentence-selection accuracy:  of predicted gaps, fraction whose matched
    label is any gap type (vs `none` / no-match → counts as FP)
  - type-classification accuracy: given a selected sentence is a real gap,
    fraction where predicted_type == label
  - confusion matrix (rows = gold label, cols = predicted type; +`miss` row
    for gold gaps the LLM didn't pick, +`hallucination` col for picks that
    matched nothing or matched a `none` label)

Usage:
    python scripts/eval_gap_extraction.py \
        --bench-dir data/bench \
        --allow-silver        # use silver_label when gold_label is empty
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gap2idea.utils import get_logger

log = get_logger(__name__)

GAP_TYPES = ["limitation", "future_work", "open_problem"]
NON_GAP = "none"

_WS_RE = re.compile(r"\s+")
_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(s: str) -> set[str]:
    return set(t.lower() for t in _TOK_RE.findall(s or ""))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _clean(v) -> str:
    """Coerce pandas value to clean string, treating NaN/'nan' as empty."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _effective_label(row: pd.Series, allow_silver: bool) -> str:
    gold = _clean(row.get("gold_label"))
    if gold:
        return gold
    if allow_silver:
        silver = _clean(row.get("silver_label"))
        return silver or NON_GAP
    return ""  # no effective label


def match_predictions(
    gaps: pd.DataFrame,
    labels: pd.DataFrame,
    match_tau: float,
    allow_silver: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the best-matching labeled sentence to each predicted gap.

    Returns (matched_preds, unmatched_gold) where:
      matched_preds = gaps + columns {matched_sent, matched_label, jaccard}
      unmatched_gold = labeled gap sentences that no prediction matched
                       (i.e. FN candidates)
    """
    labels = labels.copy()
    labels["__eff_label"] = labels.apply(lambda r: _effective_label(r, allow_silver), axis=1)
    labels["__tokens"] = labels["sentence"].astype(str).apply(_tokens)

    matched_rows = []
    matched_label_ids: set[tuple[str, int]] = set()

    for _, g in gaps.iterrows():
        pid = str(g["id"])
        pred_tokens = _tokens(str(g.get("gap_sentence") or ""))
        sub = labels[labels["paper_id"].astype(str) == pid]
        best_j, best_idx = 0.0, None
        for li, l in sub.iterrows():
            j = _jaccard(pred_tokens, l["__tokens"])
            if j > best_j:
                best_j, best_idx = j, li
        if best_idx is not None and best_j >= match_tau:
            label_row = labels.loc[best_idx]
            matched_rows.append({
                **g.to_dict(),
                "matched_sent_idx": int(label_row["sent_idx"]),
                "matched_sentence": label_row["sentence"],
                "matched_label": label_row["__eff_label"] or NON_GAP,
                "jaccard": round(best_j, 3),
            })
            matched_label_ids.add((pid, int(label_row["sent_idx"])))
        else:
            matched_rows.append({
                **g.to_dict(),
                "matched_sent_idx": -1,
                "matched_sentence": "",
                "matched_label": "",     # blank → "hallucination" bucket
                "jaccard": round(best_j, 3),
            })

    matched_preds = pd.DataFrame(matched_rows)

    # Unmatched gold gaps: labeled sentences with a gap type that no prediction touched.
    is_gap_label = labels["__eff_label"].isin(GAP_TYPES)
    unmatched_gold_rows = []
    for li, l in labels[is_gap_label].iterrows():
        key = (str(l["paper_id"]), int(l["sent_idx"]))
        if key not in matched_label_ids:
            unmatched_gold_rows.append({
                "paper_id": l["paper_id"],
                "sent_idx": l["sent_idx"],
                "sentence": l["sentence"],
                "gold_type": l["__eff_label"],
            })
    unmatched_gold = pd.DataFrame(unmatched_gold_rows)
    return matched_preds, unmatched_gold


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def compute_metrics(
    matched_preds: pd.DataFrame,
    unmatched_gold: pd.DataFrame,
) -> dict:
    rows = []

    has_unmatched = not unmatched_gold.empty and "gold_type" in unmatched_gold.columns

    # Per-type P/R/F1
    for t in GAP_TYPES:
        tp = int(((matched_preds["gap_type"] == t) & (matched_preds["matched_label"] == t)).sum())
        fp = int(((matched_preds["gap_type"] == t) & (matched_preds["matched_label"] != t)).sum())
        fn_unmatched = int((unmatched_gold["gold_type"] == t).sum()) if has_unmatched else 0
        # FN also includes gold gaps of type t that DID get matched but with a wrong predicted type.
        fn_wrong_type = int(((matched_preds["matched_label"] == t) & (matched_preds["gap_type"] != t)).sum())
        fn = fn_unmatched + fn_wrong_type
        p, r, f1 = _prf(tp, fp, fn)
        rows.append({"type": t, "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3)})

    macro_f1 = round(sum(r["f1"] for r in rows) / len(rows), 3)

    # Sentence-selection: of all predictions, fraction whose match is *any* gap label
    total_preds = len(matched_preds)
    sel_correct = int(matched_preds["matched_label"].isin(GAP_TYPES).sum())
    sel_acc = round(sel_correct / total_preds, 3) if total_preds else 0.0

    # Type classification: given selection was correct, did we get the type right?
    sel_subset = matched_preds[matched_preds["matched_label"].isin(GAP_TYPES)]
    if len(sel_subset):
        type_acc = round((sel_subset["gap_type"] == sel_subset["matched_label"]).mean(), 3)
    else:
        type_acc = 0.0

    # Confusion matrix: rows=matched_label (or 'hallucination' if no match / matched none),
    # cols=predicted gap_type
    cm = pd.crosstab(
        matched_preds["matched_label"].replace("", "hallucination").replace(NON_GAP, "hallucination"),
        matched_preds["gap_type"],
        dropna=False,
    )
    # Add an FN row: gold gaps the LLM never picked
    miss_row = unmatched_gold["gold_type"].value_counts().rename("miss") if has_unmatched else pd.Series(dtype=int)
    if len(miss_row):
        miss_frame = pd.DataFrame(0, index=["miss"], columns=cm.columns)
        for t, v in miss_row.items():
            if t in miss_frame.columns:
                miss_frame.loc["miss", t] = int(v)
            else:
                miss_frame[t] = 0
                miss_frame.loc["miss", t] = int(v)
        cm = pd.concat([cm, miss_frame], axis=0)

    return {
        "per_type": rows,
        "macro_f1": macro_f1,
        "selection_accuracy": sel_acc,
        "type_classification_accuracy": type_acc,
        "n_predictions": total_preds,
        "n_unmatched_gold": int(len(unmatched_gold)),
        "confusion_matrix": cm,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dir", type=Path, default=Path("data/bench"))
    ap.add_argument("--gaps-tsv", type=Path)
    ap.add_argument("--label-sheet", type=Path)
    ap.add_argument("--out-metrics", type=Path)
    ap.add_argument("--match-tau", type=float, default=0.6,
                    help="Min token-Jaccard between predicted gap and labeled sentence to count as a match")
    ap.add_argument("--allow-silver", action="store_true",
                    help="Use silver_label when gold_label is empty (preliminary metrics)")
    args = ap.parse_args()

    gaps_path = args.gaps_tsv or (args.bench_dir / "gaps.tsv")
    label_path = args.label_sheet or (args.bench_dir / "label_sheet.tsv")
    out_path = args.out_metrics or (args.bench_dir / "eval_metrics.tsv")

    if not gaps_path.exists():
        raise FileNotFoundError(gaps_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)

    gaps = pd.read_csv(gaps_path, sep="\t", dtype={"id": str})
    labels = pd.read_csv(label_path, sep="\t", dtype={"paper_id": str, "sent_idx": int})

    log.info("Loaded %d predictions and %d labeled sentences (%d papers)",
             len(gaps), len(labels), labels["paper_id"].nunique())

    matched_preds, unmatched_gold = match_predictions(
        gaps, labels, match_tau=args.match_tau, allow_silver=args.allow_silver,
    )

    m = compute_metrics(matched_preds, unmatched_gold)

    # ---- Print
    print("\n=== Per-type ===")
    print(pd.DataFrame(m["per_type"]).to_string(index=False))
    print(f"\nMacro F1                       : {m['macro_f1']}")
    print(f"Sentence-selection accuracy    : {m['selection_accuracy']}")
    print(f"Type-classification accuracy   : {m['type_classification_accuracy']}")
    print(f"Predictions                    : {m['n_predictions']}")
    print(f"Unmatched gold gaps (misses)   : {m['n_unmatched_gold']}")
    print("\n=== Confusion matrix (rows=gold_label, cols=predicted gap_type) ===")
    print(m["confusion_matrix"].to_string())

    # ---- Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(m["per_type"]).to_csv(out_path, sep="\t", index=False)
    matched_preds.to_csv(out_path.with_name("eval_matched_preds.tsv"), sep="\t", index=False)
    unmatched_gold.to_csv(out_path.with_name("eval_unmatched_gold.tsv"), sep="\t", index=False)
    m["confusion_matrix"].to_csv(out_path.with_name("eval_confusion.tsv"), sep="\t")
    log.info("Wrote %s and 3 companion files (matched_preds, unmatched_gold, confusion)", out_path)


if __name__ == "__main__":
    main()
