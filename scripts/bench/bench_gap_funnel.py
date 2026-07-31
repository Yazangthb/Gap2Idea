"""Minimal, staged benchmark for the cheap gap-extraction funnel.

Gold = data/bench_gold/gaps_full_flat.tsv (27 future_work/limitation gaps over
10 papers, extracted from the FULL paper by gpt-4o — the expensive teacher the
funnel approximates). We evaluate stage by stage so a regression is localized:

  Stage A (slice)      localization recall  = of gold gaps, fraction whose
                       sentence is contained in the sliced region (token-
                       containment >=0.9, NOT substring — source PDFs are
                       reading-order-scrambled, so 19/27 gold gaps are non-
                       contiguous). + load (sentences kept) = the cost proxy.
  Stage B (classify)   per-sentence precision/recall/type-acc on slice
                       sentences (rules vs model vs hybrid).
  End-to-end           per-type P/R/F1, macro-F1, type-acc vs gold; reported on
                       ALL gold and on OPEN-only (the funnel's intended target —
                       it deliberately ignores resolved_in_paper discourse gaps).
  Cost                 $/1M papers projection vs the per-paper LLM baseline.

BENCHMARK VALIDITY is printed up front: leakage check (train ∩ eval == ∅ via the
head's meta sidecar), gold composition, and the known precision caveat.

Usage:
    python scripts/bench/bench_gap_funnel.py --head data/gap_head.joblib
    python scripts/bench/bench_gap_funnel.py            # rules-only (no head needed)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    cue_label,
    extract_gaps,
    slice_terminal_regions,
    token_containment,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402
from gap2idea.utils import get_logger  # noqa: E402

log = get_logger(__name__)

TARGET_TYPES = ["limitation", "future_work"]
LOCALIZE_TAU = 0.90   # token-containment to call a gap "inside the slice"
MATCH_TAU = 0.85      # token-containment (either direction) to call pred==gold
OPEN_STATUSES = {"open", "partially_addressed"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_gold(root: Path) -> pd.DataFrame:
    g = pd.read_csv(root / "data" / "bench_gold" / "gaps_full_flat.tsv",
                    sep="\t", dtype={"paper_id": str})
    return g


def load_texts(root: Path) -> dict[str, dict]:
    """paper_id -> {text, blocks} for the gold papers, from the manifest sources."""
    mani = pd.read_csv(root / "data" / "bench_gold" / "papers_manifest.tsv",
                       sep="\t", dtype=str)
    targets = set(mani["id"])
    src_files = {str(s) for s in mani["source"]}
    out: dict[str, dict] = {}
    for src in src_files:
        p = root / src
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = str(rec.get("id"))
            if rid in targets:
                out[rid] = {"text": str(rec.get("text", "")),
                            "blocks": rec.get("blocks") if isinstance(rec.get("blocks"), list) else None}
    return out


# ---------------------------------------------------------------------------
# Matching helpers (scrambling-robust)
# ---------------------------------------------------------------------------
def matches(gold_sent: str, pred_sent: str) -> bool:
    return (token_containment(gold_sent, pred_sent) >= MATCH_TAU
            or token_containment(pred_sent, gold_sent) >= MATCH_TAU)


def best_gold_type(sent: str, gold_rows: pd.DataFrame) -> str | None:
    """Per-sentence gold label for a slice sentence: the type of the gold gap it
    matches, else None."""
    best_t, best_s = None, 0.0
    for _, gr in gold_rows.iterrows():
        s = max(token_containment(str(gr["gap_sentence"]), sent),
                token_containment(sent, str(gr["gap_sentence"])))
        if s > best_s:
            best_s, best_t = s, str(gr["gap_type"])
    return best_t if best_s >= LOCALIZE_TAU else None


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


# ---------------------------------------------------------------------------
# Validity report
# ---------------------------------------------------------------------------
def validity_report(root: Path, gold: pd.DataFrame, head_path: Path | None) -> None:
    print("\n" + "=" * 70)
    print("BENCHMARK VALIDITY")
    print("=" * 70)
    gold_ids = set(gold["paper_id"])
    # 1) leakage
    meta_p = head_path.with_suffix(".meta.json") if head_path else None
    if meta_p and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        train_ids = set(meta.get("train_ids", []))
        leak = gold_ids & train_ids
        print(f"[leakage] gold papers: {len(gold_ids)}  train papers: {len(train_ids)}  "
              f"overlap: {len(leak)}  -> {'FAIL ' + str(leak) if leak else 'OK (disjoint)'}")
        assert not leak, f"LEAKAGE: {leak}"
    else:
        print("[leakage] no head meta sidecar (rules-only run) — no train set to leak")
    # 2) gold composition
    print(f"[gold] {len(gold)} gaps / {gold['paper_id'].nunique()} papers")
    print("       gap_type      :", gold["gap_type"].value_counts().to_dict())
    print("       resolution    :", gold["resolution_status"].value_counts().to_dict())
    print("       provenance    :", gold["provenance"].value_counts().to_dict())
    n_open = int(gold["resolution_status"].isin(OPEN_STATUSES).sum())
    print(f"       OPEN target   : {n_open}/{len(gold)} "
          f"(resolved_in_paper excluded from the open-only eval)")
    # 3) caveats
    print("[caveat] precision vs this gold is an UPPER-BOUND proxy: gpt-4o was "
          "selective,\n         so a true gap it skipped counts as a false positive. "
          "Recall + type-acc\n         are the trustworthy axes; precision is a floor.")
    print("[metric] localization & matching use token-CONTAINMENT (>=0.90/0.85), "
          "not substring,\n         because 19/27 gold gaps are PDF-scrambled "
          "(non-contiguous).")


# ---------------------------------------------------------------------------
# Stage A
# ---------------------------------------------------------------------------
def bench_stage_a(gold: pd.DataFrame, texts: dict[str, dict], out_dir: Path) -> None:
    print("\n" + "=" * 70)
    print("STAGE A — structural slice (localization recall + load)")
    print("=" * 70)
    # Per-gap containment against the slice (one slice per paper, default gate).
    audit = []
    slice_sents_tot = full_sents_tot = 0
    for pid, rec in texts.items():
        regions = slice_terminal_regions(rec["text"], blocks=rec["blocks"])
        slice_text = " ".join(s for r in regions for s in r.sentences)
        slice_sents_tot += sum(len(r.sentences) for r in regions)
        full_sents_tot += len(split_sentences(_cut_before_references(rec["text"])))
        for _, gr in gold[gold["paper_id"] == pid].iterrows():
            audit.append({"gap_id": gr["gap_id"], "gap_type": gr["gap_type"],
                          "resolution_status": gr["resolution_status"],
                          "containment": round(token_containment(str(gr["gap_sentence"]), slice_text), 3)})
    adf = pd.DataFrame(audit)
    adf.to_csv(out_dir / "funnel_stage_a_audit.tsv", sep="\t", index=False)

    # Localization recall is reported at several containment thresholds because
    # 19/27 gold gaps are PDF-scrambled: at 0.90 a word scattered into another
    # column drops the gap even though its content is in the slice.
    open_mask = adf["resolution_status"].isin(OPEN_STATUSES)
    rows = []
    for tau in (0.90, 0.80, 0.75):
        hit = adf["containment"] >= tau
        rows.append({
            "containment_tau": tau,
            "loc_recall_all": round(hit.mean(), 3),
            "loc_recall_open": round(hit[open_mask].mean(), 3),
            **{f"rec_{t}": round(hit[adf["gap_type"] == t].mean(), 3) for t in TARGET_TYPES},
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    drop = 1 - slice_sents_tot / max(1, full_sents_tot)
    print(f"\n  load: full={full_sents_tot} sliced={slice_sents_tot} sentences  "
          f"drop_rate={drop:.3f} (FREE filtering before any model)")
    print("  -> Stage A is the recall ceiling. Misses at 0.90 that clear 0.75 are "
          "scramble\n     artifacts (content present, words scattered). See "
          "funnel_stage_a_audit.tsv.")
    df.to_csv(out_dir / "funnel_stage_a.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# Stage B
# ---------------------------------------------------------------------------
def _classify_sentences(sents, mode, head, thr):
    if mode == "rules":
        return [(cue_label(s) or "none") for s in sents]
    preds = head.predict(sents) if (head and sents) else [("none", 0.0)] * len(sents)
    if mode == "model":
        return [(lbl if (lbl in TARGET_TYPES and p >= thr) else "none") for lbl, p in preds]
    # hybrid: rule type wins, else model
    out = []
    for s, (lbl, p) in zip(sents, preds):
        rt = cue_label(s)
        if rt:
            out.append(rt)
        elif lbl in TARGET_TYPES and p >= thr:
            out.append(lbl)
        else:
            out.append("none")
    return out


def bench_stage_b(gold, texts, head, thr, out_dir) -> None:
    print("\n" + "=" * 70)
    print("STAGE B — per-sentence classification on the slice")
    print("=" * 70)
    # build slice sentences + per-sentence gold labels (once)
    sents, glabels = [], []
    for pid, rec in texts.items():
        regions = slice_terminal_regions(rec["text"], blocks=rec["blocks"])
        gp = gold[gold["paper_id"] == pid]
        for r in regions:
            for s in r.sentences:
                sents.append(s)
                glabels.append(best_gold_type(s, gp) or "none")
    gold_pos = sum(1 for x in glabels if x != "none")
    print(f"  slice sentences: {len(sents)}   gold-gap sentences in slice: {gold_pos}")

    modes = ["rules"] + (["model", "hybrid"] if head else [])
    rows = []
    for mode in modes:
        pred = _classify_sentences(sents, mode, head, thr)
        for t in TARGET_TYPES:
            tp = sum(1 for g, p in zip(glabels, pred) if g == t and p == t)
            fp = sum(1 for g, p in zip(glabels, pred) if p == t and g != t)
            fn = sum(1 for g, p in zip(glabels, pred) if g == t and p != t)
            P, R, F = _prf(tp, fp, fn)
            rows.append({"mode": mode, "type": t, "tp": tp, "fp": fp, "fn": fn,
                         "precision": P, "recall": R, "f1": F})
        # gap-vs-none
        tp = sum(1 for g, p in zip(glabels, pred) if g != "none" and p != "none")
        fp = sum(1 for g, p in zip(glabels, pred) if g == "none" and p != "none")
        fn = sum(1 for g, p in zip(glabels, pred) if g != "none" and p == "none")
        P, R, F = _prf(tp, fp, fn)
        rows.append({"mode": mode, "type": "ANY_GAP", "tp": tp, "fp": fp, "fn": fn,
                     "precision": P, "recall": R, "f1": F})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(out_dir / "funnel_stage_b.tsv", sep="\t", index=False)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------
def bench_end_to_end(gold, texts, head, thr, out_dir) -> None:
    print("\n" + "=" * 70)
    print("END-TO-END — funnel output vs gold")
    print("=" * 70)
    modes = ["rules"] + (["model", "hybrid"] if head else [])
    all_rows, summary = [], []
    for mode in modes:
        # gather predictions per paper
        preds_by_paper = {}
        n_pred = 0
        for pid, rec in texts.items():
            preds = extract_gaps(pid, rec["text"], blocks=rec["blocks"],
                                 head=head if mode != "rules" else None, mode=mode,
                                 model_threshold=thr)
            preds_by_paper[pid] = preds
            n_pred += len(preds)

        # a prediction is a "hard FP" only if it matches NO gold gap of ANY status
        # (matching a resolved/partial gold is not a hallucination; precision here
        # is still a floor because gpt-4o gold is selective).
        pred_matches_any = {}
        for pid, prs in preds_by_paper.items():
            for k, pr in enumerate(prs):
                pred_matches_any[(pid, k)] = any(
                    matches(str(gr["gap_sentence"]), str(pr["gap_sentence"]))
                    for _, gr in gold[gold["paper_id"] == pid].iterrows())

        for scope in ["all", "open"]:
            gsub = gold if scope == "all" else gold[gold["resolution_status"].isin(OPEN_STATUSES)]
            matched_gold = set()
            type_correct = 0
            for _, gr in gsub.iterrows():
                pid = gr["paper_id"]
                for k, pr in enumerate(preds_by_paper.get(pid, [])):
                    if matches(str(gr["gap_sentence"]), str(pr["gap_sentence"])):
                        matched_gold.add(gr["gap_id"])
                        if pr["gap_type"] == gr["gap_type"]:
                            type_correct += 1
                        break
            for t in TARGET_TYPES:
                gold_t = gsub[gsub["gap_type"] == t]
                tp = sum(1 for _, gr in gold_t.iterrows()
                         if gr["gap_id"] in matched_gold
                         and any(pr["gap_type"] == t
                                 for pr in preds_by_paper.get(gr["paper_id"], [])
                                 if matches(str(gr["gap_sentence"]), str(pr["gap_sentence"]))))
                fn = len(gold_t) - tp
                # FP = predictions of type t that match no gold gap of any status
                fp = sum(1 for pid, prs in preds_by_paper.items()
                         for k, pr in enumerate(prs)
                         if pr["gap_type"] == t and not pred_matches_any[(pid, k)])
                P, R, F = _prf(tp, fp, fn)
                all_rows.append({"mode": mode, "scope": scope, "type": t,
                                 "tp": tp, "fp": fp, "fn": fn,
                                 "precision_floor": P, "recall": R, "f1": F})
            macro_f1 = round(sum(r["f1"] for r in all_rows
                                 if r["mode"] == mode and r["scope"] == scope) / len(TARGET_TYPES), 3)
            gap_recall = round(len(matched_gold) / max(1, len(gsub)), 3)
            type_acc = round(type_correct / max(1, len(matched_gold)), 3)
            summary.append({"mode": mode, "scope": scope, "n_pred": n_pred,
                            "gold": len(gsub), "matched": len(matched_gold),
                            "gap_recall": gap_recall, "type_acc": type_acc,
                            "macro_f1_floor": macro_f1})

    print("\n--- per-type ---")
    print(pd.DataFrame(all_rows).to_string(index=False))
    print("\n--- summary ---")
    sdf = pd.DataFrame(summary)
    print(sdf.to_string(index=False))
    print("\n  baseline (per-paper LLM, current pipeline, on the OTHER bench): macro-F1 0.187")
    pd.DataFrame(all_rows).to_csv(out_dir / "funnel_e2e_per_type.tsv", sep="\t", index=False)
    sdf.to_csv(out_dir / "funnel_e2e_summary.tsv", sep="\t", index=False)


def cost_projection(texts, head, thr) -> None:
    print("\n" + "=" * 70)
    print("COST PROJECTION (per 1M papers)")
    print("=" * 70)
    slice_sents = full_sents = 0
    for pid, rec in texts.items():
        regions = slice_terminal_regions(rec["text"], blocks=rec["blocks"])
        slice_sents += sum(len(r.sentences) for r in regions)
        full_sents += len(split_sentences(_cut_before_references(rec["text"])))
    n = len(texts)
    sps, fps = slice_sents / n, full_sents / n
    # only slice sentences are embedded (Stage A is free regex)
    toks = sps * 25  # ~25 tokens/sentence avg
    print(f"  mean sentences/paper: full={fps:.0f}  sliced={sps:.0f}  "
          f"(Stage A drops {100*(1-sps/fps):.0f}% before any model)")
    print(f"  embedded tokens/paper ~= {toks:.0f}  -> {toks*1e6/1e6:.1f}M tokens/1M papers")
    print(f"  embedding @ ~$0.00 (local CPU) or ~$0.02/1M tok cloud -> "
          f"~${toks*1e6/1e6*0.02:.0f} / 1M papers")
    print(f"  vs per-paper LLM (~9K tok @ gpt-4.1-mini): ~$4000 / 1M papers")
    print(f"  -> ~{4000/max(1,(toks*1e6/1e6*0.02)):.0f}x cheaper, offline, shardable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", type=Path, default=None)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_dir = root / "data" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)

    gold = load_gold(root)
    texts = load_texts(root)
    log.info("Loaded %d gold gaps over %d papers (%d texts found)",
             len(gold), gold["paper_id"].nunique(), len(texts))

    head = None
    if args.head and args.head.exists():
        from gap2idea.pipeline.gap_funnel import EmbeddingGapHead
        head = EmbeddingGapHead.load(args.head)
        log.info("Loaded head from %s", args.head)

    validity_report(root, gold, args.head if head else None)
    bench_stage_a(gold, texts, out_dir)
    bench_stage_b(gold, texts, head, args.threshold, out_dir)
    bench_end_to_end(gold, texts, head, args.threshold, out_dir)
    cost_projection(texts, head, args.threshold)
    print("\nWrote funnel_stage_a/_b, funnel_e2e_per_type/_summary to data/bench/")


if __name__ == "__main__":
    main()
