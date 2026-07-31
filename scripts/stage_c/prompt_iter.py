"""Iterate the strict prompt against a labeled fixture.

Fixture: every gap emitted by current Stage C across v4/v5/v6 (62), labeled
TP/FP based on my audit. Column-garbled sentences are MARKED as `garbled`
(not a prompt failure) and excluded from prompt-precision computation.

Goal: minimal prompt edits that kill the FPs (5 of them) without losing TPs.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))

from gap2idea.pipeline.llm import get_llm_client  # noqa: E402
from test_strict_precision import call_batched  # noqa: E402


DEHYPHEN = re.compile(r"(\w)-\s*(\w)")
def norm_text(t):
    return DEHYPHEN.sub(r"\1\2", re.sub(r"\s+", " ", t))


# Labels: TP = real gap; FP = prompt-fixable false positive; GARB = PDF column garble
# Ordered to match v4/v5/v6 paper_gaps.tsv saved row order.
FIXTURE = [
    # ---- v4 (22) ----
    ("2407.16431", "TP"), ("2407.16431", "TP"), ("2407.16431", "TP"), ("2407.16431", "TP"),
    ("2407.16431", "TP"), ("2407.16431", "TP"),
    ("2407.04841", "FP"), ("2407.04841", "TP"), ("2407.04841", "TP"), ("2407.04841", "TP"),
    ("2508.1594", "TP"),
    ("2411.1977", "TP"),
    ("2601.06185", "TP"), ("2601.06185", "TP"), ("2601.06185", "TP"),
    ("2409.19037", "TP"), ("2409.19037", "TP"),
    ("2404.10102", "TP"),
    ("2407.0638", "GARB"), ("2407.0638", "GARB"), ("2407.0638", "GARB"),
    ("2508.0681", "TP"),
    # ---- v5 (16) ----
    ("2205.00834", "TP"), ("2205.00834", "TP"), ("2205.00834", "TP"), ("2205.00834", "FP"),
    ("2012.04551", "TP"), ("2012.04551", "TP"), ("2012.04551", "TP"),
    ("2212.08837", "FP"), ("2212.08837", "TP"), ("2212.08837", "TP"), ("2212.08837", "TP"),
    ("2212.13902", "TP"),
    ("2010.06408", "TP"), ("2010.06408", "TP"),
    ("2412.09369", "TP"),
    ("2005.14425", "TP"),
    # ---- v6 (24) ----
    ("2403.09017", "FP"), ("2403.09017", "TP"), ("2403.09017", "TP"), ("2403.09017", "TP"), ("2403.09017", "TP"),
    ("2511.17323", "GARB"), ("2511.17323", "GARB"), ("2511.17323", "GARB"),
    ("2503.21902", "TP"), ("2503.21902", "TP"),
    ("2008.09911", "TP"),
    ("2203.15386", "TP"), ("2203.15386", "TP"), ("2203.15386", "TP"), ("2203.15386", "TP"),
    ("2203.15386", "TP"), ("2203.15386", "TP"), ("2203.15386", "FP"),
    ("2004.13148", "TP"),
    ("2002.09677", "TP"), ("2002.09677", "TP"), ("2002.09677", "TP"), ("2002.09677", "TP"),
    ("2002.09677", "TP"),
]


def load_papers():
    papers = {}
    for ds in ("dataset_v4", "dataset_v5", "dataset_v6"):
        for jf in [f"papers_v{ds[-1]}.jsonl"]:
            p = ROOT / f"data/{ds}/{jf}"
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                rec["_norm_text"] = norm_text(rec["text"])
                papers[rec["id"]] = rec
    return papers


def find_ctx(sent, ntext, w=50):
    s = sent.strip()[:80]
    idx = ntext.find(s)
    if idx < 0:
        return "", ""
    end = idx + len(sent)
    b = " ".join(ntext[max(0, idx-600):idx].split()[-w:])
    a = " ".join(ntext[end:end+600].split()[:w])
    return b, a


def load_fixture(papers):
    """Build the (sentence, before, after, paper_id, label) fixture from the
    saved paper_gaps.tsv files in v4/v5/v6, aligned with FIXTURE labels."""
    rows = []
    for ds, fname in [("v4", "paper_gaps.tsv"), ("v5", "paper_gaps_v2.tsv"), ("v6", "paper_gaps.tsv")]:
        df = pd.read_csv(ROOT / f"data/dataset_{ds}/{fname}", sep="\t", dtype={"paper_id": str}).fillna("")
        for _, r in df.iterrows():
            pid = str(r["paper_id"]); s = r["gap_sentence"]
            ntext = papers.get(pid, {}).get("_norm_text", "")
            b, a = find_ctx(s, ntext)
            rows.append({"paper_id": pid, "gap_sentence": s, "before": b, "after": a, "ds": ds})
    if len(rows) != len(FIXTURE):
        print(f"WARNING: fixture mismatch — saved gaps={len(rows)} labels={len(FIXTURE)}")
    items, labels = [], []
    for i, row in enumerate(rows):
        if i >= len(FIXTURE):
            break
        pid_lab, lab = FIXTURE[i]
        if row["paper_id"] != pid_lab:
            print(f"  mismatch at #{i}: row pid={row['paper_id']!r} vs label pid={pid_lab!r}")
        items.append((row["gap_sentence"], row["before"], row["after"], row["paper_id"]))
        labels.append(lab)
    return items, labels


def metrics(preds, labels):
    """preds[i] = True if prompt kept item i. labels[i] in {TP, FP, GARB}."""
    real_tp = real_fp = lost_tp = caught_fp = garb_kept = 0
    for keep, lab in zip(preds, labels):
        if lab == "TP":
            if keep: real_tp += 1
            else: lost_tp += 1
        elif lab == "FP":
            if keep: real_fp += 1
            else: caught_fp += 1
        elif lab == "GARB":
            if keep: garb_kept += 1
    n_tp = sum(1 for l in labels if l == "TP")
    n_fp = sum(1 for l in labels if l == "FP")
    return {
        "kept_tp": real_tp, "lost_tp": lost_tp, "tp_recall": real_tp/n_tp,
        "kept_fp": real_fp, "caught_fp": caught_fp, "fp_caught_rate": caught_fp/n_fp,
        "garb_kept": garb_kept,
    }


def main():
    if len(sys.argv) < 2:
        print("usage: prompt_iter.py <prompt_file.txt>")
        sys.exit(2)
    sys_text = Path(sys.argv[1]).read_text(encoding="utf-8")
    papers = load_papers()
    items, labels = load_fixture(papers)
    print(f"Fixture: {len(items)} items "
          f"({sum(1 for l in labels if l=='TP')} TP, "
          f"{sum(1 for l in labels if l=='FP')} FP, "
          f"{sum(1 for l in labels if l=='GARB')} GARB)")
    client = get_llm_client()
    t0 = time.time()
    preds = call_batched(client, "openai/gpt-4o", sys_text, items, label_yes="GAP", label_no="JUNK")
    print(f"  judged in {time.time()-t0:.1f}s")
    m = metrics(preds, labels)
    print(f"\n=== {Path(sys.argv[1]).name} ===")
    print(f"  TP kept:     {m['kept_tp']}/{m['kept_tp']+m['lost_tp']}  (recall {m['tp_recall']:.3f})")
    print(f"  FP caught:   {m['caught_fp']}/{m['caught_fp']+m['kept_fp']}  (rate {m['fp_caught_rate']:.3f})")
    print(f"  GARB kept:   {m['garb_kept']} (PDF-level, not prompt)")
    if m['lost_tp']:
        print(f"\n  -- LOST TPs --")
        for keep, lab, (s,_,_,pid) in zip(preds, labels, items):
            if lab == "TP" and not keep:
                print(f"    [{pid}] {s[:120]!r}")
    if m['kept_fp']:
        print(f"\n  -- KEPT FPs --")
        for keep, lab, (s,_,_,pid) in zip(preds, labels, items):
            if lab == "FP" and keep:
                print(f"    [{pid}] {s[:120]!r}")


if __name__ == "__main__":
    main()
