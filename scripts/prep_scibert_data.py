"""Prep data for SciBERT funnel test on the GPU box.

Builds the same 3-class training set the bge+logreg head used (runs/* self-
distilled + ACL Limitations harvest, with leakage guards), and packages the 10
gold papers (text + blocks) into one small JSONL — so the GPU box can run the
whole experiment without needing the runs/* corpus.

Outputs:
    data/scibert_prep/train.jsonl    (sentence, label)
    data/scibert_prep/gold_papers.jsonl  (id, text, blocks)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_gap_head import build_dataset, eval_paper_ids  # noqa: E402

OUT = ROOT / "data" / "scibert_prep"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exclude = eval_paper_ids(ROOT)
    extra = ROOT / "data/bench_gap/train/gold_sentences.tsv"
    sents, labels, stats = build_dataset(ROOT, exclude, extra if extra.exists() else None, no_distant=True)

    # add the ACL limitations harvest as extra `limitation` positives (same recipe
    # as the shipped head; cap 1500 saturates per the sweep we did)
    acl = pd.read_csv(ROOT / "data/acl_limitations.tsv", sep="\t").head(1500)
    sents.extend(acl["gap_sentence"].astype(str).tolist())
    labels.extend(["limitation"] * len(acl))

    with (OUT / "train.jsonl").open("w", encoding="utf-8") as f:
        for s, l in zip(sents, labels):
            f.write(json.dumps({"sentence": s, "label": l}, ensure_ascii=False) + "\n")
    print(f"train.jsonl  rows={len(sents)}  "
          f"{pd.Series(labels).value_counts().to_dict()}")

    # 10 gold papers — extract (id, text, blocks) from the manifest sources
    mani = pd.read_csv(ROOT / "data/bench_gold/papers_manifest.tsv", sep="\t", dtype=str)
    gold_ids = set(mani["id"])
    sources = {str(s) for s in mani["source"]}
    n = 0
    with (OUT / "gold_papers.jsonl").open("w", encoding="utf-8") as g:
        for src in sources:
            for line in (ROOT / src).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(r.get("id")) in gold_ids:
                    g.write(json.dumps({"id": r["id"], "text": r.get("text", ""),
                                         "blocks": r.get("blocks", [])},
                                        ensure_ascii=False) + "\n")
                    n += 1
    print(f"gold_papers.jsonl  papers={n}")


if __name__ == "__main__":
    main()
