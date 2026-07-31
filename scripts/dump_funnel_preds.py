"""Dump funnel predictions on the gold papers, flagged by gold-match status.

Feeds the adversarial audit: agents judge whether the UNMATCHED predictions are
real gaps gpt-4o simply didn't extract (true precision) vs hallucinations, and
whether each GOLD gap is correctly labelled / in-scope (gold contamination).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gap2idea.pipeline.gap_funnel import EmbeddingGapHead, extract_gaps, token_containment  # noqa: E402

MATCH_TAU = 0.85
ROOT = Path(__file__).resolve().parents[1]


def _matches(a: str, b: str) -> bool:
    return token_containment(a, b) >= MATCH_TAU or token_containment(b, a) >= MATCH_TAU


def main() -> None:
    import json

    gold = pd.read_csv(ROOT / "data/bench_gold/gaps_full_flat.tsv", sep="\t", dtype={"paper_id": str})
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

    head = EmbeddingGapHead.load(ROOT / "data/gap_head_bge.joblib")
    rows = []
    for mode in ("rules", "hybrid"):
        for pid, rec in texts.items():
            gp = gold[gold["paper_id"] == pid]
            for pr in extract_gaps(pid, rec["text"], blocks=rec["blocks"],
                                   head=head if mode != "rules" else None, mode=mode):
                m = None
                for _, gr in gp.iterrows():
                    if _matches(str(gr["gap_sentence"]), str(pr["gap_sentence"])):
                        m = gr["gap_id"]
                        break
                rows.append({"mode": mode, "paper_id": pid, "pred_type": pr["gap_type"],
                             "section_type": pr["section_type"], "source": pr["source"],
                             "matched_gold_id": m or "", "gap_sentence": pr["gap_sentence"]})
    out = ROOT / "data/bench/funnel_predictions.tsv"
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"wrote {out} ({len(rows)} rows; "
          f"{sum(1 for r in rows if not r['matched_gold_id'])} unmatched)")


if __name__ == "__main__":
    main()
