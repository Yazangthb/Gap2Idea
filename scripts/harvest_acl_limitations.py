"""Harvest clean limitation sentences from the LimGen ACL-Anthology dataset.

LimGen (Faizullah et al., ECML-PKDD 2024, https://github.com/arbmf/LimGen,
CC BY 4.0) released 4068 ACL papers with their MANDATED "Limitations" sections.
A mandated Limitations heading is, by construction, the authors' own-work
self-critique — far cleaner distant-supervision than arbitrary conclusion
sentences. We pull those sections, split to sentences, and use them as extra
`limitation` positives to fix Stage B's data shortage (no LLM, no credits).

LEAKAGE GUARD: drop any harvested sentence that token-matches one of our eval
gold gaps (one gold paper is an NLP paper that could appear in ACL data).

Output: data/acl_limitations.tsv  (src_id, gap_type, gap_sentence)

Usage:
    python scripts/harvest_acl_limitations.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import cue_label, _looks_like_sentence, token_containment  # noqa: E402
from gap2idea.pipeline.gap_prefilter import normalize_text, split_sentences  # noqa: E402

BASE = "https://raw.githubusercontent.com/arbmf/LimGen/main/Datasets/base/"
FILES = ["test.jsonl", "valid.jsonl"]   # ~900 papers; add train.jsonl (89MB) to scale
LEAK_TAU = 0.8


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "curl"})
    return urllib.request.urlopen(req, timeout=120).read()


def main() -> None:
    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences.tsv", sep="\t")
    gold_sents = [str(s) for s in gold["gap_sentence"].tolist()]

    seen: set[str] = set()

    def fresh(s: str) -> bool:
        if not _looks_like_sentence(s):
            return False
        key = normalize_text(s)
        if not key or key in seen:
            return False
        if max((token_containment(g, s) for g in gold_sents), default=0.0) >= LEAK_TAU:
            return False
        seen.add(key)
        return True

    lim_rows, fut_rows, n_papers = [], [], 0
    for fn in FILES:
        print(f"downloading {fn} ...")
        text = fetch(BASE + fn).decode("utf-8", "replace")
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_papers += 1
            pid = f"acl-{rec.get('id')}"
            # (1) limitations — the mandated section (authors' own-work limits)
            lim = str(rec.get("limitations", "") or "")
            if len(lim) >= 40:
                for s in split_sentences(lim):
                    if fresh(s):
                        lim_rows.append({"src_id": pid, "gap_type": "limitation", "gap_sentence": s})
            # (2) future_work — cue-gated sentences from the full paper body
            #     (high precision: only emit where our future-work cue rule fires)
            content = str(rec.get("content", "") or "")
            taken = 0
            for s in split_sentences(content):
                if cue_label(s) == "future_work" and fresh(s):
                    fut_rows.append({"src_id": pid, "gap_type": "future_work", "gap_sentence": s})
                    taken += 1
                    if taken >= 6:   # cap per paper so one doesn't dominate
                        break

    lim_out = ROOT / "data/acl_limitations.tsv"
    fut_out = ROOT / "data/acl_futurework.tsv"
    pd.DataFrame(lim_rows).to_csv(lim_out, sep="\t", index=False)
    pd.DataFrame(fut_rows).to_csv(fut_out, sep="\t", index=False)
    print(f"\npapers={n_papers}  limitation kept={len(lim_rows)}  future_work kept={len(fut_rows)}")
    print(f"wrote {lim_out}\nwrote {fut_out}")
    if lim_rows:
        import textwrap
        print("\nlimitation samples:")
        for r in lim_rows[:3]:
            print("  -", textwrap.shorten(r["gap_sentence"], 100))
        print("future_work samples:")
        for r in fut_rows[:3]:
            print("  -", textwrap.shorten(r["gap_sentence"], 100))


if __name__ == "__main__":
    main()
