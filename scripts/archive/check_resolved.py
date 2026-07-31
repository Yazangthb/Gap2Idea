"""For suspect GT gap sentences, show what FOLLOWS them in the full paper text,
to judge whether the 'gap' is actually left open or resolved in-paper.
"""
from __future__ import annotations
import json, re
from pathlib import Path

papers = {}
for line in (Path("data/bench/bench_papers.jsonl")).read_text(encoding="utf-8").splitlines():
    if line.strip():
        d = json.loads(line)
        papers[str(d["id"])] = d

# (paper_id, distinctive phrase, why-suspect)
checks = [
    ("quant-ph/0402095", "a new difficulty arises", "limitation — 'Oracle Limitations' section"),
    ("quant-ph/0402095", "How does one prove such a statement", "open_problem — rhetorical?"),
    ("quant-ph/0402095", "discovered a flaw in that argument", "limitation — fixed afterward?"),
    ("quant-ph/0402095", "Can quantum computers solve", "open_problem — section opener"),
    # positive control: a sentence from a genuine 'Open Problems' list
    ("0811.3859", "Can we improve the complexity of this reduction", "open_problem — real list?"),
]

def norm_ws(s): return re.sub(r"\s+", " ", s).strip()

for pid, phrase, why in checks:
    text = norm_ws(papers[pid]["full_text"])
    i = text.find(phrase)
    print("=" * 90)
    print(f"[{pid}]  ({why})")
    print(f"  phrase: {phrase!r}  -> {'FOUND' if i>=0 else 'NOT FOUND'} at {i}")
    if i >= 0:
        start = i
        end = min(len(text), i + 750)
        print("  ...what follows (~750 chars):")
        print("   " + text[start:end])
    print()
