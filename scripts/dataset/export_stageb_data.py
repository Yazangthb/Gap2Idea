# -*- coding: utf-8 -*-
"""Export per-sentence train/val/test JSONL for the V100 Stage-B fine-tune.
Same construction as train_gold_head.py: all sentences of each split's papers,
4-class gold label (default 'none'), domain. Self-contained so the box needs no
corpus/loader — just these three files."""
import sys, json
from pathlib import Path
ROOT = Path(r"C:\Users\yazan\OneDrive\Desktop\thesis\Gap2Idea")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from gold_label_full import load_corpus
OUT = Path(r"C:\Users\yazan\AppData\Local\Temp\claude"
           r"\C--Users-yazan-OneDrive-Desktop-thesis-Gap2Idea"
           r"\10eb8776-3b7e-46e0-bfa8-b53a787900b0\scratchpad\stageb_data")
OUT.mkdir(parents=True, exist_ok=True)

gold, domain = {}, {}
for ln in (ROOT / "data/gold/gap_gold_final.jsonl").read_text(encoding="utf-8").splitlines():
    if ln.strip():
        g = json.loads(ln); gold[(g["paper_id"], g["idx"])] = g["label"]; domain[g["paper_id"]] = g["domain"]
corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in domain}
split = json.loads((ROOT / "data/gold/splits.json").read_text(encoding="utf-8"))

for part in ("train", "val", "test"):
    n = 0
    with (OUT / f"{part}.jsonl").open("w", encoding="utf-8") as f:
        for pid in split[part]:
            p = corp[pid]
            for i, s in enumerate(p["sents"]):
                f.write(json.dumps({"sentence": s, "label": gold.get((pid, i), "none"),
                                    "domain": domain[pid], "paper_id": pid, "idx": i},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(part, n, "sentences ->", (OUT / f"{part}.jsonl").name)
print("done:", OUT)
