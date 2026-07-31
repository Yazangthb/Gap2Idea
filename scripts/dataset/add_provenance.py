"""Backfill provenance/token_recall onto an existing gaps_full.jsonl without
re-calling the LLM. Rewrites gaps_full.jsonl + gaps_full_flat.tsv.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_gold_dataset import _norm, _WORD, _provenance, load_texts, _cut_before_references

OUT = Path("data/bench_gold")
man = pd.read_csv(OUT / "papers_manifest.tsv", sep="\t", dtype=str)
texts = {k: _cut_before_references(v) for k, v in load_texts(man).items()}
body_norm = {k: _norm(v) for k, v in texts.items()}
body_tok = {k: set(_WORD.findall(v)) for k, v in body_norm.items()}

per_paper, flat = [], []
for line in (OUT / "gaps_full.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    pid = str(rec["paper_id"])
    new_gaps = []
    for i, g in enumerate(rec["gaps"]):
        prov, tr, loc = _provenance(g["gap_sentence"], body_norm.get(pid, ""), body_tok.get(pid, set()))
        row = {
            "gap_id": g.get("gap_id", f"{pid}::g{i+1}"),
            "paper_id": pid, "domain": rec.get("domain", ""),
            "gap_type": g.get("gap_type", ""), "resolution_status": g.get("resolution_status", ""),
            "provenance": prov, "token_recall": tr, "location_fraction": loc,
            "gap_sentence": g["gap_sentence"], "paragraph_context": g.get("paragraph_context", ""),
        }
        new_gaps.append(row); flat.append(row)
    per_paper.append({"paper_id": pid, "domain": rec.get("domain", ""), "n_gaps": len(new_gaps), "gaps": new_gaps})

with (OUT / "gaps_full.jsonl").open("w", encoding="utf-8") as f:
    for r in per_paper:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
df = pd.DataFrame(flat)
df.to_csv(OUT / "gaps_full_flat.tsv", sep="\t", index=False)

print("=== provenance ===")
print(df["provenance"].value_counts().to_string())
print(f"\ntoken_recall mean: {df['token_recall'].mean():.3f}  min: {df['token_recall'].min():.3f}")
print("\n=== resolution_status ===")
print(df["resolution_status"].value_counts().to_string())
print("\n=== gap_type ===")
print(df["gap_type"].value_counts().to_string())
print(f"\nlocation_fraction median (exact only): "
      f"{df[df['location_fraction']>=0]['location_fraction'].median():.2f}")
print(f"papers={df['paper_id'].nunique()} gaps={len(df)}")
