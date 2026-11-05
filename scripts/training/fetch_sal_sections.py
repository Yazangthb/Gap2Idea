# -*- coding: utf-8 -*-
# Recovered 2026-11-05, code unchanged, from the Oct 8 2026 session. Writes sal_test_sections.json,
# the input of scripts/training/h100_allmodels.py.
"""Reconstruct the structured full papers for SAL test: every sentence with its
recovered BioC section_type + gold label, in order. Enables the end-to-end funnel
(Stage A section-aware -> B -> C) on real papers. Cached."""
import sys, json, csv, io, urllib.request, time
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from build_rct_pool import fetch_rows, bioc_sections, section_of

SPLIT = "test"
cache = ROOT / "data" / "bioc_cache"; cache.mkdir(parents=True, exist_ok=True)
rows = fetch_rows(SPLIT)
pmcids = list(dict.fromkeys(r["pmcids"] for r in rows))
print(f"{SPLIT}: {len(rows)} sentences, {len(pmcids)} papers", flush=True)

sect_cache = {}
for i, pid in enumerate(pmcids):
    sect_cache[pid] = bioc_sections(pid, cache)
    if (i + 1) % 5 == 0:
        print(f"  fetched {i+1}/{len(pmcids)}", flush=True)

out = []
unmatched = 0
for r in rows:
    sent = (r.get("sentences") or "").strip()
    if len(sent.split()) < 3:
        continue
    label = 0 if (r.get("category_spans", "[0]") or "[0]").strip() == "[0]" else 1
    sec = section_of(sent, sect_cache.get(r["pmcids"], []))
    if sec == "?":
        unmatched += 1
    out.append({"pmcid": r["pmcids"], "sent": sent, "sec": sec, "label": label})

from collections import Counter
print("section_type distribution:", dict(Counter(o["sec"] for o in out).most_common()))
print(f"unmatched (sec '?'): {unmatched}/{len(out)}")
# how many gold limitations fall in DISCUSS/CONCL (the Stage-A-keep ceiling)
KEEP = {"DISCUSS", "CONCL"}
pos = sum(o["label"] for o in out); pos_keep = sum(o["label"] for o in out if o["sec"] in KEEP)
print(f"gold limitations: {pos} total | {pos_keep} in DISCUSS/CONCL = {pos_keep/max(pos,1):.1%} (Stage-A recall ceiling)")
dst = Path(__file__).with_name("sal_test_sections.json")
json.dump(out, open(dst, "w", encoding="utf-8"))
print("wrote", dst)
