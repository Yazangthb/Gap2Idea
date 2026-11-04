# -*- coding: utf-8 -*-
"""Build the FINAL gap gold = both-agree-same-type (high conf) + adjudicator-confirmed one-sided.
Recompute recall/precision vs SAL, corpus stats; rewrite the flattened gold + human worksheet."""
import sys, json, csv, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from gold_label_full import load_corpus

GOLD = ROOT / "data" / "gold"
TYPES = ("limitation", "future_work", "research_gap")
PRI = {"limitation": 0, "research_gap": 1, "future_work": 2}
rows = {json.loads(l)["paper_id"]: json.loads(l) for l in (GOLD/"gap_gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
adj = {json.loads(l)["paper_id"]: json.loads(l).get("verdict", {}) for l in (GOLD/"adjudication.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
corpus = {p["paper_id"]: p for p in load_corpus()}

def typemap(side):
    m = {}
    for t in TYPES:
        for i in side.get(t, []):
            if i not in m or PRI[t] < PRI[m[i]]: m[i] = t
    return m

flat = []; kept_adj = 0; dropped_adj = 0
for pid, r in rows.items():
    sents = corpus[pid]["sents"]; secs = corpus[pid].get("secs"); salg = set(r.get("sal_gold") or [])
    O = typemap(r["votes"]["opus"]); S = typemap(r["votes"]["sonnet"])
    both_same = {i: O[i] for i in O if O.get(i) == S.get(i)}
    verdict = {int(k): v for k, v in adj.get(pid, {}).items()}
    final = {}
    for i, t in both_same.items(): final[i] = (t, "high")
    for i, lab in verdict.items():
        if lab in TYPES: final[i] = (lab, "adjudicated"); kept_adj += 1
        else: dropped_adj += 1
    for i in range(r["n_sent"]):
        if i in final:
            t, conf = final[i]
            flat.append(dict(paper_id=pid, domain=r["domain"], idx=i, section=(secs[i] if secs else None),
                             sentence=sents[i], label=t, confidence=conf, sal_gold=int(i in salg)))

# write final gold
with (GOLD/"gap_gold_final.jsonl").open("w", encoding="utf-8") as f:
    for r in flat: f.write(json.dumps(r, ensure_ascii=False) + "\n")

# stats
from collections import Counter
byt = Counter(r["label"] for r in flat); byc = Counter(r["confidence"] for r in flat)
print(f"FINAL gold: {len(flat)} gap sentences across {len(rows)} papers")
print(f"  by type: {dict(byt)}")
print(f"  by confidence: {dict(byc)}  (adjudication kept {kept_adj}, dropped {dropped_adj})")

# SAL eval: final limitation vs SAL gold + binary gap recall
sal_rows = [p for p in flat if p["domain"] == "biomed_sal"]
salgold = 205
by_paper = {}
for r in rows.values():
    if r["domain"] == "biomed_sal": by_paper[r["paper_id"]] = set(r.get("sal_gold") or [])
final_lim = {}; final_gap = {}
for r in sal_rows:
    final_gap.setdefault(r["paper_id"], set()).add(r["idx"])
    if r["label"] == "limitation": final_lim.setdefault(r["paper_id"], set()).add(r["idx"])
tp=fp=fn=tpg=0
for pid, g in by_paper.items():
    lim = final_lim.get(pid, set()); gap = final_gap.get(pid, set())
    tp += len(lim & g); fp += len(lim - g); fn += len(g - lim); tpg += len(g & gap)
P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
print(f"\nSAL validity (final gold, 205 gold limitations):")
print(f"  limitation: P={P:.3f} R={R:.3f}")
print(f"  binary-gap recall of SAL gold: {tpg/salgold:.3f}")

# refreshed human worksheet: stratify high / adjudicated-kept / adjudicated-dropped / negatives
random.seed(11)
hi = [r for r in flat if r["confidence"] == "high"]
ak = [r for r in flat if r["confidence"] == "adjudicated"]
# adjudicated-dropped + negatives from corpus
dropped = []; negs = []
for pid, r in rows.items():
    sents = corpus[pid]["sents"]; secs = corpus[pid].get("secs")
    O = typemap(r["votes"]["opus"]); S = typemap(r["votes"]["sonnet"])
    final_idx = {i for i in O if O.get(i) == S.get(i)} | {int(k) for k, v in adj.get(pid, {}).items() if v in TYPES}
    verdict = {int(k): v for k, v in adj.get(pid, {}).items()}
    flagged = set(O) | set(S)
    for i in range(r["n_sent"]):
        rec = dict(paper_id=pid, domain=r["domain"], idx=i, section=(secs[i] if secs else None), sentence=sents[i])
        if verdict.get(i) == "none": dropped.append(rec)
        elif i not in flagged: negs.append(rec)
for lst in (hi, ak, dropped, negs): random.shuffle(lst)
def ctx(pid, i): s = corpus[pid]["sents"]; return " ".join(s[j] for j in (i-1, i, i+1) if 0 <= j < len(s))
sample = hi[:70] + ak[:60] + dropped[:40] + negs[:40]; random.shuffle(sample)
with (GOLD/"human_validation_worksheet.csv").open("w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["row","paper_id","domain","section","sentence","context_pm1","gold_label","confidence",
                "human_is_gap(Y/N)","human_type(lim/fw/gap/none)","notes"])
    for n, r in enumerate(sample, 1):
        lab = r.get("label", "none(dropped)" if r in dropped else "none")
        conf = r.get("confidence", "adj-dropped" if r in dropped else "neg")
        w.writerow([n, r["paper_id"], r["domain"], r.get("section"), r["sentence"], ctx(r["paper_id"], r["idx"]),
                    lab, conf, "", "", ""])
print(f"\nartifacts: data/gold/gap_gold_final.jsonl ({len(flat)}) + refreshed human_validation_worksheet.csv ({len(sample)} rows)")
