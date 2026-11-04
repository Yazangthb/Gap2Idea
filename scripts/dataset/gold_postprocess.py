# -*- coding: utf-8 -*-
"""Post-process the gap gold: agreement report (Opus vs Sonnet kappa + SAL-gold check),
flatten to a per-sentence table, and build the human-validation worksheet."""
import sys, json, csv, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
import numpy as np
from sklearn.metrics import cohen_kappa_score
from gold_label_full import load_corpus

GOLD = ROOT / "data" / "gold"
rows = [json.loads(l) for l in (GOLD/"gap_gold.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
corpus = {p["paper_id"]: p for p in load_corpus()}
TYPES = ("limitation", "future_work", "research_gap")
PRIORITY = {"limitation": 0, "research_gap": 1, "future_work": 2}

def typemap(votes_side, n):
    """index -> single type (priority lim>gap>fw) for one judge."""
    m = {}
    for t in TYPES:
        for i in votes_side.get(t, []):
            if i not in m or PRIORITY[t] < PRIORITY[m[i]]: m[i] = t
    return m

# ---- flatten + collect per-sentence arrays ----
flat = []
yO, yS = [], []                      # binary gap (any type) per judge, pooled
yOt = {t: [] for t in TYPES}; ySt = {t: [] for t in TYPES}
for r in rows:
    pid = r["paper_id"]; n = r["n_sent"]; sents = corpus[pid]["sents"]; secs = corpus[pid].get("secs")
    salg = set(r.get("sal_gold") or [])
    O = typemap(r["votes"]["opus"], n); S = typemap(r["votes"]["sonnet"], n)
    for i in range(n):
        ot, st = O.get(i), S.get(i)
        yO.append(int(ot is not None)); yS.append(int(st is not None))
        for t in TYPES:
            yOt[t].append(int(ot == t)); ySt[t].append(int(st == t))
        if ot is None and st is None: continue      # keep only sentences some judge flagged
        agree = ot if (ot == st and ot) else ("gap_type_mismatch" if (ot and st) else "one_sided")
        conf = "high" if (ot == st and ot) else ("med" if (ot and st) else "low")
        flat.append(dict(paper_id=pid, domain=r["domain"], idx=i,
                         section=(secs[i] if secs else None), sentence=sents[i],
                         opus=ot, sonnet=st, agree=agree, confidence=conf,
                         gold_gap=int(ot is not None and st is not None),
                         sal_gold=int(i in salg)))

json.dump(flat, open(GOLD/"gap_gold_sentences.json", "w", encoding="utf-8"), ensure_ascii=False)
with (GOLD/"gap_gold_sentences.jsonl").open("w", encoding="utf-8") as f:
    for r in flat: f.write(json.dumps(r, ensure_ascii=False)+"\n")

# ---- agreement ----
yO, yS = np.array(yO), np.array(yS)
tot = len(yO); bothpos = int((yO & yS).sum()); opos = int(yO.sum()); spos = int(yS.sum())
print(f"== corpus: {len(rows)} papers, {tot} sentences ==")
print(f"gap-positive: Opus {opos} ({opos/tot:.1%}), Sonnet {spos} ({spos/tot:.1%}), BOTH {bothpos}")
print(f"\n== inter-model agreement (Cohen's kappa) ==")
print(f"  any gap : kappa={cohen_kappa_score(yO,yS):.3f}")
for t in TYPES:
    print(f"  {t:13}: kappa={cohen_kappa_score(yOt[t],ySt[t]):.3f}")

# ---- SAL validity: ensemble(both-agree) vs SAL gold ----
sal = [r for r in rows if r["domain"]=="biomed_sal"]
tp_lim=fp_lim=fn=tp_any=0; goldN=0
for r in sal:
    salg=set(r.get("sal_gold") or []); goldN+=len(salg)
    both_lim=set(r["agree"]["limitation"]); both_any=set(r["gap_any_agree"])
    tp_lim+=len(both_lim&salg); fp_lim+=len(both_lim-salg); fn+=len(salg-both_lim)
    tp_any+=len(salg&both_any)
P=tp_lim/(tp_lim+fp_lim) if tp_lim+fp_lim else 0; R=tp_lim/(tp_lim+fn) if tp_lim+fn else 0
print(f"\n== SAL validity (40 papers, {goldN} SAL gold limitations) ==")
print(f"  ensemble-both limitation vs SAL gold: P={P:.3f} R={R:.3f}")
print(f"  SAL gold caught as ANY agreed gap type (recall): {tp_any/goldN:.3f}")

# ---- human-validation worksheet: stratified ~200 ----
random.seed(7)
both_pos=[r for r in flat if r["gold_gap"]==1]
disagree=[r for r in flat if r["confidence"]=="low"]          # one-sided (the informative cases)
mismatch=[r for r in flat if r["agree"]=="gap_type_mismatch"]
# negatives: sentences NEITHER judge flagged -> sample from corpus
negs=[]
for r in rows:
    pid=r["paper_id"]; sents=corpus[pid]["sents"]; secs=corpus[pid].get("secs")
    O=typemap(r["votes"]["opus"],r["n_sent"]); S=typemap(r["votes"]["sonnet"],r["n_sent"])
    flagged=set(O)|set(S)
    for i in range(r["n_sent"]):
        if i not in flagged: negs.append(dict(paper_id=pid,domain=r["domain"],idx=i,
            section=(secs[i] if secs else None),sentence=sents[i],opus=None,sonnet=None,
            agree="none",confidence="neg",gold_gap=0,sal_gold=0))
random.shuffle(both_pos); random.shuffle(disagree); random.shuffle(mismatch); random.shuffle(negs)
sample = both_pos[:70] + disagree[:60] + mismatch[:20] + negs[:60]
random.shuffle(sample)
def ctx(pid,i):
    s=corpus[pid]["sents"]; return " ".join(s[j] for j in (i-1,i,i+1) if 0<=j<len(s))
with (GOLD/"human_validation_worksheet.csv").open("w",encoding="utf-8",newline="") as fh:
    w=csv.writer(fh)
    w.writerow(["row","paper_id","domain","section","sentence","context_pm1",
                "opus_label","sonnet_label","ensemble_agree","confidence","sal_gold",
                "human_is_gap(Y/N)","human_type(lim/fw/gap/none)","notes"])
    for n,r in enumerate(sample,1):
        w.writerow([n,r["paper_id"],r["domain"],r.get("section"),r["sentence"],ctx(r["paper_id"],r["idx"]),
                    r["opus"],r["sonnet"],r["agree"],r["confidence"],r["sal_gold"],"","",""])
print(f"\n== artifacts ==")
print(f"  per-sentence gold : data/gold/gap_gold_sentences.jsonl ({len(flat)} flagged sentences)")
print(f"  human worksheet   : data/gold/human_validation_worksheet.csv ({len(sample)} rows: "
      f"{len(both_pos[:70])} both-pos, {len(disagree[:60])} one-sided, {len(mismatch[:20])} type-mismatch, {len(negs[:60])} neg)")
print(f"  high-confidence gaps (both agree same type): {sum(1 for r in flat if r['confidence']=='high')}")
