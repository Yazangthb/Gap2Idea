# -*- coding: utf-8 -*-
"""FAITHFUL redo: judge exactly PubMedBERT's own A+B predicted-positives (document order,
batch 20) — the real deployment batch — not a dense 1036-union (whose batch composition
biased the judge stricter; see check: same sentences flipped 56->37 kept).

Gives both:
  (a) full end-to-end A+B+C funnel for PubMedBERT (raw prevalence, 40 papers, all gold)
  (b) prevalence-20 section pool vs their 0.821
with two prompts: NEW keep-biased SAL-criteria, and rct_loose (old/shipped)."""
import os, sys, json, time
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
from gap2idea.pipeline.llm import get_llm_client
from bench_stagec_opt import SYSTEM_RCT_LOOSE, SHOTS_RCT

HERE = Path(__file__).parent
rows = json.load(open(HERE / "sal_test_allmodels.json", encoding="utf-8"))
def window(i, k=1):
    r = rows[i]
    return " ".join(rows[j]["sent"] for j in range(i-k, i+k+1)
                    if 0 <= j < len(rows) and rows[j]["pmcid"] == r["pmcid"])
KEEP = {"DISCUSS", "CONCL"}
N = len(rows); GOLD_ALL = sum(r["label"] for r in rows)
pool = [i for i in range(N) if rows[i]["sec"] in KEEP]            # Stage A
pb = sorted([i for i in pool if rows[i]["preds"]["PubMedBERT-FT"] == 1])  # A+B positives, doc order

from gap2idea.pipeline.stage_c_prompts import SAL_SYS as SYS_SAL, SAL_SHOTS as SHOTS_SAL

CLIENT = get_llm_client("yandex"); F = os.environ["YANDEX_FOLDER_ID"].strip()
MODEL = f"gpt://{F}/yandexgpt-5-lite/latest"

def judge(idxs, sys_p, shots):
    f = LLMGapFilter(backend="api", model=MODEL, mode="validate_rct",
                     client=CLIENT, context_style="inline")
    f._sys, f._shots = sys_p, shots
    keep = {}
    for k in range(0, len(idxs), 20):
        ch = idxs[k:k+20]
        items = [{"sentence": rows[i]["sent"], "context": window(i, 1)} for i in ch]
        for i, kp in zip(ch, f.judge_batch(items)):
            keep[i] = bool(kp)
    return keep, f.n_calls

print(f"PubMedBERT A+B positives (doc order): {len(pb)}  | all gold={GOLD_ALL}\n", flush=True)
t0 = time.time()
V_sal, c1 = judge(pb, SYS_SAL, SHOTS_SAL)
V_rct, c2 = judge(pb, SYSTEM_RCT_LOOSE, SHOTS_RCT)
print(f"judged {len(pb)} x2 prompts in {c1+c2} calls [{time.time()-t0:.0f}s]\n", flush=True)

# ---- (a) end-to-end funnel, raw prevalence, PubMedBERT ----
def efunnel(keep_map=None):
    kept = [i for i in pb if (keep_map is None or keep_map.get(i, True))]
    tp = sum(rows[i]["label"] for i in kept)
    R = tp / GOLD_ALL; P = tp/len(kept) if kept else 0; Fv = 2*P*R/(P+R) if P+R else 0
    return P, R, Fv
print("(a) END-TO-END FUNNEL (PubMedBERT, raw prevalence, 40 papers)")
for tag, km in [("A+B (no Stage C)", None), ("A+B+C  SAL keep-biased", V_sal), ("A+B+C  rct_loose (old)", V_rct)]:
    P, R, Fv = efunnel(km)
    print(f"  {tag:28} P={P:.3f} R={R:.3f} F1={Fv:.3f}")

# ---- (b) prevalence-20 section pool ----
gold = np.array([rows[i]["label"] for i in pool])
det = np.array([1 if rows[i]["preds"]["PubMedBERT-FT"] == 1 else 0 for i in pool])
def apply_C(km):
    out = det.copy()
    for j, i in enumerate(pool):
        if det[j] == 1 and not km.get(i, True):
            out[j] = 0
    return out
def prf(g, p):
    tp = int(((p==1)&(g==1)).sum()); fp = int(((p==1)&(g==0)).sum()); fn = int(((p==0)&(g==1)).sum())
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    return P, R, (2*P*R/(P+R) if P+R else 0)
def pm_f1(g, p, target=0.207, seeds=8):
    pos = np.where(g==1)[0]; neg = np.where(g==0)[0]
    n = min(len(neg), int(round(len(pos)*(1-target)/target)))
    return float(np.mean([prf(g[np.concatenate([pos, np.random.default_rng(s).choice(neg,n,replace=False)])],
                              p[np.concatenate([pos, np.random.default_rng(s).choice(neg,n,replace=False)])])[2]
                          for s in range(seeds)]))
print(f"\n(b) PREVALENCE-20 SECTION POOL | pool={len(pool)} ({int(gold.sum())} lim, {gold.mean():.1%} native)")
print(f"{'config':<30}{'P':>7}{'R':>7}{'F1':>7}{'F1@20.7%':>11}")
print(f"{'their PubMedBERT (reported)':<30}{0.751:>7.3f}{0.907:>7.3f}{0.821:>7.3f}{0.821:>11.3f}")
Pd,Rd,Fd = prf(gold, det); print(f"{'detector only':<30}{Pd:>7.3f}{Rd:>7.3f}{Fd:>7.3f}{pm_f1(gold,det):>11.3f}")
for tag, km in [("+ Stage C (SAL keep-biased)", V_sal), ("+ Stage C (rct_loose, old)", V_rct)]:
    fin = apply_C(km); P,R,Fv = prf(gold, fin); fpm = pm_f1(gold, fin)
    print(f"{tag:<30}{P:>7.3f}{R:>7.3f}{Fv:>7.3f}{fpm:>11.3f}{'  >=0.821' if fpm>=0.821 else ''}")
json.dump({"pb_n": len(pb), "sal": {str(k):v for k,v in V_sal.items()},
           "rct": {str(k):v for k,v in V_rct.items()}}, open(HERE/"faithful_pubmed_verdicts.json","w"))
