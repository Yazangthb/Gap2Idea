# -*- coding: utf-8 -*-
"""Cost analysis Part 1 (token accounting, no API): tokens sent to the LLM by the funnel
(Stage C sees only detector-positive sentences + context) vs the full-paper-LLM baseline
(whole paper to the LLM). SAL test, 40 papers. tiktoken cl100k_base (vendor-neutral)."""
import sys, json, math
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, tiktoken
from gap2idea.pipeline.gap_llm_filter import SYSTEM_VALIDATE_RCT, SHOTS_VALIDATE_RCT

ENC = tiktoken.get_encoding("cl100k_base")
def tok(s): return len(ENC.encode(s))

js = json.load(open(ROOT/"stage_c_package"/"sal_criteria"/"sal_test_allmodels.json", encoding="utf-8"))
KEEP = {"DISCUSS", "CONCL"}
# group by paper, preserve order
papers = {}
for i, r in enumerate(js):
    papers.setdefault(r["pmcid"], []).append(i)

SYS_T = tok(SYSTEM_VALIDATE_RCT)
SHOTS_T = sum(tok(s) for s, _ in SHOTS_VALIDATE_RCT) + 4*len(SHOTS_VALIDATE_RCT)
BATCH_OVERHEAD = 40      # JSON-array instruction per call
ITEM_WRAP = 15           # {"sentence":..,"context":..} json keys/quotes per item
FULL_PROMPT = 120        # full-paper extraction instruction
BATCH = 20

def ctx(idxs, k, pos):   # +/-1 context within paper
    out = []
    for j in (k-1, k, k+1):
        if 0 <= j < len(idxs): out.append(js[idxs[j]]["sent"])
    return " ".join(out)

rows = []
for pid, idxs in papers.items():
    sents = [js[i]["sent"] for i in idxs]
    gold = sum(js[i]["label"] for i in idxs)
    # full-paper baseline: whole paper to the LLM
    full_t = tok(" ".join(sents)) + FULL_PROMPT
    # funnel: Stage A (section gate) then Stage B (PubMedBERT det positives) -> Stage C
    kept = [k for k, i in enumerate(idxs) if js[i]["sec"] in KEEP]          # Stage A
    detpos = [k for k in kept if js[idxs[k]]["preds"]["PubMedBERT-FT"] == 1]  # Stage B
    item_t = sum(tok(js[idxs[k]]["sent"]) + tok(ctx(idxs, k, None)) + ITEM_WRAP for k in detpos)
    n_batches = max(1, math.ceil(len(detpos)/BATCH)) if detpos else 0
    stagec_t = n_batches*(SYS_T + SHOTS_T + BATCH_OVERHEAD) + item_t
    rows.append(dict(paper=pid, sents=len(sents), gold=gold,
                     kept_A=len(kept), detpos_B=len(detpos),
                     full_tokens=full_t, funnel_tokens=stagec_t))

df = pd.DataFrame(rows)
tot_full = df.full_tokens.sum(); tot_fun = df.funnel_tokens.sum()
print(f"SAL cost accounting — {len(df)} papers, {df.sents.sum()} sentences, {df.gold.sum()} gold limitations\n")
print(f"{'':28}{'per paper (mean)':>18}{'total (40 papers)':>20}")
print(f"{'full-paper LLM  tokens':28}{df.full_tokens.mean():>18,.0f}{tot_full:>20,.0f}")
print(f"{'funnel Stage-C  tokens':28}{df.funnel_tokens.mean():>18,.0f}{tot_fun:>20,.0f}")
print(f"{'reduction (x cheaper)':28}{df.full_tokens.mean()/df.funnel_tokens.mean():>17.1f}x{tot_full/tot_fun:>19.1f}x")
print(f"\nStage A keeps {df.kept_A.sum()}/{df.sents.sum()} sents ({df.kept_A.sum()/df.sents.sum():.1%}); "
      f"Stage B -> {df.detpos_B.sum()} to the LLM ({df.detpos_B.sum()/df.sents.sum():.1%} of all sentences)")
print(f"funnel sends ~{df.detpos_B.mean():.0f} sentences/paper to the LLM vs the whole paper "
      f"(~{df.sents.mean():.0f} sentences)")

# $ at a representative cheap-LLM input price (illustrative; vendor-neutral tokens above are primary)
for label, price in [("$0.15 / 1M in (GPT-4o-mini-like)", 0.15), ("$0.20 / 1M in", 0.20)]:
    print(f"\n@ {label}:  full-paper ${tot_full/1e6*price:.4f} total / ${df.full_tokens.mean()/1e6*price*1000:.3f} per 1k papers"
          f"   |  funnel ${tot_fun/1e6*price:.4f} total / ${df.funnel_tokens.mean()/1e6*price*1000:.3f} per 1k papers")
# ---- variants that change the picture ----
FIX = SYS_T + SHOTS_T + BATCH_OVERHEAD
item_total = int((df.funnel_tokens - [max(1,math.ceil(d/BATCH))*FIX for d in df.detpos_B]).sum())
# per-paper batching (base) already in tot_fun. cross-paper batching: overhead paid per 20 items globally
xbatch_fun = math.ceil(df.detpos_B.sum()/BATCH)*FIX + item_total
# prompt-caching: SYS+SHOTS cached (paid ~once), only item tokens + tiny per-call overhead billed
cached_fun = item_total + len(df)*BATCH_OVERHEAD
print("\n--- funnel Stage-C variants (total tokens over 40 papers) ---")
print(f"  base (1 batch/paper)        {tot_fun:>10,.0f}   {tot_full/tot_fun:>5.1f}x cheaper")
print(f"  cross-paper batching        {xbatch_fun:>10,.0f}   {tot_full/xbatch_fun:>5.1f}x")
print(f"  + prompt caching (SYS+shots){cached_fun:>10,.0f}   {tot_full/cached_fun:>5.1f}x")

print("\n--- scaling: funnel cost is ~flat per paper; full-paper grows with length ---")
import numpy as np
r = np.corrcoef(df.sents, df.funnel_tokens)[0,1]; rf = np.corrcoef(df.sents, df.full_tokens)[0,1]
print(f"  corr(paper length, tokens): funnel {r:.2f}  vs  full-paper {rf:.2f}")
for L in (7000, 10000, 15000, 25000):
    print(f"  at a {L//1000}k-token paper: ratio ~ {L/df.funnel_tokens.mean():.1f}x (funnel ~{df.funnel_tokens.mean():.0f} flat)")

print("\n*** Stage A+B use NO LLM (local bge-small encode, ~ms/paper, $0 API) and already reach "
      f"recall ~0.88 on SAL. Stage C (the {df.funnel_tokens.mean():.0f} tokens/paper above) is OPTIONAL precision polish. ***")
df.to_csv(Path(__file__).parent/"cost_tokens.csv", index=False)
print("\nsaved -> cost_tokens.csv")
