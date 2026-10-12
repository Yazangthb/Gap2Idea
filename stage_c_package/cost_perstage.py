# -*- coding: utf-8 -*-
"""Per-stage metrics on the SAME 10 papers used in the cost experiment:
Stage A (section gate) -> A+B (detector) -> A+B+C (LLM filter), recall/precision each,
vs the full-paper LLM. Makes the cost experiment's funnel metrics explicit."""
import os, sys, json, random
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
from gap2idea.pipeline.llm import get_llm_client

js = json.load(open(ROOT/"stage_c_package"/"sal_criteria"/"sal_test_allmodels.json", encoding="utf-8"))
papers = {}
for i, r in enumerate(js): papers.setdefault(r["pmcid"], []).append(i)
cand = [(p, idx) for p, idx in papers.items() if sum(js[i]["label"] for i in idx) >= 2]
random.seed(1); random.shuffle(cand); sample = cand[:10]           # SAME sample as cost_fullpaper.py
KEEP = {"DISCUSS", "CONCL"}

# flatten the sample in reading order
flat = [i for _, idx in sample for i in idx]
gold = [i for i in flat if js[i]["label"] == 1]
def window(i):
    pm = js[i]["pmcid"]; pos = flat.index(i) if i in flat else None
    j = flat.index(i)
    return " ".join(js[flat[k]]["sent"] for k in (j-1, j, j+1)
                    if 0 <= k < len(flat) and js[flat[k]]["pmcid"] == pm)

def PR(keep_idx):
    keep = set(keep_idx); tp = len(keep & set(gold))
    P = tp/len(keep) if keep else 0; R = tp/len(gold)
    return round(P,3), round(R,3), round(2*P*R/(P+R) if P+R else 0,3), len(keep), tp

# Stage A: section gate
A = [i for i in flat if js[i]["sec"] in KEEP]
# Stage A+B: + PubMedBERT detector
AB = [i for i in A if js[i]["preds"]["PubMedBERT-FT"] == 1]
# Stage A+B+C: run Stage C (lite, validate_rct, +/-1) on the AB positives
CLIENT = get_llm_client("yandex"); F = os.environ["YANDEX_FOLDER_ID"].strip()
f = LLMGapFilter(backend="api", model=f"gpt://{F}/yandexgpt-5-lite/latest",
                 mode="validate_rct", client=CLIENT, context_style="inline")
items = [{"sentence": js[i]["sent"], "context": window(i)} for i in AB]
keepC = []
for k in range(0, len(items), 20): keepC.extend(f.judge_batch(items[k:k+20]))
ABC = [i for i, kp in zip(AB, keepC) if kp]

print(f"cost-experiment sample: {len(sample)} papers, {len(flat)} sentences, {len(gold)} gold limitations\n")
print(f"{'stage':34}{'P':>7}{'R':>7}{'F1':>7}{'kept':>7}{'TP':>5}")
for name, idxs in [("Stage A (section gate)", A), ("A+B (+ PubMedBERT detector)", AB),
                   ("A+B+C (+ Stage C LLM filter)", ABC)]:
    P,R,Fv,k,tp = PR(idxs); print(f"{name:34}{P:>7.3f}{R:>7.3f}{Fv:>7.3f}{k:>7}{tp:>5}")
print(f"{'full-paper LLM (same model)':34}{0.559:>7.3f}{0.345:>7.3f}{0.429:>7.3f}{'-':>7}{'-':>5}")
print(f"\nStage A keeps {len(A)}/{len(flat)} sentences; Stage C made {f.n_calls} calls on {len(AB)} candidates.")
json.dump(dict(papers=len(sample), sents=len(flat), gold=len(gold),
               A=PR(A), AB=PR(AB), ABC=PR(ABC)), open(Path(__file__).parent/"cost_perstage_results.json","w"), indent=1)
