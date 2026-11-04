# -*- coding: utf-8 -*-
"""Pick the Stage C model UNDER THE NEW (keep-biased SAL-criteria) PROMPT.

The earlier SAL-criteria run used only yandexgpt-5-pro. Here we sweep the 3 native
YandexGPT judges (lite / pro / 5.1) on the same hard 120-sentence sample (PubMedBERT
A+B positives, 60 gold-POS + 60 gold-NEG), +/-1 context, batch 20, vs SAL gold.

KEEP-BIASED SAL-criteria: align keep/drop to SAL's POS/NEG definition (cause of a
self-acknowledged limitation of the authors' OWN study, 16 categories) BUT, unlike the
strict cause-only prompt that collapsed recall (0.82->0.50), recover the cause from the
+/-1 CONTEXT and keep a consequence sentence when its cause is evident nearby. Drop only
CLEAR non-limitations.  Goal: hold precision ~0.66+ while pulling recall back to ~0.75-0.80.
"""
import os, sys, json, random
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
from gap2idea.pipeline.llm import get_llm_client

HERE = Path(__file__).parent
rows = json.load(open(HERE / "sal_test_allmodels.json", encoding="utf-8"))
def window(i, k=1):
    r = rows[i]
    return " ".join(rows[j]["sent"] for j in range(i-k, i+k+1)
                    if 0 <= j < len(rows) and rows[j]["pmcid"] == r["pmcid"])

KEEP = {"DISCUSS", "CONCL"}
cand = [i for i, r in enumerate(rows) if r["sec"] in KEEP and r["preds"]["PubMedBERT-FT"] == 1]
random.seed(0); random.shuffle(cand)
pos = [i for i in cand if rows[i]["label"] == 1]
neg = [i for i in cand if rows[i]["label"] == 0]
samp = pos[:60] + neg[:60]
random.seed(1); random.shuffle(samp)
G = np.array([rows[i]["label"] for i in samp])

CLIENT = get_llm_client("yandex"); F = os.environ["YANDEX_FOLDER_ID"].strip()
def uri(n): return f"gpt://{F}/{n}/latest"

# --- refined KEEP-BIASED SAL-criteria prompt ---
from gap2idea.pipeline.stage_c_prompts import SAL_SYS as SYS, SAL_SHOTS as SHOTS

def run(model):
    f = LLMGapFilter(backend="api", model=model, mode="validate_rct",
                     client=CLIENT, context_style="inline")
    f._sys, f._shots = SYS, SHOTS
    items = [{"sentence": rows[i]["sent"], "context": window(i, 1)} for i in samp]
    keep = []
    for b in range(0, len(items), 20):
        keep.extend(f.judge_batch(items[b:b+20]))
    kp = np.array([1 if x else 0 for x in keep])
    tp = int(((kp == 1) & (G == 1)).sum()); fp = int(((kp == 1) & (G == 0)).sum())
    fn = int(((kp == 0) & (G == 1)).sum())
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    F1 = 2*P*R/(P+R) if P+R else 0
    return dict(P=round(P, 3), R=round(R, 3), F1=round(F1, 3), calls=f.n_calls), kp

print(f"sample {len(samp)} (lim {int(G.sum())}/neg {len(G)-int(G.sum())}) "
      f"— keep-biased SAL-criteria, +/-1 ctx, batch 20, vs SAL gold\n", flush=True)
print(f"{'model':20}{'P':>7}{'R':>7}{'F1':>7}{'calls':>7}")
results = {}
for nm in ["yandexgpt-5-lite", "yandexgpt-5-pro", "yandexgpt-5.1"]:
    m, kp = run(uri(nm))
    results[nm] = (m, kp)
    print(f"{nm:20}{m['P']:>7.3f}{m['R']:>7.3f}{m['F1']:>7.3f}{m['calls']:>7}", flush=True)

best = max(results, key=lambda n: results[n][0]["F1"])
m, kp = results[best]
fp = [samp[i] for i in range(len(samp)) if kp[i] == 1 and G[i] == 0]
fn = [samp[i] for i in range(len(samp)) if kp[i] == 0 and G[i] == 1]
print(f"\nbest F1: {best}  (P={m['P']} R={m['R']} F1={m['F1']})")
print(f"remaining FP {len(fp)} (kept, gold=NEG — SAL under-label vs real LLM error?):")
for i in fp[:8]: print("  -", rows[i]['sent'][:130])
print(f"FN {len(fn)} (dropped, gold=POS — did we over-drop real limitations?):")
for i in fn[:8]: print("  -", rows[i]['sent'][:130])
json.dump({n: results[n][0] for n in results}, open(HERE/"newcriteria_sweep_results.json", "w"), indent=1)
