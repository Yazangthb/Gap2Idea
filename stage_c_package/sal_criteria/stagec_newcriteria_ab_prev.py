# -*- coding: utf-8 -*-
"""Run the NEW keep-biased SAL-criteria Stage C through:
  (a) the full end-to-end A+B+C funnel (40 full papers, 205 gold, all 8 Stage B models)
  (b) the prevalence-20 section pool (their SOTA condition), head-to-head vs their 0.821

Both derive from ONE Stage C pass over the union of all models' A+B positives, judged with
TWO prompts on identical data: keep-biased SAL-criteria (new) and rct_loose (the shipped/old).
Model = yandexgpt-5-lite, +/-1 inline context, batch 20 — same config as the broad funnel (0.701)
so the only variable is the prompt.
"""
import os, sys, json, time
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np, pandas as pd
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
from gap2idea.pipeline.llm import get_llm_client
from bench_stagec_opt import SYSTEM_RCT_LOOSE, SHOTS_RCT   # the shipped KEEP-biased RCT prompt

HERE = Path(__file__).parent
rows = json.load(open(HERE / "sal_test_allmodels.json", encoding="utf-8"))
def window(i, k=1):
    r = rows[i]
    return " ".join(rows[j]["sent"] for j in range(i-k, i+k+1)
                    if 0 <= j < len(rows) and rows[j]["pmcid"] == r["pmcid"])
KEEP = {"DISCUSS", "CONCL"}
for r in rows:
    r["inA"] = 1 if r["sec"] in KEEP else 0
N = len(rows); GOLD = sum(r["label"] for r in rows)
MODELS = list(rows[0]["preds"].keys())
AB = {m: [r["inA"] & r["preds"][m] for r in rows] for m in MODELS}

# --- new keep-biased SAL-criteria prompt (validated in the 3-model sweep) ---
SYS_SAL = ("You judge whether the «marked» TARGET sentence is a SELF-ACKNOWLEDGED LIMITATION of the "
 "AUTHORS' OWN study, using the SAL annotation criteria. A limitation is a weakness/caveat/bias/"
 "constraint the authors admit about THIS trial, fitting a known category: study design; loose or "
 "narrow population / convenience sampling; single-center or narrow setting; intervention "
 "composite/non-standardized; control (no placebo / active placebo / care-as-usual); outcome measure "
 "excluded/imprecise/unvalidated; missing data or drop-out; small sample / underpowered; randomization "
 "unbalanced/poor/open; blinding (patient or team not blinded); short duration or follow-up; "
 "multiple/post-hoc testing or unadjusted confounding; funding; limited generalizability.\n"
 "DECISION RULE (keep-biased):\n"
 "- keep=true if the TARGET states such a limitation, OR states a consequence ('could introduce bias', "
 "'cannot be assessed', 'results may be confounded', 'risk of type 1 error') whose underlying CAUSE "
 "(no blinding, small sample, single center, multiple testing, short follow-up...) is evident in the "
 "TARGET or its SURROUNDING context. When in doubt and a real weakness of this study is present, keep=true.\n"
 "- keep=false ONLY for CLEAR non-limitations: restated positive results; clinical recommendations; "
 "critique of prior/other work; plain methods description; background; or a bare header like "
 "'This study has several limitations.' with no specific weakness.")
SHOTS_SAL = [("the lack of a no treatment arm in this study", "YES"),
 ("our study included only a relatively small sample of participants", "YES"),
 ("The absence of masking is most relevant for the assessment of events.", "YES"),
 ("the crossover design and the relatively short treatment periods are limitations", "YES"),
 ("the trial was conducted at a single center", "YES"),
 ("Because outcomes were exploratory, the risk of type 1 error was considerable.", "YES"),
 ("This could have introduced a bias in the results.", "YES"),
 ("This study has several limitations.", "NO"),
 ("The intervention significantly reduced symptoms at 12 weeks.", "NO"),
 ("Clinicians should consider these findings when counseling patients.", "NO"),
 ("Existing methods cannot capture long-range dependencies.", "NO"),
 ("Participants were randomized 1:1 using a computer-generated sequence.", "NO")]

CLIENT = get_llm_client("yandex"); F = os.environ["YANDEX_FOLDER_ID"].strip()
MODEL = f"gpt://{F}/yandexgpt-5-lite/latest"

def judge_union(union, sys_p, shots):
    f = LLMGapFilter(backend="api", model=MODEL, mode="validate_rct",
                     client=CLIENT, context_style="inline")
    f._sys, f._shots = sys_p, shots
    verdict = {}
    for k in range(0, len(union), 20):
        ch = union[k:k+20]
        items = [{"sentence": rows[i]["sent"], "context": window(i, 1)} for i in ch]
        for i, keep in zip(ch, f.judge_batch(items)):
            verdict[i] = bool(keep)
    return verdict, f.n_calls

union = sorted({i for m in MODELS for i, k in enumerate(AB[m]) if k})
print(f"union of A+B positives to judge: {len(union)} sentences\n", flush=True)
t0 = time.time()
V_sal, c1 = judge_union(union, SYS_SAL, SHOTS_SAL)
V_rct, c2 = judge_union(union, SYSTEM_RCT_LOOSE, SHOTS_RCT)
print(f"Stage C judged union x2 prompts in {c1+c2} calls [{time.time()-t0:.0f}s]\n", flush=True)
json.dump({"sal": {str(k): v for k, v in V_sal.items()},
           "rct": {str(k): v for k, v in V_rct.items()}},
          open(HERE / "ab_prev_verdicts.json", "w"))

# ---------- (a) full end-to-end funnel: raw prevalence, 40 papers ----------
def prf_full(flags):
    kept = [r for r, k in zip(rows, flags) if k]
    tp = sum(r["label"] for r in kept)
    R = tp / GOLD; P = tp / len(kept) if kept else 0; Fv = 2*P*R/(P+R) if P+R else 0
    return round(P, 3), round(R, 3), round(Fv, 3)

print("=" * 78)
print("(a) FULL END-TO-END FUNNEL  | 40 papers | 205 gold | Stage A=DISCUSS+CONCL | lite, +/-1")
print("=" * 78)
rows_out = []
for m in MODELS:
    pab, rab, fab = prf_full(AB[m])
    abc_sal = [AB[m][i] and V_sal.get(i, False) for i in range(N)]
    abc_rct = [AB[m][i] and V_rct.get(i, False) for i in range(N)]
    ps, rs, fs = prf_full(abc_sal); pr, rr, fr = prf_full(abc_rct)
    rows_out.append(dict(model=m, AB_F1=fab,
                         ABC_sal_P=ps, ABC_sal_R=rs, ABC_sal_F1=fs,
                         ABC_rct_P=pr, ABC_rct_R=rr, ABC_rct_F1=fr))
tbl = pd.DataFrame(rows_out).sort_values("ABC_sal_F1", ascending=False).set_index("model")
print(tbl.to_string(), flush=True)
json.dump(rows_out, open(HERE / "ab_prev_funnel_results.json", "w"), indent=1)

# ---------- (b) prevalence-20 section pool (their SOTA condition) ----------
def prf(gold, pred):
    tp = int(((pred == 1) & (gold == 1)).sum()); fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    P = tp/(tp+fp) if tp+fp else 0; R = tp/(tp+fn) if tp+fn else 0
    return P, R, (2*P*R/(P+R) if P+R else 0)

def pm_f1(gold, pred, target=0.207, seeds=8):
    pos = np.where(gold == 1)[0]; neg = np.where(gold == 0)[0]
    n_neg = min(len(neg), int(round(len(pos) * (1-target) / target)))
    fs = []
    for sd in range(seeds):
        rng = np.random.default_rng(sd)
        idx = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
        fs.append(prf(gold[idx], pred[idx])[2])
    return float(np.mean(fs))

# pool = section-gated sentences; detector = PubMedBERT-FT; prevalence-match to 20.7%
pool = [i for i in range(N) if rows[i]["inA"] == 1]
gold = np.array([rows[i]["label"] for i in pool])
det = np.array([rows[i]["preds"]["PubMedBERT-FT"] for i in pool])
def apply_C(verdict):
    out = det.copy()
    for j, i in enumerate(pool):
        if det[j] == 1 and not verdict.get(i, True):
            out[j] = 0
    return out
fin_sal = apply_C(V_sal); fin_rct = apply_C(V_rct)
Pd, Rd, Fd = prf(gold, det)
print("\n" + "=" * 78)
print(f"(b) PREVALENCE-20 SECTION POOL | pool={len(pool)} ({int(gold.sum())} lim, {gold.mean():.1%} native)")
print(f"    detector=PubMedBERT-FT on DISCUSS+CONCL, prevalence-matched to 20.7%")
print("=" * 78)
print(f"{'config':<32}{'P':>7}{'R':>7}{'F1':>7}{'F1@20.7%':>11}")
print(f"{'their PubMedBERT (reported)':<32}{0.751:>7.3f}{0.907:>7.3f}{0.821:>7.3f}{0.821:>11.3f}")
print(f"{'detector only':<32}{Pd:>7.3f}{Rd:>7.3f}{Fd:>7.3f}{pm_f1(gold,det):>11.3f}")
for tag, fin in [("+ Stage C (SAL keep-biased)", fin_sal), ("+ Stage C (rct_loose, old)", fin_rct)]:
    P, R, Fv = prf(gold, fin)
    beat = "  >=0.821" if pm_f1(gold, fin) >= 0.821 else ""
    print(f"{tag:<32}{P:>7.3f}{R:>7.3f}{Fv:>7.3f}{pm_f1(gold,fin):>11.3f}{beat}")
