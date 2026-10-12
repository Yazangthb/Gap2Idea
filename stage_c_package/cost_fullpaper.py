# -*- coding: utf-8 -*-
"""Cost analysis Part 2 (small API run): the full-paper-LLM baseline.
Send each paper's numbered sentences to yandexgpt-5-lite, ask for the indices of
self-acknowledged limitation sentences. Measure recall/precision + ACTUAL input tokens,
and compare to the funnel (Stage A+B detector recall, free) on the same papers."""
import os, sys, json, re, random
from pathlib import Path
ROOT = Path(r"C:/Users/yazan/OneDrive/Desktop/thesis/Gap2Idea")
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
import numpy as np
from gap2idea.pipeline.llm import get_llm_client

js = json.load(open(ROOT/"stage_c_package"/"sal_criteria"/"sal_test_allmodels.json", encoding="utf-8"))
papers = {}
for i, r in enumerate(js): papers.setdefault(r["pmcid"], []).append(i)
# sample 10 papers with >=2 gold limitations
cand = [(p, idx) for p, idx in papers.items() if sum(js[i]["label"] for i in idx) >= 2]
random.seed(1); random.shuffle(cand); sample = cand[:10]

CLIENT = get_llm_client("yandex"); F = os.environ["YANDEX_FOLDER_ID"].strip()
MODEL = f"gpt://{F}/yandexgpt-5-lite/latest"
SYS = ("You extract SELF-ACKNOWLEDGED LIMITATIONS from a research paper. You are given the paper's "
       "sentences, each prefixed by its index [n]. Return ONLY a JSON array of the integer indices of "
       "sentences in which the AUTHORS acknowledge a limitation/weakness/caveat of THEIR OWN study "
       "(e.g. small sample, no blinding, single-center, short follow-up, confounding, selection bias). "
       "No prose, just the JSON array.")

def extract(idx):
    sents = [js[i]["sent"] for i in idx]
    body = "\n".join(f"[{k}] {s}" for k, s in enumerate(sents))
    resp = CLIENT.chat.completions.create(model=MODEL, temperature=0,
        messages=[{"role":"system","content":SYS},{"role":"user","content":body}])
    txt = resp.choices[0].message.content
    m = re.search(r"\[[\d,\s]*\]", txt)
    pred = set(int(x) for x in re.findall(r"\d+", m.group(0))) if m else set()
    usage = getattr(resp, "usage", None)
    ptok = getattr(usage, "prompt_tokens", None) if usage else None
    return pred, ptok

print(f"full-paper LLM (yandexgpt-5-lite) on {len(sample)} SAL papers\n")
print(f"{'paper':13}{'sents':>6}{'gold':>5}{'LLM recall':>11}{'LLM prec':>9}{'in-tok':>8}{'funnel A+B R':>13}")
tot=dict(gr=0,gp=0,gold=0,pred=0,tok=0, abr=0)
for p, idx in sample:
    gold = {k for k,i in enumerate(idx) if js[i]["label"]==1}
    pred, ptok = extract(idx)
    pred = {x for x in pred if 0 <= x < len(idx)}
    tp = len(gold & pred)
    R = tp/len(gold) if gold else 0; P = tp/len(pred) if pred else 0
    # funnel Stage A+B (PubMedBERT det) recall on this paper, free
    detpos = {k for k,i in enumerate(idx) if js[i]["sec"] in {"DISCUSS","CONCL"} and js[i]["preds"]["PubMedBERT-FT"]==1}
    abr = len(gold & detpos)/len(gold) if gold else 0
    print(f"{p:13}{len(idx):>6}{len(gold):>5}{R:>11.2f}{P:>9.2f}{(ptok or 0):>8}{abr:>13.2f}")
    tot["gr"]+=tp; tot["gold"]+=len(gold); tot["pred"]+=len(pred); tot["tok"]+=(ptok or 0)
    tot["abr"]+=len(gold & detpos)
R=tot["gr"]/tot["gold"]; P=tot["gr"]/tot["pred"] if tot["pred"] else 0; ABR=tot["abr"]/tot["gold"]
print(f"\nFULL-PAPER LLM   : recall {R:.3f}  precision {P:.3f}  | mean input {tot['tok']/len(sample):,.0f} tokens/paper (ACTUAL)")
print(f"FUNNEL Stage A+B : recall {ABR:.3f}  | 0 LLM tokens (local encode)")
print(f"FUNNEL Stage C   : ~1,891 tokens/paper (only ~10 sentences; see cost_tokens.py)")
json.dump(dict(n=len(sample), fullpaper_recall=round(R,3), fullpaper_prec=round(P,3),
               fullpaper_tok=round(tot['tok']/len(sample)), funnel_AB_recall=round(ABR,3)),
          open(Path(__file__).parent/"cost_fullpaper_results.json","w"), indent=1)
