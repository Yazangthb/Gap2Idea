# -*- coding: utf-8 -*-
# Recovered 2026-11-05, code unchanged, from the Oct 8 2026 session that produced
# stage_c_package/sal_criteria/sal_test_allmodels.json (the 8 SAL Stage B detectors, incl. PubMedBERT-FT).
# Ran on an H100 with ~/sc_venv; input sal_test_sections.json comes from scripts/training/fetch_sal_sections.py.
# Not seeded for torch, so a rerun can differ slightly from the saved predictions.
"""On H100: all 8 Stage-B detectors on SAL -> per-test-sentence prediction, aligned
to sal_test_sections.json. 5 frozen (encoder+logreg) + 3 fine-tunes (dev-tuned)."""
import json, csv, io, urllib.request, time
import numpy as np, torch

RAW = "https://raw.githubusercontent.com/MengfeiLan/SAL_Type_Classification/main/data/%s.csv"
def load(split):
    raw = urllib.request.urlopen(urllib.request.Request(RAW % split, headers={"User-Agent":"x"}), timeout=120).read().decode("utf-8","replace")
    s, y = [], []
    for r in csv.DictReader(io.StringIO(raw)):
        t=(r.get("sentences") or "").strip()
        if len(t.split())<4: continue
        s.append(t); y.append(0 if (r.get("category_spans","[0]") or "[0]").strip()=="[0]" else 1)
    return s, np.array(y)

dev = "cuda" if torch.cuda.is_available() else "cpu"
tr_s,tr_y = load("train"); dv_s,dv_y = load("dev")
test = json.load(open("/home/yazan/sal_test_sections.json",encoding="utf-8")); te_s=[t["sent"] for t in test]
print(f"train {len(tr_s)} ({int(tr_y.sum())} lim) | dev {len(dv_s)} | test {len(te_s)} | {dev}", flush=True)
preds = {}

# ---- 5 frozen encoders + logreg ----
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
FROZEN = {"MiniLM-L6":"sentence-transformers/all-MiniLM-L6-v2",
          "mpnet":"sentence-transformers/all-mpnet-base-v2",
          "bge-small":"BAAI/bge-small-en-v1.5",
          "bge-base":"BAAI/bge-base-en-v1.5",
          "e5-base":"intfloat/e5-base-v2"}
for name, hf in FROZEN.items():
    t0=time.time(); enc=SentenceTransformer(hf, device=dev)
    em=lambda xs: enc.encode(xs, normalize_embeddings=True, batch_size=512, show_progress_bar=False)
    clf=LogisticRegression(C=2,max_iter=2000,class_weight="balanced").fit(em(tr_s),tr_y)
    preds[name+"(frozen)"]=clf.predict(em(te_s)).astype(int).tolist()
    del enc; torch.cuda.empty_cache()
    print(f"  frozen {name}: {sum(preds[name+'(frozen)'])} pos [{time.time()-t0:.0f}s]", flush=True)

# ---- 3 full fine-tunes (dev-threshold-tuned) ----
from transformers import AutoModelForSequenceClassification, AutoTokenizer
FT = {"DistilBERT-FT":"distilbert-base-uncased",
      "SciBERT-FT":"allenai/scibert_scivocab_uncased",
      "PubMedBERT-FT":"microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"}
def f1_at(p,y,t):
    pr=(p>=t).astype(int); tp=int(((pr==1)&(y==1)).sum()); fp=int(((pr==1)&(y==0)).sum()); fn=int(((pr==0)&(y==1)).sum())
    P=tp/(tp+fp) if tp+fp else 0; R=tp/(tp+fn) if tp+fn else 0; return (2*P*R/(P+R)) if P+R else 0
for name, hf in FT.items():
    t0=time.time()
    tok=AutoTokenizer.from_pretrained(hf); model=AutoModelForSequenceClassification.from_pretrained(hf,num_labels=2).to(dev)
    cnt=np.bincount(tr_y,minlength=2); w=torch.tensor(cnt.sum()/(2*np.maximum(cnt,1)),dtype=torch.float32,device=dev)
    lossf=torch.nn.CrossEntropyLoss(weight=w); opt=torch.optim.AdamW(model.parameters(),lr=3e-5)
    rng=np.random.default_rng(0)
    for ep in range(3):
        model.train(); order=rng.permutation(len(tr_s))
        for i in range(0,len(order),32):
            idx=order[i:i+32]
            enc=tok([tr_s[k] for k in idx],return_tensors="pt",padding=True,truncation=True,max_length=96).to(dev)
            loss=lossf(model(**enc).logits, torch.tensor(tr_y[idx],dtype=torch.long,device=dev))
            loss.backward(); opt.step(); opt.zero_grad()
    @torch.no_grad()
    def proba(sents,bs=128):
        model.eval(); out=[]
        for i in range(0,len(sents),bs):
            enc=tok(sents[i:i+bs],return_tensors="pt",padding=True,truncation=True,max_length=96).to(dev)
            out.extend(torch.softmax(model(**enc).logits,-1)[:,1].cpu().tolist())
        return np.array(out)
    dp=proba(dv_s); best=max(np.linspace(0.1,0.9,33), key=lambda t: f1_at(dp,dv_y,t))
    preds[name]=(proba(te_s)>=best).astype(int).tolist()
    del model; torch.cuda.empty_cache()
    print(f"  FT {name}: thr {best:.2f}, {sum(preds[name])} pos [{time.time()-t0:.0f}s]", flush=True)

for t in test: t["preds"]={}
for k,v in preds.items():
    for t,val in zip(test,v): t["preds"][k]=val
json.dump(test, open("/home/yazan/sal_test_allmodels.json","w"))
print("models:", list(preds.keys()), "\nwrote sal_test_allmodels.json", flush=True)
