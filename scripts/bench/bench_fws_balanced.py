"""Unified FWS comparison on Zhang et al.'s protocol (balanced 1:1) so cheap lexical
models and neural encoders sit in ONE table.

Their reported recognition SOTA (BernoulliNB, 10-fold CV) = Macro-F1 0.9073. We put on
the same balanced data (9009 pos + 9009 neg, single random split with sentence-level
leakage, as in their CV):
  - tuned BernoulliNB (their model class, lemmatized BoW)     -- the cheap SOTA
  - TF-IDF word + logreg                                       -- cheap linear
  - fine-tuned SciBERT / PubMedBERT / bge-small (raw text)     -- the encoders
  - our SHIPPED zero-shot head (cue+gap_head, no training)     -- the deployed pipeline
All report positive-F1 and Macro-F1 on the same held-out fold.

    python scripts/bench/bench_fws_balanced.py
"""
from __future__ import annotations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import csv, io, pickle, random, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "scripts" / "training"))
from bench_limgen import prf  # noqa: E402
from bench_fws import macro_f1  # noqa: E402
from gap2idea.pipeline.gap_funnel import cue_label, EmbeddingGapHead  # noqa: E402

RAWCSV = ROOT / "data" / "fws_recognition.csv"
LEMM = ROOT / "data" / "fws_lemm.pkl"


def load_aligned():
    rows = list(csv.DictReader(io.StringIO(RAWCSV.read_text(encoding="utf-8", errors="replace"))))
    raw = [r["text"] for r in rows]; y = np.array([int(r["label"]) for r in rows])
    lem = pickle.load(open(LEMM, "rb"))[0] if LEMM.exists() else raw
    return raw, lem, y


def balanced_split(y, seed=0):
    rng = random.Random(seed)
    pos = [i for i in range(len(y)) if y[i] == 1]
    neg = [i for i in range(len(y)) if y[i] == 0]
    rng.shuffle(neg); idx = pos + neg[:len(pos)]; rng.shuffle(idx)
    n = len(idx); te = idx[:n // 10]; dv = idx[n // 10:n // 5]; tr = idx[n // 5:]
    return tr, dv, te


def tune(y, p, metric):
    bt, bm = 0.5, -1
    for t in np.linspace(0.1, 0.9, 81):
        m = metric(y, (p >= t).astype(int))
        if m > bm: bm, bt = m, t
    return bt


def row(name, y, pred):
    P, R, F = prf(y, pred); M = macro_f1(y, pred)
    print(f"{name:<30} P={P:.3f} R={R:.3f} posF1={F:.3f} macroF1={M:.3f}"
          f"{'  >=0.907' if M >= 0.9073 else ''}", flush=True)


def main():
    raw, lem, y = load_aligned()
    tr, dv, te = balanced_split(y)
    yte = y[te]; ydv = y[dv]
    print(f"balanced split: train {len(tr)} / dev {len(dv)} / test {len(te)} (test pos {yte.mean():.0%})")
    print("reference: Zhang 2022 BernoulliNB 10-fold CV Macro-F1 0.9073\n")

    # --- cheap lexical (lemmatized BoW), tuned NB + logreg ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 4), min_df=3, sublinear_tf=True).fit([lem[i] for i in tr])
    Xtr = vec.transform([lem[i] for i in tr]); sel = SelectKBest(chi2, k=min(14000, Xtr.shape[1])).fit(Xtr, y[tr])
    nb = BernoulliNB(alpha=1e-5).fit(sel.transform(Xtr), y[tr])
    row("BernoulliNB(1-4,chi2) [cheap]", yte, nb.predict(sel.transform(vec.transform([lem[i] for i in te]))))
    lr = LogisticRegression(C=8, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr])
    pdv = lr.predict_proba(vec.transform([lem[i] for i in dv]))[:, 1]
    t = tune(ydv, pdv, macro_f1)
    row("TF-IDF word+logreg [cheap]", yte, (lr.predict_proba(vec.transform([lem[i] for i in te]))[:, 1] >= t).astype(int))

    # --- shipped zero-shot head (raw text, no training) ---
    head = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")
    pr = head.predict([raw[i] for i in te])
    zs = np.array([1 if (cue_label(raw[te[j]]) == "future_work" or (l == "future_work" and p >= 0.5)) else 0
                   for j, (l, p) in enumerate(pr)])
    row("shipped cue+head [zero-shot]", yte, zs)

    # --- fine-tuned encoders (raw text) ---
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from finetune_rct import train, proba
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trs = [raw[i] for i in tr]; trly = y[tr]
    for m in ["allenai/scibert_scivocab_uncased",
              "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
              "BAAI/bge-small-en-v1.5"]:
        tok = AutoTokenizer.from_pretrained(m)
        model = AutoModelForSequenceClassification.from_pretrained(m, num_labels=2).to(device)
        train(model, tok, trs, trly, 3, 0, device)
        pdv = proba(model, tok, [raw[i] for i in dv], device); t = tune(ydv, pdv, macro_f1)
        pte = proba(model, tok, [raw[i] for i in te], device)
        row(f"{m.split('/')[-1][:24]} [ft]", yte, (pte >= t).astype(int))


if __name__ == "__main__":
    main()
