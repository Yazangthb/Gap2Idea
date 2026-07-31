"""Airtight single-split comparison vs prior-art methods + push metrics higher.

ONE LimGen train/val/test split for EVERY method (ours + reproduced prior art),
so the table is fully apples-to-apples. Then a new enhancement: stack the
fine-tuned BERT together with the frozen models.

Methods (all on the identical split, leakage-clean):
  cue rules (lexical)                 our Stage-B limitation cues
  BernoulliNB (Zhang's best recog.)   reproduced classical baseline
  tfidf + logreg                      classical
  bge + logreg                        OURS (current Stage B)
  stacking[bge+tfidf+cue]             OURS (frozen ensemble)
  DistilBERT fine-tuned               their fine-tuned-BERT approach
  stacking[+DistilBERT]               OURS, NEW (fine-tuned + frozen fused)

Published numbers (RCT/PubMedBERT 0.82 biomed, Zhang 0.91 ACL future-work) are
on DIFFERENT data we can't reproduce — kept as context only.

    python -u scripts/bench/bench_research_single.py --cap-pos 2500 --test-papers 180 --ft-epochs 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import BernoulliNB  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer  # noqa: E402

from gap2idea.pipeline.gap_funnel import cue_label  # noqa: E402
from bench_limgen import build_xy, fetch, prf  # noqa: E402

REPORT = ROOT / "docs/experiments/research_comparison_singlesplit.md"
rows = []


def best_thr(proba, y):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        _, _, f = prf(y, (proba >= t).astype(int))
        if f > bf:
            bf, bt = f, float(t)
    return bt


def emit(name, kind, P, R, F, note=""):
    rows.append({"method": name, "kind": kind, "P": P, "R": R, "F1": F, "note": note})
    print(f">>> {name:38} [{kind:6}] P={P} R={R} F1={F}  {note}", flush=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write(f"| {name} | {kind} | {P} | {R} | {F} | {note} |\n")


def train_distilbert(tr_s, tr_y, epochs, seed):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    counts = np.bincount(tr_y, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5); bs = 16
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(tr_s)); tot = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([tr_s[k] for k in idx], return_tensors="pt", padding=True, truncation=True, max_length=96)
            loss = lossf(model(**enc).logits, torch.tensor(tr_y[idx], dtype=torch.long))
            loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss.detach())
        print(f"    distilbert epoch {ep+1}/{epochs} loss {tot/(len(order)//bs+1):.3f}", flush=True)

    def proba(sents):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(sents), 32):
                enc = tok(sents[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=96)
                out.extend(torch.softmax(model(**enc).logits, -1)[:, 1].tolist())
        return np.array(out)
    return proba


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-pos", type=int, default=2500)
    ap.add_argument("--train-papers", type=int, default=4000)
    ap.add_argument("--test-papers", type=int, default=180)
    ap.add_argument("--ft-epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr_s, tr_y = build_xy(fetch("train.jsonl"), args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(fetch("test.jsonl"), args.test_papers, None, args.seed + 1)
    rng = np.random.default_rng(args.seed); perm = rng.permutation(len(tr_s))
    nv = len(tr_s) // 5; vi, ti = perm[:nv], perm[nv:]
    sub_s = [tr_s[k] for k in ti]; ytr = tr_y[ti]
    val_s = [tr_s[k] for k in vi]; yval = tr_y[vi]

    REPORT.write_text(
        "# Airtight single-split comparison — LimGen limitation detection\n\n"
        f"One split for ALL methods (leakage-clean). train={len(sub_s)} val={len(val_s)} "
        f"test={len(te_s)} ({int(te_y.sum())} limitation / {int((1-te_y).sum())} other, 1:3).\n\n"
        "| method | kind | precision | recall | F1 | note |\n|---|---|---|---|---|---|\n",
        encoding="utf-8")
    print(f"train={len(sub_s)} val={len(val_s)} test={len(te_s)}", flush=True)

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("encoding bge ...", flush=True)
    Xtr = enc.encode(sub_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xval = enc.encode(val_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xte = enc.encode(te_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    # --- cue rules ---
    cue_te = np.array([1 if cue_label(s) == "limitation" else 0 for s in te_s])
    emit("cue rules (lexical)", "ours", *prf(te_y, cue_te))
    cue_val = np.array([1.0 if cue_label(s) == "limitation" else 0.0 for s in val_s])

    # --- BernoulliNB (Zhang's best recognition model) ---
    cv = CountVectorizer(ngram_range=(1, 2), binary=True, min_df=2, max_features=50000)
    Xtr_c = cv.fit_transform(sub_s)
    nb = BernoulliNB().fit(Xtr_c, ytr)
    emit("BernoulliNB (Zhang repro)", "prior", *prf(te_y, nb.predict(cv.transform(te_s))))

    # --- tfidf + logreg ---
    tfv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000)
    Xtr_t = tfv.fit_transform(sub_s)
    tf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr_t, ytr)
    pt_te = tf.predict_proba(tfv.transform(te_s))[:, 1]; pt_val = tf.predict_proba(tfv.transform(val_s))[:, 1]
    emit("tfidf + logreg", "classical", *prf(te_y, (pt_te >= 0.5).astype(int)))

    # --- bge + logreg (OURS, current) ---
    bge = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    pb_te = bge.predict_proba(Xte)[:, 1]; pb_val = bge.predict_proba(Xval)[:, 1]
    emit("bge + logreg (OURS current)", "ours", *prf(te_y, (pb_te >= 0.5).astype(int)))

    # --- stacking[bge+tfidf+cue] (OURS frozen ensemble) ---
    Fval = np.column_stack([pb_val, pt_val, cue_val]); Fte = np.column_stack([pb_te, pt_te, cue_te])
    meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Fval, yval)
    ps_te = meta.predict_proba(Fte)[:, 1]
    t = best_thr(meta.predict_proba(Fval)[:, 1], yval)
    emit("stacking[bge+tfidf+cue] (OURS)", "ours", *prf(te_y, (ps_te >= t).astype(int)), note=f"thr={t:.2f}")

    # --- DistilBERT fine-tuned (their approach) ---
    print("fine-tuning DistilBERT ...", flush=True)
    proba = train_distilbert(tr_s, tr_y, args.ft_epochs, args.seed)
    pd_te = proba(te_s); pd_val = proba(val_s)
    emit("DistilBERT fine-tuned (their method)", "prior", *prf(te_y, (pd_te >= 0.5).astype(int)),
         note=f"{args.ft_epochs}ep")

    # --- stacking[+DistilBERT] (OURS, NEW — fused) ---
    Fval2 = np.column_stack([pb_val, pt_val, cue_val, pd_val]); Fte2 = np.column_stack([pb_te, pt_te, cue_te, pd_te])
    meta2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Fval2, yval)
    pf_te = meta2.predict_proba(Fte2)[:, 1]
    t2 = best_thr(meta2.predict_proba(Fval2)[:, 1], yval)
    emit("stacking[+DistilBERT] (OURS NEW)", "ours", *prf(te_y, (pf_te >= t2).astype(int)), note=f"thr={t2:.2f}")

    import pandas as pd
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    print("\n=== SORTED ===\n" + df.to_string(index=False), flush=True)
    with REPORT.open("a", encoding="utf-8") as f:
        f.write("\n_Published (different data/domain, NOT reproducible here): RCT/PubMedBERT 0.82 "
                "(biomed limitations); Zhang et al. 0.91 (ACL future-work recognition)._\n")
    print(f"saved -> {REPORT}", flush=True)


if __name__ == "__main__":
    main()
