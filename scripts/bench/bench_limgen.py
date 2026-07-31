"""Head-to-head vs prior art on LimGen (same data, same split, leakage-clean).

LimGen (arbmf/LimGen, CC-BY-4.0) ships its own train/test split of ACL papers
with the mandated `limitations` section. We frame LIMITATION DETECTION as binary
sentence classification (positive = a sentence from the Limitations section;
negative = a sentence from elsewhere in the paper), train every method on
LimGen-TRAIN, and evaluate on held-out LimGen-TEST.

Leakage note: the shipped head was trained on LimGen test+valid, so it is NOT
used here — all classifiers are trained fresh on the train split only.

Methods compared (all on identical data):
  - cue rules            our Stage-B limitation cue regex (no training; lexical baseline ~ Hu&Wan/Zhang)
  - tfidf + logreg       classical bag-of-words (a Zhang-style cheap baseline)
  - bge-small + logreg   OURS (frozen embedding + logreg)
  - SciBERT (fine-tuned) THEIR method (domain BERT, fine-tuned end-to-end; ~Zhang/RCT-PubMedBERT)

Metric: binary precision / recall / F1 for the positive (limitation) class, at a
fixed pos:neg test ratio (documented).

    python scripts/bench/bench_limgen.py --train-papers 1500 --epochs 2
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import cue_label, _looks_like_sentence  # noqa: E402
from gap2idea.pipeline.gap_prefilter import normalize_text, split_sentences  # noqa: E402

BASE = "https://raw.githubusercontent.com/arbmf/LimGen/main/Datasets/base/"
CACHE = ROOT / "data" / "limgen"
NEG_RATIO = 3


def fetch(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if not dest.exists() or dest.stat().st_size < 1000:
        print(f"downloading {name} ...")
        req = urllib.request.Request(BASE + name, headers={"User-Agent": "curl"})
        with urllib.request.urlopen(req, timeout=600) as r, dest.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    return dest


def build_xy(path: Path, n_papers: int | None, cap_pos: int | None, seed: int):
    rng = np.random.default_rng(seed)
    pos, neg, seen = [], [], set()
    np_seen = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if n_papers and np_seen >= n_papers:
            break
        np_seen += 1
        lim = str(rec.get("limitations", "") or "")
        content = str(rec.get("content", "") or "")
        lim_sents, lim_keys = [], set()
        for s in split_sentences(lim):
            if _looks_like_sentence(s):
                k = normalize_text(s)
                if k and k not in seen:
                    seen.add(k); lim_keys.add(k); lim_sents.append(s)
        for s in lim_sents:
            pos.append(s)
        cand = []
        for s in split_sentences(content):
            if _looks_like_sentence(s):
                k = normalize_text(s)
                if k and k not in seen and k not in lim_keys:
                    seen.add(k); cand.append(s)
        rng.shuffle(cand)
        for s in cand[:max(1, len(lim_sents) * NEG_RATIO)]:
            neg.append(s)
    if cap_pos and len(pos) > cap_pos:
        pos = list(np.array(pos)[rng.permutation(len(pos))[:cap_pos]])
        neg = list(np.array(neg)[rng.permutation(len(neg))[:cap_pos * NEG_RATIO]])
    sents = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    return sents, y


def prf(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return round(P, 3), round(R, 3), round(2 * P * R / (P + R), 3) if P + R else 0.0


def m_rules(tr_s, tr_y, te_s):
    return np.array([1 if cue_label(s) == "limitation" else 0 for s in te_s])


def m_tfidf(tr_s, tr_y, te_s):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000),
                         LogisticRegression(max_iter=2000, class_weight="balanced"))
    pipe.fit(tr_s, tr_y)
    return pipe.predict(te_s)


def m_bge(tr_s, tr_y, te_s):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5")
    Xtr = enc.encode(tr_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xte = enc.encode(te_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, tr_y)
    return clf.predict(Xte)


def m_specter(tr_s, tr_y, te_s):
    """Frozen SCIENTIFIC-domain encoder (SPECTER) + logreg — domain-encoder baseline."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    enc = SentenceTransformer("sentence-transformers/allenai-specter")
    Xtr = enc.encode(tr_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xte = enc.encode(te_s, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(Xtr, tr_y)
    return clf.predict(Xte)


def m_bert_ft(tr_s, tr_y, te_s, epochs, seed, model_name="distilbert-base-uncased"):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    counts = np.bincount(tr_y, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    bs = 16
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(tr_s)); tot = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([tr_s[k] for k in idx], return_tensors="pt", padding=True, truncation=True, max_length=96)
            yb = torch.tensor(tr_y[idx], dtype=torch.long)
            loss = lossf(model(**enc).logits, yb)
            loss.backward(); opt.step(); opt.zero_grad(); tot += float(loss.detach())
        print(f"    {model_name.split('/')[-1]} epoch {ep+1}/{epochs} loss {tot/(len(order)//bs+1):.3f}", flush=True)
    model.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(te_s), 32):
            enc = tok(te_s[i:i + 32], return_tensors="pt", padding=True, truncation=True, max_length=96)
            preds.extend(model(**enc).logits.argmax(-1).tolist())
    return np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-papers", type=int, default=1500)
    ap.add_argument("--cap-pos", type=int, default=1800, help="cap positives (fine-tune CPU budget)")
    ap.add_argument("--test-papers", type=int, default=None, help="cap test papers (for fast consistent eval)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr_path, te_path = fetch("train.jsonl"), fetch("test.jsonl")
    tr_s, tr_y = build_xy(tr_path, args.train_papers, args.cap_pos, args.seed)
    te_s, te_y = build_xy(te_path, args.test_papers, None, args.seed + 1)
    print(f"TRAIN: {len(tr_s)} sents ({int(tr_y.sum())} limitation / {int((1-tr_y).sum())} other)")
    print(f"TEST : {len(te_s)} sents ({int(te_y.sum())} limitation / {int((1-te_y).sum())} other)  "
          f"pos:neg = 1:{NEG_RATIO}")

    methods = [("cue rules (ours, lexical)", m_rules),
               ("tfidf + logreg (classical)", m_tfidf),
               ("bge-small + logreg (OURS)", m_bge),
               ("SPECTER frozen + logreg (domain encoder)", m_specter),
               ("DistilBERT fine-tuned (their approach)",
                lambda a, b, c: m_bert_ft(a, b, c, args.epochs, args.seed))]
    rows = []
    for name, fn in methods:
        print(f"\n>>> {name}")
        try:
            P, R, F = prf(te_y, fn(tr_s, tr_y, te_s))
            rows.append({"method": name, "precision": P, "recall": R, "f1": F})
            print(f"    P={P} R={R} F1={F}")
        except Exception as e:
            print(f"    FAILED: {e}")
            rows.append({"method": name, "precision": None, "recall": None, "f1": None})

    df = pd.DataFrame(rows).sort_values("f1", ascending=False, na_position="last")
    print("\n=== LimGen limitation-detection (held-out test, leakage-clean) ===")
    print(df.to_string(index=False))

    out = ROOT / "docs/experiments/limgen_comparison.md"
    lines = ["# LimGen head-to-head — limitation sentence detection", "",
             f"Same data & split as LimGen (ACL papers). Binary: limitation-section sentence vs other. "
             f"Trained on LimGen-TRAIN ({len(tr_s)} sents), evaluated on held-out LimGen-TEST "
             f"({len(te_s)} sents, pos:neg 1:{NEG_RATIO}). All methods trained fresh on the same split "
             "(the shipped head — trained on test+valid — is deliberately excluded to avoid leakage).", "",
             "| method | precision | recall | F1 |", "|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r['method']} | {r['precision']} | {r['recall']} | {r['f1']} |")
    lines += ["", "_Published references (different splits/domains, NOT directly comparable):_ "
              "RCT/PubMedBERT limitation detection F1 0.82 (biomed); Zhang et al. future-work "
              "recognition Macro-F1 0.91 (ACL).", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
