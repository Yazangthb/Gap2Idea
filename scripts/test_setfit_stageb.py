"""Test SetFit-style contrastive fine-tuning as the Stage-B classifier.

The `setfit` package is unusable here (it needs transformers<5 and its datasets/
dill path is broken on Python 3.14), so we implement SetFit's actual recipe
directly on sentence-transformers + torch:

  1. CONTRASTIVE body fine-tuning — sample same-class (target cos=1) and
     different-class (target cos=0) sentence pairs, train the bge encoder with a
     cosine-similarity (MSE) loss. This is exactly SetFit's Sentence-Transformer
     fine-tuning step.
  2. A logistic-regression head on the fine-tuned embeddings (SetFit's head).

Comparison (same eval + same training sentences as test_bert_stageb.py):
  BEFORE = bge+logreg (FROZEN encoder + logreg) and rules-only
  AFTER  = SetFit(bge) — the SAME encoder, CONTRASTIVELY FINE-TUNED, + logreg

SetFit targets the scarce-label regime that limits our limitation class, so this
is the fair test of "does few-shot contrastive fine-tuning beat frozen embeddings?".

Usage:
    python scripts/test_setfit_stageb.py --pairs 1500 --epochs 1
    python scripts/test_setfit_stageb.py --few-shot 24 --pairs 1200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train_gap_head import build_dataset, eval_paper_ids  # noqa: E402
from test_bert_stageb import (  # noqa: E402
    LABELS, TARGET, build_eval, bert_per_sentence_pred, per_sentence, end_to_end,
)
from bench_gap_recall import load as load_gold  # noqa: E402

BODY = "BAAI/bge-small-en-v1.5"


def make_train(sents, labels, max_neg_ratio, few_shot, seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"s": sents, "y": labels})
    if few_shot:
        parts = []
        for lab, g in df.groupby("y"):
            k = min(len(g), few_shot * (2 if lab == "none" else 1))
            parts.append(g.iloc[rng.permutation(len(g))[:k]])
        train = pd.concat(parts)
    else:
        pos, neg = df[df.y != "none"], df[df.y == "none"]
        cap = int(len(pos) * max_neg_ratio)
        if len(neg) > cap:
            neg = neg.iloc[rng.permutation(len(neg))[:cap]]
        train = pd.concat([pos, neg])
    return train.sample(frac=1, random_state=seed).reset_index(drop=True)


def make_pairs(train, n_pairs, rng):
    by_class = {c: train[train.y == c].s.tolist() for c in train.y.unique()}
    classes = [c for c in by_class if len(by_class[c]) >= 2]
    pairs = []
    for _ in range(n_pairs):
        c = classes[rng.integers(len(classes))]
        i, j = rng.choice(len(by_class[c]), 2, replace=False)
        pairs.append((by_class[c][i], by_class[c][j], 1.0))          # positive
        c1, c2 = rng.choice(len(classes), 2, replace=False)
        a = by_class[classes[c1]][rng.integers(len(by_class[classes[c1]]))]
        b = by_class[classes[c2]][rng.integers(len(by_class[classes[c2]]))]
        pairs.append((a, b, 0.0))                                    # negative
    return pairs


def train_setfit(train, epochs, n_pairs, body, seed):
    import torch
    import torch.nn.functional as F
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    from gap2idea.pipeline.gap_funnel import EmbeddingGapHead

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    model = SentenceTransformer(body)
    pairs = make_pairs(train, n_pairs, rng)
    print(f"  contrastive pairs: {len(pairs)} (half positive / half negative)")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    bs = 16
    model.train()
    for ep in range(epochs):
        rng.shuffle(pairs)
        tot = 0.0
        for i in range(0, len(pairs), bs):
            batch = pairs[i:i + bs]
            fa = model.tokenize([p[0] for p in batch])
            fb = model.tokenize([p[1] for p in batch])
            tgt = torch.tensor([p[2] for p in batch], dtype=torch.float32)
            ea = model(fa)["sentence_embedding"]
            eb = model(fb)["sentence_embedding"]
            loss = F.mse_loss(F.cosine_similarity(ea, eb), tgt)
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach())
        print(f"  epoch {ep+1}/{epochs}  contrastive loss {tot/(len(pairs)//bs+1):.4f}")

    model.eval()
    X = model.encode(train.s.tolist(), normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X, train.y.values)
    return EmbeddingGapHead(model, clf, body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-neg-ratio", type=float, default=3.0)
    ap.add_argument("--few-shot", type=int, default=0, help="N positives/class (SetFit's native regime); 0=full set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exclude = eval_paper_ids(ROOT)
    extra = ROOT / "data/bench_gap/train/gold_sentences.tsv"
    sents, labels, stats = build_dataset(ROOT, exclude, extra if extra.exists() else None, no_distant=True)
    assert not (set(stats["train_ids"]) & exclude), "LEAKAGE"
    train = make_train(sents, labels, args.max_neg_ratio, args.few_shot, args.seed)
    print(f"SetFit train rows: {len(train)}  {train.y.value_counts().to_dict()}  (few_shot={args.few_shot or 'off'})")

    gold, texts = load_gold()
    ev_sents, ev_glab = build_eval(gold, texts)

    print(f"\n>>> SetFit contrastive fine-tune of {BODY} ...")
    setfit = train_setfit(train, args.epochs, args.pairs, BODY, args.seed)

    from gap2idea.pipeline.gap_funnel import EmbeddingGapHead
    bge = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")

    print("\n=== PER-SENTENCE on slice (hybrid: rule wins else model) ===")
    rows = (per_sentence("AFTER  SetFit(bge)", ev_glab, bert_per_sentence_pred(setfit, ev_sents))
            + per_sentence("BEFORE bge+logreg", ev_glab, bert_per_sentence_pred(bge, ev_sents)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n=== END-TO-END vs clean gold (before vs after) ===")
    e2e = [end_to_end("AFTER  SetFit(bge)", gold, texts, setfit),
           end_to_end("BEFORE bge+logreg", gold, texts, bge),
           end_to_end("BEFORE rules-only", gold, texts, None)]
    print(pd.DataFrame(e2e).to_string(index=False))
    print("\nNOTE: eval = 19 gold gaps / 9 papers; ±1-2 gaps is noise.")


if __name__ == "__main__":
    main()
