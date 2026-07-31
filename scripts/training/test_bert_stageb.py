"""Test a fine-tuned BERT as the Stage-B classifier vs the frozen bge+logreg head.

Fair, leakage-safe comparison:
  * Train data  = the SAME self-distilled sentences/labels train_gap_head builds
                  from runs/* (the 10 gold papers are excluded).
  * Eval        = the clean gold (data/bench_gap), evaluated TWO ways, identical
                  for both models:
                    - per-sentence on the Stage-A slice (isolates classifier quality)
                    - end-to-end through extract_gaps (the real funnel output)
  * BERT is fine-tuned end-to-end (transformer weights update) — the real "test
    BERT", unlike the current frozen-embedding + logreg head.

Usage:
    python scripts/training/test_bert_stageb.py --model distilbert-base-uncased --epochs 4
    python scripts/training/test_bert_stageb.py --model distilbert-base-uncased --distant
    python scripts/training/test_bert_stageb.py --model allenai/scibert_scivocab_uncased
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    cue_label, extract_gaps, slice_terminal_regions, token_containment, _looks_like_sentence,
)
from train_gap_head import build_dataset, eval_paper_ids  # noqa: E402
from bench_gap_recall import load as load_gold  # noqa: E402

LABELS = ["future_work", "limitation", "none"]
L2I = {l: i for i, l in enumerate(LABELS)}
TARGET = ["limitation", "future_work"]
TAU = 0.80


# --------------------------------------------------------------------------- eval data
def best_gold_type(sent, gold_rows):
    best_t, best = None, 0.0
    for _, g in gold_rows.iterrows():
        c = max(token_containment(g["gap_sentence"], sent), token_containment(sent, g["gap_sentence"]))
        if c > best:
            best, best_t = c, g["gap_type"]
    return best_t if best >= TAU else "none"


def build_eval(gold, texts):
    sents, glab = [], []
    for pid, rec in texts.items():
        gp = gold[gold["paper_id"] == pid]
        for r in slice_terminal_regions(rec["text"], blocks=rec["blocks"]):
            for s in r.sentences:
                sents.append(s)
                glab.append(best_gold_type(s, gp))
    return sents, glab


# --------------------------------------------------------------------------- BERT
class BertGapHead:
    """Duck-types EmbeddingGapHead.predict so it drops into extract_gaps()."""

    def __init__(self, model, tok, max_len=96):
        self.model, self.tok, self.max_len = model, tok, max_len

    def predict(self, sentences):
        import torch
        if not sentences:
            return []
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), 64):
                batch = sentences[i:i + 64]
                enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.max_len)
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=-1).numpy()
                for row in p:
                    j = int(row.argmax())
                    out.append((LABELS[j], float(row[j])))
        return out


def train_bert(model_name, sents, labels, epochs, lr, seed, max_neg_ratio):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({"s": sents, "y": labels})
    pos = df[df.y != "none"]
    neg = df[df.y == "none"]
    cap = int(len(pos) * max_neg_ratio)
    if len(neg) > cap:
        neg = neg.iloc[rng.permutation(len(neg))[:cap]]
    train = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"  BERT train rows: {len(train)}  {train.y.value_counts().to_dict()}")

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    y = train.y.map(L2I).values
    # balanced class weights
    counts = np.bincount(y, minlength=3)
    w = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))), dtype=torch.float32)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    X = train.s.tolist()
    bs = 16
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train))
        tot = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([X[k] for k in idx], return_tensors="pt", padding=True, truncation=True, max_length=96)
            yb = torch.tensor(y[idx], dtype=torch.long)
            logits = model(**enc).logits
            loss = lossf(logits, yb)
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach())
        print(f"  epoch {ep+1}/{epochs}  loss {tot/(len(order)//bs+1):.3f}")
    return BertGapHead(model, tok)


# --------------------------------------------------------------------------- metrics
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def per_sentence(name, glab, pred):
    rows = []
    for t in TARGET + ["ANY_GAP"]:
        if t == "ANY_GAP":
            tp = sum(g != "none" and p != "none" for g, p in zip(glab, pred))
            fp = sum(g == "none" and p != "none" for g, p in zip(glab, pred))
            fn = sum(g != "none" and p == "none" for g, p in zip(glab, pred))
        else:
            tp = sum(g == t and p == t for g, p in zip(glab, pred))
            fp = sum(p == t and g != t for g, p in zip(glab, pred))
            fn = sum(g == t and p != t for g, p in zip(glab, pred))
        P, R, F = prf(tp, fp, fn)
        rows.append({"model": name, "type": t, "P": P, "R": R, "f1": F})
    return rows


def end_to_end(name, gold, texts, head, thr=0.6):
    preds = {pid: extract_gaps(pid, rec["text"], blocks=rec["blocks"], head=head, mode="hybrid",
                               model_threshold=thr) for pid, rec in texts.items()}
    npred = sum(len(v) for v in preds.values())
    matched, tok = set(), 0
    for _, g in gold.iterrows():
        for pr in preds.get(g["paper_id"], []):
            if max(token_containment(g["gap_sentence"], pr["gap_sentence"]),
                   token_containment(pr["gap_sentence"], g["gap_sentence"])) >= TAU:
                matched.add(g["gap_id"])
                if pr["gap_type"] == g["gap_type"]:
                    tok += 1
                break
    return {"model": name, "n_pred": npred, "gold": len(gold), "matched": len(matched),
            "recall": round(len(matched)/len(gold), 3), "type_acc": round(tok/max(1, len(matched)), 3)}


def bert_per_sentence_pred(head, sents, thr=0.6):
    """Mimic hybrid extract logic at the sentence level: rule wins, else model>=thr."""
    preds = head.predict(sents)
    out = []
    for s, (lbl, p) in zip(sents, preds):
        rt = cue_label(s)
        if rt:
            out.append(rt)
        elif lbl in TARGET and p >= thr and _looks_like_sentence(s):
            out.append(lbl)
        else:
            out.append("none")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--distant", action="store_true", help="use distant-supervision weak positives too")
    ap.add_argument("--max-neg-ratio", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    exclude = eval_paper_ids(ROOT)
    extra = ROOT / "data/bench_gap/train/gold_sentences.tsv"
    sents, labels, stats = build_dataset(ROOT, exclude, extra if extra.exists() else None,
                                         no_distant=not args.distant)
    assert not (set(stats["train_ids"]) & exclude), "LEAKAGE"
    print(f"train papers={stats['papers']} pos_fut={stats['pos_fut']} pos_lim={stats['pos_lim']} "
          f"neg={stats['neg']} distant={args.distant}")

    gold, texts = load_gold()
    ev_sents, ev_glab = build_eval(gold, texts)
    print(f"eval slice sentences={len(ev_sents)} gold-in-slice={sum(g!='none' for g in ev_glab)}")

    # ---- BERT ----
    print(f"\n>>> fine-tuning {args.model} ...")
    bert = train_bert(args.model, sents, labels, args.epochs, args.lr, args.seed, args.max_neg_ratio)

    # ---- baseline frozen bge+logreg ----
    from gap2idea.pipeline.gap_funnel import EmbeddingGapHead
    bge = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")

    rows = (per_sentence(f"BERT:{args.model.split('/')[-1]}", ev_glab, bert_per_sentence_pred(bert, ev_sents))
            + per_sentence("bge+logreg", ev_glab, bert_per_sentence_pred(bge, ev_sents)))
    print("\n=== PER-SENTENCE on slice (hybrid: rule wins else model) ===")
    print(pd.DataFrame(rows).to_string(index=False))

    e2e = [end_to_end(f"BERT:{args.model.split('/')[-1]}", gold, texts, bert),
           end_to_end("bge+logreg", gold, texts, bge),
           end_to_end("rules-only", gold, texts, None)]
    print("\n=== END-TO-END vs clean gold ===")
    print(pd.DataFrame(e2e).to_string(index=False))
    print("\nNOTE: eval set is 19 gold gaps / 9 papers — differences of 1-2 gaps are within noise.")


if __name__ == "__main__":
    main()
