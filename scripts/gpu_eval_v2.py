"""GPU-side full pipeline evaluation: Stage A v2 + SciBERT-FT + GAP/JUNK Stage C
on gold v2. Trains SciBERT-FT on our existing training data, runs extraction
with the v2 slicer, classifies with SciBERT, applies GAP/JUNK Stage C, reports.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    slice_terminal_regions, slice_with_midpaper_anchors,
    extract_gaps, token_containment, cue_label, _looks_like_sentence,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402

LABELS = ["future_work", "limitation", "none"]
L2I = {l: i for i, l in enumerate(LABELS)}
MATCH_TAU = 0.70


# === SciBERT-FT 3-class head training (same as test_scibert_gold.py) ===
class BertGapHead:
    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device

    def predict(self, sentences):
        import torch
        if not sentences:
            return []
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(sentences), 64):
                batch = sentences[i:i + 64]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               truncation=True, max_length=96).to(self.device)
                p = torch.softmax(self.model(**enc).logits, -1).cpu().numpy()
                for row in p:
                    j = int(row.argmax())
                    out.append((LABELS[j], float(row[j])))
        return out


def train_scibert(train_path, model_name="allenai/scibert_scivocab_uncased",
                  epochs=2, lr=3e-5, bs=24, max_neg_ratio=4.0, seed=0):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    df = pd.read_json(train_path, lines=True)
    pos = df[df.label != "none"]
    neg = df[df.label == "none"]
    cap = int(len(pos) * max_neg_ratio)
    if len(neg) > cap:
        neg = neg.iloc[rng.permutation(len(neg))[:cap]]
    train = pd.concat([pos, neg]).sample(frac=1, random_state=seed).reset_index(drop=True)
    y = train.label.map(L2I).values
    print(f"  train rows={len(train)}  {train.label.value_counts().to_dict()}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(dev)
    counts = np.bincount(y, minlength=3)
    w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32).to(dev)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sents = train.sentence.tolist()
    for ep in range(epochs):
        model.train(); order = rng.permutation(len(sents)); tot, nb = 0.0, 0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            enc = tok([sents[k] for k in idx], return_tensors="pt", padding=True,
                      truncation=True, max_length=96).to(dev)
            loss = lossf(model(**enc).logits, torch.tensor(y[idx], dtype=torch.long).to(dev))
            loss.backward(); opt.step(); opt.zero_grad()
            tot += float(loss.detach()); nb += 1
        print(f"  epoch {ep+1}/{epochs} loss {tot/max(1,nb):.4f}", flush=True)
    return BertGapHead(model, tok, dev)


def gold_match(sent, sub):
    for _, g in sub.iterrows():
        if max(token_containment(g["gap_sentence"], sent),
               token_containment(sent, g["gap_sentence"])) >= MATCH_TAU:
            return g["gap_id"]
    return None


def stage_a_metrics(papers, gold, slice_fn, name):
    slice_total = full_total = 0
    loc_70 = 0
    for pid, rec in papers.items():
        regs = slice_fn(rec["text"], blocks=rec.get("blocks"))
        slice_text = " ".join(s for r in regs for s in r.sentences)
        full_total += len(split_sentences(_cut_before_references(rec["text"])))
        slice_total += sum(len(r.sentences) for r in regs)
        for _, g in gold[gold["paper_id"] == pid].iterrows():
            if token_containment(g["gap_sentence"], slice_text) >= 0.70:
                loc_70 += 1
    return {
        "name": name, "drop": 1 - slice_total / max(1, full_total),
        "slice_per_paper": slice_total / len(papers),
        "loc@0.70": loc_70 / len(gold),
    }


def stage_b_extract(papers, head, slice_fn):
    rows = []
    for pid, rec in papers.items():
        gaps = []
        regs = slice_fn(rec["text"], blocks=rec.get("blocks"))
        slice_sents = [s for r in regs for s in r.sentences]
        if not slice_sents:
            continue
        preds = head.predict(slice_sents)
        for s, (lbl, p) in zip(slice_sents, preds):
            if lbl in ("future_work", "limitation") and p >= 0.5:
                if _looks_like_sentence(s):
                    gaps.append({"paper_id": pid, "gap_type": lbl, "source": "model",
                                  "section_type": "n/a", "gap_sentence": s})
        # Also add cue-rule hits (rules complement model)
        for r in regs:
            for s in r.sentences:
                rt = cue_label(s)
                if rt:
                    gaps.append({"paper_id": pid, "gap_type": rt, "source": "rule",
                                  "section_type": r.section_type, "gap_sentence": s})
        # Dedup by normalized sentence
        seen = set(); deduped = []
        for g in gaps:
            key = " ".join(g["gap_sentence"].lower().split())[:80]
            if key not in seen:
                seen.add(key); deduped.append(g)
        rows.extend(deduped)
    return pd.DataFrame(rows)


def stage_b_metrics(preds, gold):
    matched = set(); tp = 0
    for _, p in preds.iterrows():
        m = gold_match(p["gap_sentence"], gold[gold["paper_id"] == p["paper_id"]])
        if m: matched.add(m); tp += 1
    recall = len(matched) / len(gold)
    prec = tp / max(1, len(preds))
    f1 = 2 * recall * prec / max(1e-9, recall + prec)
    return {"n_pred": len(preds), "matched": len(matched),
            "recall": recall, "precision": prec, "F1": f1}


def main():
    gold = pd.read_csv(ROOT / "data/bench_gap/gold_sentences_v2.tsv", sep="\t", dtype={"paper_id": str})
    papers = {}
    for line in (ROOT / "data/scibert_prep/gold_papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            papers[str(rec["id"])] = rec
    print(f"Gold v2: {len(gold)} gaps / {len(papers)} papers", flush=True)

    # === Stage A both versions ===
    a_v1 = stage_a_metrics(papers, gold, slice_terminal_regions, "v1 (frozen)")
    a_v2 = stage_a_metrics(papers, gold, slice_with_midpaper_anchors, "v2 (midpaper)")
    print(f"\n=== Stage A versions on gold v2 ===")
    print(f"  v1: drop={a_v1['drop']*100:.1f}%  slice={a_v1['slice_per_paper']:.0f}/paper  loc@0.70={a_v1['loc@0.70']:.3f}")
    print(f"  v2: drop={a_v2['drop']*100:.1f}%  slice={a_v2['slice_per_paper']:.0f}/paper  loc@0.70={a_v2['loc@0.70']:.3f}")

    # === Train SciBERT-FT ===
    print(f"\n=== Training SciBERT-FT ===", flush=True)
    head = train_scibert(ROOT / "data/scibert_prep/train.jsonl")

    # === Stage A v1 + B ===
    print(f"\n=== Stage A v1 + B ===", flush=True)
    preds_v1 = stage_b_extract(papers, head, slice_terminal_regions)
    m1 = stage_b_metrics(preds_v1, gold)
    print(f"  n_pred={m1['n_pred']}  matched={m1['matched']}/{len(gold)}  "
          f"R={m1['recall']:.3f}  P={m1['precision']:.3f}  F1={m1['F1']:.3f}", flush=True)

    # === Stage A v2 + B ===
    print(f"\n=== Stage A v2 + B ===", flush=True)
    preds_v2 = stage_b_extract(papers, head, slice_with_midpaper_anchors)
    m2 = stage_b_metrics(preds_v2, gold)
    print(f"  n_pred={m2['n_pred']}  matched={m2['matched']}/{len(gold)}  "
          f"R={m2['recall']:.3f}  P={m2['precision']:.3f}  F1={m2['F1']:.3f}", flush=True)

    # Save predictions for off-GPU Stage C
    preds_v1.to_csv(ROOT / "data/scibert_prep/scibert_gold_gaps_v1slice.tsv", sep="\t", index=False)
    preds_v2.to_csv(ROOT / "data/scibert_prep/scibert_gold_gaps_v2slice.tsv", sep="\t", index=False)
    print(f"\nSaved predictions for both slicer versions to data/scibert_prep/")


if __name__ == "__main__":
    main()
