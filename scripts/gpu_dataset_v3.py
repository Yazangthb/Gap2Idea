"""GPU-side full precision-first pipeline on v2-extracted papers.

  Stage A v2 (slice_with_midpaper_anchors) → Stage B (SciBERT-FT) →
  saves predictions for off-GPU Stage C strict+confirm via gpt-4o.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    slice_with_midpaper_anchors, _looks_like_sentence, cue_label,
)

LABELS = ["future_work", "limitation", "none"]
L2I = {l: i for i, l in enumerate(LABELS)}


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


def train_scibert(train_path, epochs=2, lr=3e-5, bs=24, max_neg_ratio=4.0, seed=0):
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
    tok = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased", num_labels=3).to(dev)
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


def main():
    # Load v2 papers
    papers_path = ROOT / "data/dataset_v2/papers_v2.jsonl"
    papers = []
    for line in papers_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            papers.append(json.loads(line))
    print(f"Loaded {len(papers)} papers (v2-extracted)")

    # === Train SciBERT-FT ===
    print(f"\n=== Training SciBERT-FT ===", flush=True)
    head = train_scibert(ROOT / "data/scibert_prep/train.jsonl")

    # === Stage A v2 + B ===
    print(f"\n=== Stage A v2 + Stage B (SciBERT-FT) on v2 papers ===", flush=True)
    all_preds = []
    for rec in papers:
        pid = rec["id"]
        regs = slice_with_midpaper_anchors(rec["text"], blocks=rec.get("blocks"))
        slice_sents = [s for r in regs for s in r.sentences if _looks_like_sentence(s)]
        if not slice_sents:
            continue
        preds = head.predict(slice_sents)
        # Stage B keeps: rule hits OR model-positive with confidence >= 0.5
        for s, (lbl, p) in zip(slice_sents, preds):
            keep = False
            source = ""
            if lbl in ("future_work", "limitation") and p >= 0.5:
                keep = True; source = "model"
            rt = cue_label(s)
            if rt:
                keep = True; source = "rule+model" if source else "rule"
                lbl = rt
            if keep:
                all_preds.append({"paper_id": pid, "gap_type": lbl, "source": source,
                                   "confidence": round(p, 3), "gap_sentence": s})
        print(f"  {pid}: slice={len(slice_sents)}  emitted={sum(1 for x in all_preds if x['paper_id']==pid)}",
              flush=True)

    # Dedup
    df = pd.DataFrame(all_preds)
    if not df.empty:
        seen = set(); rows = []
        for _, r in df.iterrows():
            key = (r["paper_id"], " ".join(r["gap_sentence"].lower().split())[:60])
            if key in seen:
                continue
            seen.add(key); rows.append(r.to_dict())
        df = pd.DataFrame(rows)
    out = ROOT / "data/dataset_v3/scibert_preds_v2papers.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)
    print(f"\nStage B predictions: {len(df)} from {df['paper_id'].nunique()} papers")
    print(f"  per paper: {df.groupby('paper_id').size().to_dict()}")
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
