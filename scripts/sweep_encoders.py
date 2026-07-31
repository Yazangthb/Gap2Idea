"""Enhance Stage B by sweeping the FROZEN encoder (features), classifier fixed.

The earlier finding (logreg ≈ DistilBERT ≈ SetFit) was about the CLASSIFIER on
bge features. A different frozen encoder changes the FEATURES — a separate axis,
and the literature's lever (SciBERT/SPECTER are domain encoders). All variants
are frozen-encode + logreg, so scalability is identical. Data = runs/* self-
distilled + 1500 ACL mandated-Limitations positives (the shipped recipe).

Picks the best encoder by end-to-end recall (tie: limitation recall, then fewer
preds) and saves it as data/gap_head.joblib.

    python scripts/sweep_encoders.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from gap2idea.pipeline.gap_funnel import EmbeddingGapHead  # noqa: E402
from train_gap_head import build_dataset, eval_paper_ids  # noqa: E402
from test_bert_stageb import build_eval, end_to_end, bert_per_sentence_pred  # noqa: E402
from bench_gap_recall import load as load_gold  # noqa: E402

ENCODERS = [
    "BAAI/bge-small-en-v1.5",            # current (33M, general)
    "BAAI/bge-base-en-v1.5",             # bigger general (109M)
    "sentence-transformers/all-mpnet-base-v2",   # strong general (110M)
    "sentence-transformers/allenai-specter",     # SCIENTIFIC domain encoder
]
LIM_CAP, BASE_NEG_CAP, SEED = 1500, 2500, 0


def recall_of(pred, glab, t):
    tp = sum(g == t and p == t for g, p in zip(glab, pred))
    fn = sum(g == t and p != t for g, p in zip(glab, pred))
    return round(tp / (tp + fn), 3) if tp + fn else 0.0


def main():
    rng = np.random.default_rng(SEED)
    exclude = eval_paper_ids(ROOT)
    extra = ROOT / "data/bench_gap/train/gold_sentences.tsv"
    sents, labels, stats = build_dataset(ROOT, exclude, extra if extra.exists() else None, no_distant=True)
    train_ids = sorted(stats["train_ids"])
    df = pd.DataFrame({"s": sents, "y": labels})
    base_pos = df[df.y != "none"]
    base_neg = df[df.y == "none"]
    if len(base_neg) > BASE_NEG_CAP:
        base_neg = base_neg.iloc[rng.permutation(len(base_neg))[:BASE_NEG_CAP]]
    lim = pd.read_csv(ROOT / "data/acl_limitations.tsv", sep="\t")["gap_sentence"].astype(str).tolist()
    rng.shuffle(lim); lim = lim[:LIM_CAP]

    train_sents = base_pos.s.tolist() + base_neg.s.tolist() + lim
    train_y = list(base_pos.y) + ["none"] * len(base_neg) + ["limitation"] * len(lim)

    gold, texts = load_gold()
    ev_sents, ev_glab = build_eval(gold, texts)

    rows, heads = [], {}
    for name in ENCODERS:
        try:
            enc = EmbeddingGapHead.load_encoder(name)
            X = enc.encode(train_sents, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
            clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X, train_y)
            head = EmbeddingGapHead(enc, clf, name)
            e = end_to_end("cfg", gold, texts, head)
            pred = bert_per_sentence_pred(head, ev_sents)
            r = {"encoder": name.split("/")[-1], "recall": e["recall"], "type_acc": e["type_acc"],
                 "n_pred": e["n_pred"], "lim_R": recall_of(pred, ev_glab, "limitation"),
                 "fut_R": recall_of(pred, ev_glab, "future_work")}
            rows.append(r); heads[name] = head
            print(f"  {r['encoder']:<26} recall={r['recall']:.3f} type_acc={r['type_acc']:.2f} "
                  f"n_pred={r['n_pred']:<3} lim_R={r['lim_R']:.3f} fut_R={r['fut_R']:.3f}")
        except Exception as ex:
            print(f"  {name}: FAILED {ex}")

    res = pd.DataFrame(rows).sort_values(["recall", "lim_R"], ascending=False)
    print("\n=== ENCODER SWEEP (ACL-augmented, frozen + logreg) ===")
    print(res.to_string(index=False))

    best = max(rows, key=lambda r: (r["recall"], r["lim_R"], -r["n_pred"]))
    bname = next(n for n in ENCODERS if n.split("/")[-1] == best["encoder"])
    cur = next((r for r in rows if r["encoder"] == "bge-small-en-v1.5"), None)
    print(f"\ncurrent bge-small: recall={cur['recall']} lim_R={cur['lim_R']}" if cur else "")
    print(f"BEST encoder: {best['encoder']} (recall={best['recall']}, lim_R={best['lim_R']})")
    if best["encoder"] != "bge-small-en-v1.5":
        heads[bname].save(ROOT / "data/gap_head.joblib")
        meta = {"encoder": bname, "source": "runs/* + ACL LimGen (lim_cap=1500)",
                "train_ids": train_ids, "excluded_eval_ids": sorted(exclude), "bench": best}
        (ROOT / "data/gap_head.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"-> saved {best['encoder']} head as data/gap_head.joblib")
    else:
        print("-> current bge-small remains best; no change.")


if __name__ == "__main__":
    main()
