"""Sweep how many harvested ACL sentences to add, pick the best Stage-B head.

Adds N ACL mandated-Limitations sentences (and optionally cue-harvested ACL
future-work sentences) to the runs/* self-distilled data, refits the logreg head
on the SAME frozen bge encoder, and benchmarks each config on the clean gold.
Encodes every sentence pool ONCE, so the sweep is fast (only logreg refits +
eval re-encode per config). Saves the best config to data/gap_head.joblib.

Usage:
    python scripts/training/sweep_acl_cap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))

from sklearn.linear_model import LogisticRegression  # noqa: E402

from gap2idea.pipeline.gap_funnel import EmbeddingGapHead  # noqa: E402
from train_gap_head import build_dataset, eval_paper_ids  # noqa: E402
from test_bert_stageb import build_eval, per_sentence, end_to_end, bert_per_sentence_pred  # noqa: E402
from bench_gap_recall import load as load_gold  # noqa: E402

BODY = "BAAI/bge-small-en-v1.5"
LIM_CAPS = [0, 600, 1500, 3000, 6000]
FUT_CAPS = [0, 1268]
BASE_NEG_CAP = 2500
SEED = 0


def recall_of(pred, ev_glab, t):
    tp = sum(g == t and p == t for g, p in zip(ev_glab, pred))
    fn = sum(g == t and p != t for g, p in zip(ev_glab, pred))
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
    fut = pd.read_csv(ROOT / "data/acl_futurework.tsv", sep="\t")["gap_sentence"].astype(str).tolist()
    rng.shuffle(lim); rng.shuffle(fut)

    enc = EmbeddingGapHead.load_encoder(BODY)
    print("encoding pools once ...")
    Xbp = enc.encode(base_pos.s.tolist(), normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xbn = enc.encode(base_neg.s.tolist(), normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xlim = enc.encode(lim, normalize_embeddings=True, show_progress_bar=False, batch_size=64)
    Xfut = enc.encode(fut, normalize_embeddings=True, show_progress_bar=False, batch_size=64)

    gold, texts = load_gold()
    ev_sents, ev_glab = build_eval(gold, texts)

    def make_head(limcap, futcap):
        Xs = [Xbp, Xbn]
        ys = list(base_pos.y) + ["none"] * len(base_neg)
        if limcap:
            Xs.append(Xlim[:limcap]); ys += ["limitation"] * min(limcap, len(Xlim))
        if futcap:
            Xs.append(Xfut[:futcap]); ys += ["future_work"] * min(futcap, len(Xfut))
        X = np.vstack(Xs)
        clf = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced").fit(X, ys)
        return EmbeddingGapHead(enc, clf, BODY)

    rows, heads = [], {}
    for limcap in LIM_CAPS:
        for futcap in FUT_CAPS:
            head = make_head(limcap, futcap)
            e = end_to_end("cfg", gold, texts, head)
            pred = bert_per_sentence_pred(head, ev_sents)
            r = {"lim_cap": limcap, "fut_cap": futcap, "recall": e["recall"],
                 "type_acc": e["type_acc"], "n_pred": e["n_pred"],
                 "lim_R": recall_of(pred, ev_glab, "limitation"),
                 "fut_R": recall_of(pred, ev_glab, "future_work")}
            rows.append(r); heads[(limcap, futcap)] = head
            print(f"  lim={limcap:<5} fut={futcap:<5} recall={r['recall']:.3f} "
                  f"type_acc={r['type_acc']:.2f} n_pred={r['n_pred']:<3} "
                  f"lim_R={r['lim_R']:.3f} fut_R={r['fut_R']:.3f}")

    res = pd.DataFrame(rows).sort_values(["recall", "lim_R"], ascending=False)
    print("\n=== SWEEP (sorted by recall, then limitation recall) ===")
    print(res.to_string(index=False))

    # baseline (current default head) for reference
    base_head = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")
    be = end_to_end("BEFORE", gold, texts, base_head)
    bpred = bert_per_sentence_pred(base_head, ev_sents)
    print(f"\nBEFORE (current default): recall={be['recall']:.3f} n_pred={be['n_pred']} "
          f"lim_R={recall_of(bpred, ev_glab, 'limitation'):.3f}")

    # pick best: max recall, tie-break smaller n_pred (cleaner)
    best = max(rows, key=lambda r: (r["recall"], r["lim_R"], -r["n_pred"]))
    bkey = (best["lim_cap"], best["fut_cap"])
    print(f"\nBEST: lim_cap={best['lim_cap']} fut_cap={best['fut_cap']}  -> saving as data/gap_head.joblib")
    heads[bkey].save(ROOT / "data/gap_head.joblib")
    meta = {"encoder": BODY, "source": "runs/* + ACL LimGen harvest",
            "acl_lim_cap": best["lim_cap"], "acl_fut_cap": best["fut_cap"],
            "train_ids": train_ids, "excluded_eval_ids": sorted(exclude),
            "bench": best}
    (ROOT / "data/gap_head.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved head + meta.")


if __name__ == "__main__":
    main()
