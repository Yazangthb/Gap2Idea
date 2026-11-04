# -*- coding: utf-8 -*-
"""Train the Stage-B gap detector on the LLM-ensemble gold, with a proper
paper-level, domain-stratified train/val/test split.

- Split UNIT = paper (no sentence leakage), stratified by domain (arxiv_cs / biomed_sal),
  fixed seed. Saved to data/gold/splits.json.
- Task = per-sentence 4-class {none, limitation, future_work, research_gap}.
  (Funnel-compatible: extract_gaps only fires on limitation/future_work.)
- Model = frozen sentence-encoder + multinomial LogisticRegression (EmbeddingGapHead).
  Sweeps several cached encoders; tunes the binary-gap decision threshold on VAL;
  reports TEST. Saves the best head (bundle: clf + encoder_name) + metrics + splits.

  python scripts/training/train_gold_head.py                         # default encoder sweep
  python scripts/training/train_gold_head.py --encoders bge-small    # one encoder (fast)

Writes: data/gold/splits.json, data/gold/gap_head_gold.joblib (+ .meta.json),
docs/experiments/gold_detector_results.md.
"""
from __future__ import annotations
import sys, json, argparse, random
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
import numpy as np
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import EmbeddingGapHead

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
SPLITS = ROOT / "data" / "gold" / "splits.json"
OUTHEAD = ROOT / "data" / "gold" / "gap_head_gold.joblib"
OUTDOC = ROOT / "docs" / "experiments" / "gold_detector_results.md"
OLDHEAD = ROOT / "data" / "pipeline_head.joblib"
CLASSES = ["none", "limitation", "future_work", "research_gap"]
GAP = CLASSES[1:]
ENC = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "pubmedbert": "pritamdeka/S-PubMedBert-MS-MARCO",
    "specter": "sentence-transformers/allenai-specter",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
}


def make_splits(domain_of, seed, val, test):
    by_dom = defaultdict(list)
    for pid, d in domain_of.items():
        by_dom[d].append(pid)
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}
    for d, pids in by_dom.items():
        pids = sorted(pids); rng.shuffle(pids)
        n = len(pids); nt = int(round(n * test)); nv = int(round(n * val))
        split["test"] += pids[:nt]
        split["val"] += pids[nt:nt + nv]
        split["train"] += pids[nt + nv:]
    return split


def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return P, R, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="bge-small,pubmedbert,specter,bge-base")
    ap.add_argument("--neg-ratio", type=float, default=4.0)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    # gold per (paper, idx) -> label; domain per paper
    gold = {}; domain = {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if not ln.strip(): continue
        g = json.loads(ln)
        gold[(g["paper_id"], g["idx"])] = g["label"]
        domain[g["paper_id"]] = g["domain"]
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in domain}

    split = make_splits(domain, args.seed, args.val, args.test)
    SPLITS.write_text(json.dumps(split, indent=2), encoding="utf-8")
    paper_split = {pid: s for s, pids in split.items() for pid in pids}
    print("papers:", {s: len(v) for s, v in split.items()},
          "| by domain:", {s: dict(Counter(domain[p] for p in v)) for s, v in split.items()})

    # per-sentence rows per split
    def rows_for(pids):
        S, Y, D = [], [], []
        for pid in pids:
            p = corp[pid]
            for i, s in enumerate(p["sents"]):
                S.append(s); Y.append(gold.get((pid, i), "none")); D.append(domain[pid])
        return S, Y, D
    trS, trY, _ = rows_for(split["train"])
    vaS, vaY, vaD = rows_for(split["val"])
    teS, teY, teD = rows_for(split["test"])

    # balance train negatives
    rng = np.random.default_rng(args.seed)
    pos_idx = [i for i, y in enumerate(trY) if y != "none"]
    neg_idx = [i for i, y in enumerate(trY) if y == "none"]
    cap = int(len(pos_idx) * args.neg_ratio)
    if len(neg_idx) > cap:
        neg_idx = list(rng.permutation(neg_idx)[:cap])
    keep = sorted(pos_idx + neg_idx)
    trS = [trS[i] for i in keep]; trY = [trY[i] for i in keep]
    print(f"train rows: {len(trS)} {dict(Counter(trY))} | val sents: {len(vaS)} | test sents: {len(teS)}")

    from sklearn.linear_model import LogisticRegression

    def binmetrics(proba, classes, Y, D, tau):
        none_j = list(classes).index("none")
        gap_js = [list(classes).index(c) for c in GAP if c in classes]
        tp = fp = fn = 0
        by_dom = defaultdict(lambda: [0, 0, 0])
        by_type = defaultdict(lambda: [0, 0])  # [detected, total]
        type_correct = tot_gap_pred_true = 0
        for row, y, d in zip(proba, Y, D):
            pgap = 1.0 - row[none_j]
            pred_gap = pgap >= tau
            pred_type = classes[gap_js[int(np.argmax([row[j] for j in gap_js]))]] if gap_js else "none"
            true_gap = y != "none"
            if pred_gap and true_gap: tp += 1; by_dom[d][0] += 1
            elif pred_gap and not true_gap: fp += 1; by_dom[d][1] += 1
            elif not pred_gap and true_gap: fn += 1; by_dom[d][2] += 1
            if true_gap:
                by_type[y][1] += 1
                if pred_gap:
                    by_type[y][0] += 1
                    tot_gap_pred_true += 1
                    type_correct += (pred_type == y)
        P, R, F = prf(tp, fp, fn)
        return dict(P=P, R=R, F=F, tp=tp, fp=fp, fn=fn,
                    by_dom={k: prf(v[0], v[1], v[2]) for k, v in by_dom.items()},
                    by_type={k: (v[0] / v[1] if v[1] else 0, v[1]) for k, v in by_type.items()},
                    type_acc=type_correct / tot_gap_pred_true if tot_gap_pred_true else 0)

    results = {}
    encoders = [e.strip() for e in args.encoders.split(",")]
    for name in encoders:
        if name not in ENC:
            print("skip unknown encoder", name); continue
        print(f"\n=== {name} ({ENC[name]}) ===", flush=True)
        enc = EmbeddingGapHead.load_encoder(ENC[name])
        Xtr = enc.encode(trS, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        clf = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced").fit(Xtr, trY)
        cls = list(clf.classes_)
        Xva = enc.encode(vaS, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        Pva = clf.predict_proba(Xva)
        # tune tau on val for binary F1
        best = (0.5, -1)
        for tau in np.arange(0.20, 0.86, 0.05):
            f = binmetrics(Pva, cls, vaY, vaD, tau)["F"]
            if f > best[1]: best = (round(float(tau), 2), f)
        tau = best[0]
        Xte = enc.encode(teS, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        Pte = clf.predict_proba(Xte)
        vm = binmetrics(Pva, cls, vaY, vaD, tau)
        tm = binmetrics(Pte, cls, teY, teD, tau)
        results[name] = dict(tau=tau, val=vm, test=tm, clf=clf, enc=enc, enc_id=ENC[name])
        print(f"  tau={tau} | VAL F1={vm['F']:.3f} | TEST P={tm['P']:.3f} R={tm['R']:.3f} F1={tm['F']:.3f} "
              f"| type-acc={tm['type_acc']:.3f}", flush=True)

    if not results:
        sys.exit("no encoders trained")
    best_name = max(results, key=lambda n: results[n]["val"]["F"])
    best = results[best_name]
    EmbeddingGapHead(best["enc"], best["clf"], best["enc_id"]).save(OUTHEAD)
    OUTHEAD.with_suffix(".meta.json").write_text(json.dumps(
        {"encoder": best["enc_id"], "tau": best["tau"], "classes": list(best["clf"].classes_),
         "seed": args.seed, "splits_file": str(SPLITS.name),
         "test": {k: best["test"][k] for k in ("P", "R", "F", "type_acc")}}, indent=2), encoding="utf-8")

    # ---- old head comparison (bge-small, classes none/lim/fw) ----
    old = None
    if OLDHEAD.exists():
        import joblib
        oclf = joblib.load(OLDHEAD)
        oenc = results.get("bge-small", {}).get("enc") or EmbeddingGapHead.load_encoder("BAAI/bge-small-en-v1.5")
        Xte = oenc.encode(teS, normalize_embeddings=True, batch_size=128, show_progress_bar=False)
        Po = oclf.predict_proba(Xte)
        old = binmetrics(Po, list(oclf.classes_), teY, teD, 0.5)

    # ---- report ----
    L = ["# Stage-B detector trained on the gap gold\n",
         f"Paper-level split (seed {args.seed}), stratified by domain: "
         f"train {len(split['train'])} / val {len(split['val'])} / test {len(split['test'])} papers. "
         "Per-sentence 4-class (none/limitation/future_work/research_gap); frozen encoder + "
         "multinomial logreg; binary-gap threshold tuned on val. **Labels are LLM-ensemble gold "
         "(not yet human-certified).**\n",
         "## Encoder sweep (binary gap = any type)\n",
         "| encoder | tau | VAL F1 | TEST P | TEST R | TEST F1 | type-acc |\n|---|---|---|---|---|---|---|"]
    for n in encoders:
        if n not in results: continue
        r = results[n]; t = r["test"]
        star = " **(best)**" if n == best_name else ""
        L.append(f"| {n}{star} | {r['tau']} | {r['val']['F']:.3f} | {t['P']:.3f} | {t['R']:.3f} "
                 f"| {t['F']:.3f} | {t['type_acc']:.3f} |")
    if old:
        L.append(f"| _old pipeline_head_ | 0.50 | — | {old['P']:.3f} | {old['R']:.3f} | {old['F']:.3f} | — |")

    t = best["test"]
    L.append(f"\n## Best: {best_name} (test)\n")
    L.append("Per-domain binary-gap P/R/F1 (test):")
    for d, (P, R, F) in sorted(t["by_dom"].items()):
        L.append(f"- {d}: P={P:.3f} R={R:.3f} F1={F:.3f}")
    L.append("\nPer-type detection recall (test): " + ", ".join(
        f"{k} {v[0]*100:.1f}% (n={v[1]})" for k, v in sorted(t["by_type"].items())))
    L.append(f"\nType accuracy among correctly-detected gaps: {t['type_acc']*100:.1f}%")
    L.append("\n### Notes\n"
             "- This is a **standalone per-sentence detector** at realistic prevalence (all test "
             "sentences), not the Stage-A-gated funnel; it isolates Stage B's ceiling.\n"
             "- 'type-acc' = of gaps correctly flagged, fraction with the right type.\n"
             f"- Best head saved to `{OUTHEAD.name}` (bundle: clf + encoder); splits in `{SPLITS.name}`.\n"
             "- Test labels are LLM-gold; the human pass will re-certify the test set.")
    OUTDOC.parent.mkdir(parents=True, exist_ok=True)
    OUTDOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n" + "\n".join(L[-12:]))
    print("\nwrote", OUTHEAD, "|", OUTDOC, "|", SPLITS)


if __name__ == "__main__":
    main()
