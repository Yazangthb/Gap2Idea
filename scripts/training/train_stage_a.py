# -*- coding: utf-8 -*-
"""Train a cheap, recall-first Stage-A localizer on the gold and compare it to the
current rule-based Stage A.

Stage A's job: keep the gap-bearing sentences, drop the rest, using only CHEAP
structural features (no embeddings) — position in paper, section, length. It learns
section weights from the corpus instead of hardcoded regex, and for arXiv (no section
labels) it derives a coarse section from inline headings.

Target per sentence = is-any-gold-gap (binary). Trained recall-first; the operating
threshold is tuned on VAL to hit a target recall, then retention (% sentences kept)
and recall are reported on TEST, per domain, vs the rule-based slice.

  python scripts/training/train_stage_a.py                 # target recall 0.95
  python scripts/training/train_stage_a.py --target 0.90

Writes data/gold/stage_a_model.joblib (+ .meta.json) and docs/experiments/stage_a_results.md.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts" / "dataset")); sys.path.insert(0, str(ROOT / "scripts" / "bench"))
import numpy as np
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import (
    classify_grobid_heading, SECTION_HEAD_RE, slice_section_aware, token_containment,
)
from benchmark_gold import sal_blocks

GOLD = ROOT / "data" / "gold" / "gap_gold_final.jsonl"
SPLITS = ROOT / "data" / "gold" / "splits.json"
OUT = ROOT / "data" / "gold" / "stage_a_model.joblib"
DOC = ROOT / "docs" / "experiments" / "stage_a_results.md"

# unified section vocabulary (biomed SAL codes + arxiv-derived classes)
SEC_VOCAB = ["DISCUSS", "CONCL", "INTRO", "METHODS", "RESULTS", "ABSTRACT", "TITLE",
             "discussion", "future_work", "limitations", "introduction", "background",
             "BODY", "OTHER"]


def arxiv_sections(sents):
    """Per-sentence coarse section for arXiv raw text: scan for inline headings
    (short line matching a section regex) and carry the classified label forward."""
    out = []
    cur = "BODY"
    for s in sents:
        w = s.split()
        if len(w) <= 12 and SECTION_HEAD_RE.search(s):
            c = classify_grobid_heading(s)
            cur = c if c != "other" else "BODY"
        out.append(cur)
    return out


def feats(p, domain):
    sents = p["sents"]; n = len(sents)
    secs = p["secs"] if (domain == "biomed_sal" and p.get("secs")) else arxiv_sections(sents)
    rows = []
    for i, s in enumerate(sents):
        pos = i / max(1, n - 1)
        sec = secs[i] if secs[i] in SEC_VOCAB else "OTHER"
        row = {"pos": pos, "pos2": pos * pos, "dist_end": (n - 1 - i) / max(1, n - 1),
               "last30": 1.0 if i >= n - 30 else 0.0, "first10pct": 1.0 if pos < 0.10 else 0.0,
               "loglen": np.log1p(len(s)), "dom_arxiv": 1.0 if domain == "arxiv_cs" else 0.0}
        for v in SEC_VOCAB:
            row[f"sec_{v}"] = 1.0 if sec == v else 0.0
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.95, help="target recall tuned on val")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    split = json.loads(SPLITS.read_text())
    gold = defaultdict(set); dom = {}
    for ln in GOLD.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            g = json.loads(ln); gold[g["paper_id"]].add(g["idx"]); dom[g["paper_id"]] = g["domain"]
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in dom}

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    def build(pids):
        X, y, D, meta = [], [], [], []
        for pid in pids:
            p = corp[pid]; d = dom[pid]
            for i, r in enumerate(feats(p, d)):
                X.append(r); y.append(1 if i in gold[pid] else 0); D.append(d); meta.append((pid, i))
        return X, np.array(y), D, meta

    Xtr, ytr, _, _ = build(split["train"])
    Xva, yva, Dva, _ = build(split["val"])
    Xte, yte, Dte, _ = build(split["test"])
    dv = DictVectorizer(sparse=False)
    Atr = dv.fit_transform(Xtr); Ava = dv.transform(Xva); Ate = dv.transform(Xte)
    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(Atr, ytr)
    pos = list(clf.classes_).index(1)
    pva = clf.predict_proba(Ava)[:, pos]; pte = clf.predict_proba(Ate)[:, pos]

    # tune threshold on val to hit target recall
    order = np.argsort(-pva)
    cum = np.cumsum(yva[order]); totpos = yva.sum()
    tau = 0.5
    for k in range(len(order)):
        if cum[k] / totpos >= args.target:
            tau = pva[order[k]]; break

    def report(p, y, D, tau):
        keep = p >= tau
        rec = (keep & (y == 1)).sum() / max(1, (y == 1).sum())
        ret = keep.mean()
        out = {"recall": rec, "retention": ret, "n": len(y)}
        for d in ("arxiv_cs", "biomed_sal"):
            m = np.array([x == d for x in D])
            yk = y[m]; kk = keep[m]
            out[d] = (((kk & (yk == 1)).sum() / max(1, (yk == 1).sum())), kk.mean(), int((yk == 1).sum()))
        return out

    tr_m = report(pte, yte, Dte, tau)

    # ---- rule-based Stage A baseline on TEST (recall + retention of ALL gold gaps) ----
    def rule_slice_recall(pids):
        rec_hit = rec_tot = kept = total = 0
        by = defaultdict(lambda: [0, 0, 0, 0])  # hit, tot, kept, total
        for pid in pids:
            p = corp[pid]; d = dom[pid]; n = len(p["sents"])
            blocks = sal_blocks(p["sents"], p["secs"]) if d == "biomed_sal" else None
            text = "" if d == "biomed_sal" else " ".join(p["sents"])
            if d != "biomed_sal":
                # use real arxiv text
                pass
            sl = [s for r in slice_section_aware(text if d != "biomed_sal" else "", blocks=blocks) for s in r.sentences]
            slnorm = sl
            total += n; by[d][3] += n
            kept += len(sl); by[d][2] += len(sl)
            for idx in gold[pid]:
                gs = p["sents"][idx]
                hit = any(token_containment(gs, ss) >= 0.7 for ss in slnorm)
                rec_tot += 1; rec_hit += hit; by[d][1] += 1; by[d][0] += hit
        out = {"recall": rec_hit / max(1, rec_tot), "retention": kept / max(1, total)}
        for d, v in by.items():
            out[d] = (v[0] / max(1, v[1]), v[2] / max(1, v[3]), v[1])
        return out
    # arxiv rule baseline needs real text
    arxiv_text = {}
    for ln in (ROOT / "data" / "paper_texts.jsonl").read_text(encoding="utf-8").splitlines():
        if ln.strip():
            r = json.loads(ln); arxiv_text[str(r["id"])] = str(r["text"])
    for pid in split["test"]:
        if dom[pid] == "arxiv_cs" and pid in arxiv_text:
            corp[pid] = {**corp[pid], "_text": arxiv_text[pid]}

    def rule_slice_recall2(pids):
        rec_hit = rec_tot = kept = total = 0
        by = defaultdict(lambda: [0, 0, 0, 0])
        for pid in pids:
            p = corp[pid]; d = dom[pid]; n = len(p["sents"])
            if d == "biomed_sal":
                sl = [s for r in slice_section_aware("", blocks=sal_blocks(p["sents"], p["secs"])) for s in r.sentences]
            else:
                sl = [s for r in slice_section_aware(arxiv_text.get(pid, " ".join(p["sents"]))) for s in r.sentences]
            total += n; kept += len(sl); by[d][2] += len(sl); by[d][3] += n
            for idx in gold[pid]:
                hit = any(token_containment(p["sents"][idx], ss) >= 0.7 for ss in sl)
                rec_tot += 1; rec_hit += hit; by[d][1] += 1; by[d][0] += hit
        out = {"recall": rec_hit / max(1, rec_tot), "retention": kept / max(1, total)}
        for d, v in by.items():
            out[d] = (v[0] / max(1, v[1]), v[2] / max(1, v[3]), v[1])
        return out
    rule_m = rule_slice_recall2(split["test"])

    # save model
    import joblib
    joblib.dump({"dv": dv, "clf": clf, "tau": float(tau), "sec_vocab": SEC_VOCAB}, OUT)
    OUT.with_suffix(".meta.json").write_text(json.dumps({
        "target_recall": args.target, "tau": float(tau),
        "test_trained": {"recall": tr_m["recall"], "retention": tr_m["retention"]},
        "test_rule": {"recall": rule_m["recall"], "retention": rule_m["retention"]}}, indent=2), encoding="utf-8")

    def fmt(m):
        return (f"recall={m['recall']*100:.1f}%  retention={m['retention']*100:.1f}%  "
                f"| arxiv R={m['arxiv_cs'][0]*100:.1f}% keep={m['arxiv_cs'][1]*100:.1f}% "
                f"| biomed R={m['biomed_sal'][0]*100:.1f}% keep={m['biomed_sal'][1]*100:.1f}%")

    L = ["# Stage A — trained structural localizer vs rules\n",
         f"Cheap recall-first localizer (position + section + length, no embeddings), tuned on val to "
         f"target recall {args.target:.0%}. Compared to the rule-based `slice_section_aware` on the same "
         "TEST papers (recall/retention over ALL gold gaps). Lower retention = more sentences dropped.\n",
         "| Stage A | recall | retention (kept) | arXiv R / keep | biomed R / keep |\n|---|---|---|---|---|",
         f"| **trained** (tau={tau:.3f}) | {tr_m['recall']*100:.1f}% | {tr_m['retention']*100:.1f}% | "
         f"{tr_m['arxiv_cs'][0]*100:.1f}% / {tr_m['arxiv_cs'][1]*100:.1f}% | "
         f"{tr_m['biomed_sal'][0]*100:.1f}% / {tr_m['biomed_sal'][1]*100:.1f}% |",
         f"| rule slice | {rule_m['recall']*100:.1f}% | {rule_m['retention']*100:.1f}% | "
         f"{rule_m['arxiv_cs'][0]*100:.1f}% / {rule_m['arxiv_cs'][1]*100:.1f}% | "
         f"{rule_m['biomed_sal'][0]*100:.1f}% / {rule_m['biomed_sal'][1]*100:.1f}% |"]
    # learned section weights (interpretability)
    names = dv.get_feature_names_out()
    w = clf.coef_[0]
    sw = sorted([(names[i], w[i]) for i in range(len(names)) if names[i].startswith("sec_")],
                key=lambda x: -x[1])
    L.append("\n## Learned section weights (higher = more gap-bearing)\n")
    L.append(", ".join(f"{n.replace('sec_','')}:{v:+.2f}" for n, v in sw))
    L.append("\n### Notes\n- Trained Stage A learns section + position weights from the corpus; "
             "for arXiv (no section labels) it derives a coarse section from inline headings.\n"
             "- Recall-first: tuned to keep ~target% of gaps while dropping as much as possible.\n"
             "- Labels are LLM-gold; retention/recall over ALL gold gap types.")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("TRAINED:", fmt(tr_m))
    print("RULE   :", fmt(rule_m))
    print("\ntop gap-bearing sections:", ", ".join(f"{n.replace('sec_','')}{v:+.2f}" for n, v in sw[:6]))
    print("wrote", OUT, DOC)


if __name__ == "__main__":
    main()
