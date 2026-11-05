# -*- coding: utf-8 -*-
"""Leakage and data-integrity audit for the datasets behind the paper's results.
Read-only, no API calls. Prints counts; sentences are not printed in full."""
import ast
import json
import os
import random
import re
import sys
import types
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
# Only load_corpus is used. The stub stops gold_label_full from building a real API client.
_stub = types.ModuleType("anthropic")
_stub.Anthropic = lambda **kwargs: None
sys.modules["anthropic"] = _stub
os.environ.setdefault("ANTHROPIC_API_KEY", "unused")
from gold_label_full import load_corpus

SAL = ROOT / "data/external/sal_rct"
FIXTURE = ROOT / "stage_c_package/sal_criteria/sal_test_allmodels.json"
HEAD_TRAIN = ROOT / "data/scibert_prep/train.jsonl"


def norm(s):
    return re.sub(r"\W+", " ", str(s)).strip().lower()


def meaningful(s):
    return len(s.split()) >= 6


def parse_lim(x):
    try:
        return 1 if any(bool(b) for b in ast.literal_eval(x)) else 0
    except Exception:
        return 1 if "True" in str(x) else 0


def section(title):
    print(f"\n===== {title} =====")


def audit_sal():
    section("SAL splits and fixture")
    frames = {n: pd.read_csv(SAL / f"{n}.csv") for n in ("train", "dev", "test")}
    print("columns:", list(frames["test"].columns))
    id_col = next((c for c in frames["test"].columns if c.lower() in ("pmcids", "pmcid", "pmid", "article_id", "id", "paper_id", "doc_id", "article")), None)
    print("paper id column:", id_col)
    for n, f in frames.items():
        f["y"] = f.limitation_annotation.map(parse_lim)
        f["n"] = f.sentences.map(norm)
        print(n, "rows", len(f), "positives", int(f.y.sum()), "papers", f[id_col].nunique() if id_col else "?")
    if id_col:
        ids = {n: set(f[id_col]) for n, f in frames.items()}
        for a, b in combinations(ids, 2):
            print(f"paper overlap {a}/{b}:", len(ids[a] & ids[b]))
    sets = {n: set(x for x in f.n if meaningful(x)) for n, f in frames.items()}
    for a, b in (("train", "test"), ("dev", "test"), ("train", "dev")):
        print(f"sentence overlap {a}/{b} (6+ words):", len(sets[a] & sets[b]))
    pos_train = set(x for x, y in zip(frames["train"].n, frames["train"].y) if y and meaningful(x))
    print("test positives whose exact text is a train positive:",
          sum(1 for x, y in zip(frames["test"].n, frames["test"].y) if y and x in pos_train))

    fx = json.load(open(FIXTURE, encoding="utf-8"))
    print("fixture rows", len(fx), "positives", sum(r["label"] for r in fx), "papers", len({r["pmcid"] for r in fx}))
    fx_norm = Counter(norm(r["sent"]) for r in fx)
    te = frames["test"]
    missing = te[~te.n.isin(fx_norm)]
    print("test.csv rows absent from fixture:", len(missing), "of which positive:", int(missing.y.sum()))
    if id_col:
        print("papers among absent rows:", missing[id_col].nunique(), "| test papers fully absent:",
              len(set(te[id_col]) - set(missing[id_col]).symmetric_difference(set(te[id_col]))) if False else
              sum(1 for p, g in te.groupby(id_col) if not g.n.isin(fx_norm).any()))
    print("fixture rows absent from test.csv:", sum(1 for r in fx if norm(r["sent"]) not in set(te.n)))
    fx_label = {norm(r["sent"]): r["label"] for r in fx}
    print("label disagreements fixture vs test.csv:",
          sum(1 for x, y in zip(te.n, te.y) if x in fx_label and fx_label[x] != y))
    print("fixture section counts:", dict(Counter(r["sec"] for r in fx).most_common(12)))
    print("fixture positives by section:", dict(Counter(r["sec"] for r in fx if r["label"]).most_common(8)))
    print("prediction columns:", list(fx[0]["preds"]))
    return frames, fx


def audit_gold():
    section("Gold corpus")
    gold, domain, source = {}, {}, {}
    for line in (ROOT / "data/gold/gap_gold_final.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            g = json.loads(line)
            gold[(g["paper_id"], g["idx"])] = g["label"]
            domain[g["paper_id"]] = g["domain"]
    splits = json.loads((ROOT / "data/gold/splits.json").read_text(encoding="utf-8"))
    print("papers per split:", {k: len(v) for k, v in splits.items()},
          "| by domain:", {k: dict(Counter(domain[p] for p in v)) for k, v in splits.items()})
    for a, b in combinations(splits, 2):
        print(f"paper overlap {a}/{b}:", len(set(splits[a]) & set(splits[b])))
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in domain}
    sets = {}
    for name, pids in splits.items():
        sets[name] = defaultdict(set)
        for pid in pids:
            for i, s in enumerate(corp[pid]["sents"]):
                if meaningful(norm(s)):
                    sets[name][norm(s)].add(gold.get((pid, i), "none"))
    for a, b in (("train", "test"), ("val", "test"), ("train", "val")):
        shared = set(sets[a]) & set(sets[b])
        lab = [x for x in shared if (sets[a][x] | sets[b][x]) != {"none"}]
        print(f"sentence overlap {a}/{b}:", len(shared), "with a gap label:", len(lab))

    # Near-duplicate papers across splits: a shared-sentence share above 10% is suspicious.
    split_of = {p: k for k, v in splits.items() for p in v}
    psets = {p: {norm(s) for s in corp[p]["sents"] if meaningful(norm(s))} for p in corp}
    pairs = []
    for a, b in combinations(psets, 2):
        if not psets[a] or not psets[b]:
            continue
        inter = len(psets[a] & psets[b])
        if inter >= 5 and inter / min(len(psets[a]), len(psets[b])) >= 0.05:
            pairs.append((round(inter / min(len(psets[a]), len(psets[b])), 3), a, split_of[a], b, split_of[b]))
    pairs.sort(reverse=True)
    print("paper pairs sharing 5%+ of sentences:", len(pairs), "| across splits:",
          sum(p[2] != p[4] for p in pairs))
    for p in pairs[:12]:
        print("  ", p)

    cands = json.loads((ROOT / "data/gold/funnel_v2_candidates_balanced.json").read_text(encoding="utf-8"))
    print("candidate papers outside test:", sum(c["id"] not in set(splits["test"]) for c in cands))
    gaps = [c for c in cands if c["label"] != "none"]
    print("gap_type shown to Stage C matches gold type on gaps:",
          f"{sum(c['gap_type'] == c['label'] for c in gaps)}/{len(gaps)}",
          "| gap_type values on non-gaps:", dict(Counter(c["gap_type"] for c in cands if c["label"] == "none")))
    print("candidate keys:", sorted(cands[0]))
    return corp, splits


def audit_head_train(fx, corp, splits):
    section("pipeline_head training data (used for FWS and Sci-Sentence frozen results)")
    rows = [json.loads(l) for l in HEAD_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    print("rows", len(rows), "keys", sorted(rows[0]))
    src_key = next((k for k in rows[0] if k in ("source", "src", "dataset", "origin")), None)
    if src_key:
        print("sources:", dict(Counter(r[src_key] for r in rows).most_common(12)))
    print("labels:", dict(Counter(str(r.get("label")) for r in rows).most_common(8)))
    train = {norm(r.get("sentence", r.get("text", ""))) for r in rows}

    sx = pd.read_excel(ROOT / "data/external/scisentence_repo/datasets/Based_Version.xlsx")
    te, tr = sx[sx.Classification == "TEST"], sx[sx.Classification == "TRAINING"]
    print("Sci-Sentence columns:", list(sx.columns))
    te_n, tr_n = te.Sentence.map(norm), tr.Sentence.map(norm)
    pos = te.Category.isin(["LIMITATION", "RESEARCH GAP"])
    print("Sci-Sentence test rows", len(te), "positives", int(pos.sum()))
    print("  test sentences in head training:", int(te_n.isin(train).sum()), "positives among them:", int((te_n.isin(train) & pos).sum()))
    print("  test sentences in Sci-Sentence train:", int(te_n.isin(set(tr_n)).sum()))

    df = pd.read_csv(ROOT / "data/external/fws_corpus.csv")
    df["n"] = df.text.map(norm)
    in_train = df.n.isin(train)
    print("FWS rows", len(df), "papers", df.id.nunique(), "rows in head training", int(in_train.sum()),
          "| papers touched", df[in_train].id.nunique())
    # Rebuild the sample used by breadth_funnel.py.
    raw_train = {json.loads(l)["sentence"].strip() for l in HEAD_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()}
    guarded = df[~df.text.str.strip().isin(raw_train)]
    papers = guarded.id.unique().tolist()
    random.seed(0)
    random.shuffle(papers)
    sel, n = [], 0
    for pid in papers:
        sub = guarded[guarded.id == pid]
        sel.append(pid)
        n += len(sub)
        if n >= 1500:
            break
    touched = set(df[in_train].id)
    print("FWS eval sample: papers", len(sel), "sentences", n,
          "| papers that also supplied training sentences:", sum(p in touched for p in sel))
    print("  normalised sample sentences still in head training:",
          int(guarded[guarded.id.isin(sel)].n.isin(train).sum()))

    print("SAL-40 sentences in head training:", sum(norm(r["sent"]) in train for r in fx))
    test_sents = {norm(s) for p in splits["test"] for s in corp[p]["sents"] if meaningful(norm(s))}
    print("gold test sentences in head training:", len(test_sents & train))


def main():
    _, fx = audit_sal()
    corp, splits = audit_gold()
    audit_head_train(fx, corp, splits)
    section("Cost baseline")
    print((ROOT / "stage_c_package/cost_fullpaper_results.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
