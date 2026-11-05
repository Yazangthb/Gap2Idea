# -*- coding: utf-8 -*-
"""NLI recall/precision variants on the gold VALIDATION papers. The test papers are never loaded.

  export              write label-free sentences for nli_score_remote.py, split val into halves A and B
  explore             score variants on half A only
  confirm NAME...     score named variants on half B, using thresholds chosen on half A

Scoring is per sentence against the LLM gold, any gap type, with all sentences in the denominator.
"""
import json
import os
import random
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
_stub = types.ModuleType("anthropic")  # only load_corpus is used; no API calls
_stub.Anthropic = lambda **kwargs: None
sys.modules["anthropic"] = _stub
os.environ.setdefault("ANTHROPIC_API_KEY", "unused")
from gold_label_full import load_corpus
from gap2idea.pipeline.gap_funnel import EmbeddingGapHead, _is_reference, _looks_like_sentence

SCORES = ROOT / "data/gold/nli_val_scores.npz"
OUT = ROOT / "docs/experiments/nli_variants.json"
DROP_SEC = {"TITLE", "TABLE", "FIG", "ABBR", "COMP_INT", "AUTH_CONT", "ACK_FUND", "SUPPL"}
TYPES = ("lim", "fw", "gap")
NEG = ("neg_result", "neg_method", "neg_background", "neg_claim")


def load_val(part="val"):
    gold, domain = {}, {}
    for raw in (ROOT / "data/gold/gap_gold_final.jsonl").read_text(encoding="utf-8").splitlines():
        if raw.strip():
            g = json.loads(raw)
            gold[(g["paper_id"], g["idx"])] = g["label"]
            domain[g["paper_id"]] = g["domain"]
    val = json.loads((ROOT / "data/gold/splits.json").read_text(encoding="utf-8"))[part]
    corp = {p["paper_id"]: p for p in load_corpus() if p["paper_id"] in set(val)}
    rows = []
    for pid in val:
        p, secs = corp[pid], corp[pid].get("secs")
        s = p["sents"]
        for i, sent in enumerate(s):
            sec = secs[i] if secs else None
            rows.append(dict(pid=pid, idx=i, sentence=sent, context=" ".join(s[max(0, i - 1):i + 2]), dom=domain[pid],
                             label=gold.get((pid, i), "none"),
                             junk=sec in DROP_SEC or _is_reference(sent) or not _looks_like_sentence(sent)))
    return val, rows


def halves(val, domain_of):
    """Paper-level split, stratified by domain, fixed seed."""
    rng, a = random.Random(2026), set()
    for dom in ("arxiv_cs", "biomed_sal"):
        papers = sorted(p for p in val if domain_of[p] == dom)
        rng.shuffle(papers)
        a |= set(papers[:len(papers) // 2])
    return a


def prf(pred, y):
    tp = int((pred & y).sum())
    p = tp / int(pred.sum()) if pred.sum() else 0.0
    r = tp / int(y.sum())
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def rank01(x):
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def build_variants(npz, head):
    def entail(prem, h):
        return npz[f"{prem}__{h}"][:, 0]

    def e_vs_c(prem, h):  # the original zero-shot score: entailment against contradiction only
        a = npz[f"{prem}__{h}"]
        return a[:, 0] / (a[:, 0] + a[:, 2] + 1e-9)

    def typed(prem, tag, f=entail):
        return np.stack([f(prem, f"{t}_{tag}") for t in TYPES], 1)

    def negs(prem):
        return np.stack([entail(prem, h) for h in NEG], 1).max(1)

    v = {"orig_evc": e_vs_c("sent", "orig"), "orig_entail": entail("sent", "orig"),
         "orig_ctx_evc": e_vs_c("ctx", "orig"), "orig_ctx_entail": entail("ctx", "orig")}
    for tag in "abc":
        t = typed("sent", tag)
        v[f"typed_{tag}_max"] = t.max(1)
        v[f"typed_{tag}_mean"] = t.mean(1)
        v[f"typed_{tag}_max_evc"] = typed("sent", tag, e_vs_c).max(1)
        v[f"typed_{tag}_ctx_max"] = typed("ctx", tag).max(1)
        v[f"typed_{tag}_minus_neg"] = t.max(1) - negs("sent")
    all_typed = np.concatenate([typed("sent", t) for t in "abc"], 1)
    v["all_typed_mean"] = all_typed.mean(1)
    v["all_typed_max"] = all_typed.max(1)
    v["orig_minus_neg"] = entail("sent", "orig") - negs("sent")
    v["sent_ctx_avg"] = (typed("sent", "b").max(1) + typed("ctx", "b").max(1)) / 2
    # Combined with the frozen bge-small head, trained on gold train only.
    v["head_only"] = head
    for name in ("orig_evc", "typed_b_max", "all_typed_mean"):
        v[f"{name}+head_rank"] = (rank01(v[name]) + rank01(head)) / 2
    return v


def head_scores(rows):
    import joblib
    bundle = joblib.load(ROOT / "data/gold/gap_head_gold.joblib")
    enc = EmbeddingGapHead.load_encoder(bundle["encoder_name"])
    clf = bundle["clf"]
    if not hasattr(clf, "multi_class"):  # pickled by a newer scikit-learn; lbfgs with 4 classes is multinomial
        clf.multi_class = "multinomial"
    none_j = list(clf.classes_).index("none")
    x = enc.encode([r["sentence"] for r in rows], normalize_embeddings=True, batch_size=128, show_progress_bar=False)
    return 1.0 - clf.predict_proba(x)[:, none_j]


def operating_points(score, y, mask):
    """Thresholds picked on `mask` only: best F1, and the highest threshold reaching each recall target."""
    s, t = score[mask], y[mask]
    grid = np.unique(np.quantile(s, np.linspace(0.02, 0.999, 300)))
    pts = {}
    best = max(grid, key=lambda g: prf(s >= g, t)[2])
    pts["maxF1"] = float(best)
    for target in (0.70, 0.80, 0.90):
        ok = [g for g in grid if prf(s >= g, t)[1] >= target]
        if ok:
            pts[f"R>={int(target * 100)}"] = float(max(ok))
    return pts


def setup():
    val, rows = load_val()
    domain_of = {r["pid"]: r["dom"] for r in rows}
    a = halves(val, domain_of)
    y = np.array([r["label"] != "none" for r in rows])
    in_a = np.array([r["pid"] in a for r in rows])
    junk = np.array([r["junk"] for r in rows])
    return rows, y, in_a, junk, a


def explore():
    rows, y, in_a, junk, a = setup()
    print(f"half A: {len(a)} papers, {int(in_a.sum())} sentences, {int(y[in_a].sum())} gaps, prevalence {y[in_a].mean():.3f}")
    variants = build_variants(np.load(SCORES), head_scores(rows))
    keep = ~junk
    out = {}
    print(f"{'variant':26}{'maxF1 P/R/F1':>21}{'R>=70 P':>9}{'R>=80 P':>9}{'R>=90 P':>9}{'AUPRC':>7}")
    for name, sc in variants.items():
        pts = operating_points(sc[keep], y[keep], in_a[keep])
        res = {k: prf((sc >= t) & keep & in_a, y & in_a) for k, t in pts.items()}
        # Average precision among non-junk half-A sentences, against all half-A gaps.
        order = np.argsort(-sc[keep & in_a])
        hits = y[keep & in_a][order]
        ap = float((np.cumsum(hits) / np.arange(1, len(hits) + 1))[hits].sum() / y[in_a].sum())
        out[name] = {"thresholds": pts, "A": {k: [round(x, 3) for x in v] for k, v in res.items()}, "AP": round(ap, 3)}
    for name, r in sorted(out.items(), key=lambda kv: -kv[1]["A"]["maxF1"][2]):
        m = r["A"]["maxF1"]
        cell = lambda k: f"{r['A'][k][0]:.3f}" if k in r["A"] else "-"
        print(f"{name:26}{m[0]:7.3f}{m[1]:7.3f}{m[2]:7.3f}{cell('R>=70'):>9}{cell('R>=80'):>9}{cell('R>=90'):>9}{r['AP']:7.3f}")
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", OUT)


def confirm(names):
    rows, y, in_a, junk, _ = setup()
    variants = build_variants(np.load(SCORES), head_scores(rows))
    keep, in_b = ~junk, ~in_a
    saved = json.loads(OUT.read_text(encoding="utf-8"))
    print(f"half B: {int(in_b.sum())} sentences, {int(y[in_b].sum())} gaps")
    for n in names:
        for k, t in saved[n]["thresholds"].items():
            p, r, f = prf((variants[n] >= t) & keep & in_b, y & in_b)
            flagged = (variants[n] >= t) & keep & in_b
            by_type = {}
            for label in ("limitation", "future_work", "research_gap"):
                of_type = np.array([x["label"] == label for x in rows]) & in_b
                by_type[label] = round(float((flagged & of_type).sum() / of_type.sum()), 3)
            by_dom = {}
            for dom in ("arxiv_cs", "biomed_sal"):
                d = np.array([x["dom"] == dom for x in rows])
                by_dom[dom[:5]] = [round(v, 3) for v in prf(flagged & d, y & in_b & d)]
            print(f"{n:22}{k:7} tau {t:6.3f} kept {int(flagged.sum()):5} P {p:.3f} R {r:.3f} F1 {f:.3f} | "
                  f"lim {by_type['limitation']:.3f} fw {by_type['future_work']:.3f} gap {by_type['research_gap']:.3f} | {by_dom}")


def export():
    val, rows = load_val()
    target = Path(sys.argv[2])
    target.write_text(json.dumps([{"sentence": r["sentence"], "context": r["context"]} for r in rows], ensure_ascii=False), encoding="utf-8")
    print("papers", len(val), "sentences", len(rows), "gaps", sum(r["label"] != "none" for r in rows), "->", target)


if __name__ == "__main__":
    {"export": export, "explore": explore, "confirm": lambda: confirm(sys.argv[2:])}[sys.argv[1]]()
