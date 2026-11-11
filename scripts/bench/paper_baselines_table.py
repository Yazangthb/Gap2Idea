# -*- coding: utf-8 -*-
"""Table of published-method baselines against the funnel, from saved predictions only (no model is run).

Baselines: the Task 5b runs imported by import_task5b_baselines.py (docs/experiments/task5b_baselines/kept_*.json).
Funnel: the paper's main funnel (zero-shot NLI Stage B, Qwen3.8-27B Stage C, 20 candidates per call) in
high-precision mode, rebuilt from the saved NLI scores and the saved Stage C verdicts.

Before scoring, the script checks that
  - the rebuilt funnel reproduces the candidate and kept counts reported in the paper, and
  - every baseline was scored on exactly this branch's test rows (paper id, sentence index, text hash, label).
Paired differences use the same resampling of papers as the other tables (2000 draws, random.Random(0)).

  python scripts/bench/paper_baselines_table.py
writes manuscript_mdpi/tables/baselines_table.tex and docs/experiments/task5b_baselines/results.json
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nli_variants import ROOT, load_val  # noqa: E402

SRC = ROOT / "docs/experiments/task5b_baselines"
TAB = ROOT / "manuscript_mdpi/tables"
KEEP_SECTIONS = {"DISCUSS", "CONCL"}
GOLD_TAU = 0.354
TYPES = ("lim_c", "fw_c", "gap_c")
EXPECT = {"sal": (639, 263), "fws": (400, 217), "gold": (2522, 440)}  # candidates, kept (paper, Table 3)
LABELS = {
    "kilicoglu_section_proxy": r"Section and cue rules, after Kilicoglu et al.~\cite{kilicoglu2018automatic}",
    "zhang_nb": r"Naive Bayes with selected $n$-grams, after Zhang et al.~\cite{zhang2023fws}",
    "tfidf_lr": "TF-IDF with logistic regression",
    "binary_nb": r"Bernoulli naive Bayes on binary $n$-grams",
    "modernbert_nli": "NLI model alone, one broad hypothesis",
    "modernbert_nli_typed": "NLI model alone, short type hypotheses",
}
ORDER = ["kilicoglu_section_proxy", "zhang_nb", "tfidf_lr", "binary_nb", "modernbert_nli", "modernbert_nli_typed"]
SETS = [("sal", "SAL-40 (human labels)"), ("fws", "FWS (human labels)"), ("arxiv", "arXiv (28 papers, LLM labels)")]


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evc(path, hyp):
    a = np.load(path)[f"sent__{hyp}"]
    return a[:, 0] / (a[:, 0] + a[:, 2] + 1e-9)


def jl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def threshold(rows, scores, passed):
    """Highest tau whose recall on development data, counting Stage A losses, is at least 90% (nli_sal_fws.py)."""
    y = np.array([r["label"] == 1 for r in rows])
    ranked = np.sort(scores[passed & y])[::-1]
    return float(ranked[int(np.ceil(0.90 * y.sum())) - 1])


def bench_funnel(name):
    """SAL-40 or FWS: test rows and the sentences the main funnel kept in high-precision mode."""
    hyp = {"sal": "lim_c", "fws": "fw_c"}[name]
    split = {part: jl(ROOT / f"data/splits/{name}_{part}.jsonl") for part in ("dev", "test")}
    gate = {part: np.array([name == "fws" or r["section"] in KEEP_SECTIONS for r in rows]) for part, rows in split.items()}
    tau = threshold(split["dev"], evc(ROOT / f"data/gold/nli_{name}_dev_scores.npz", hyp), gate["dev"])
    score = evc(ROOT / f"data/gold/nli_{name}_test_scores.npz", hyp)
    ids = [i for i in range(len(split["test"])) if gate["test"][i] and score[i] >= tau]
    verdict = {r["i"]: r["keep"] for r in jl(ROOT / f"docs/experiments/stage_c_sal40_modes_nli_{name}_test_high_precision.jsonl")}
    rows = [dict(pid=r["paper_id"], idx=r["idx"], text=r["sentence"], gap=r["label"] == 1) for r in split["test"]]
    kept = {(rows[ids[i]]["pid"], rows[ids[i]]["idx"]) for i, k in verdict.items() if k}
    return rows, kept, len(ids)


def gold_funnel():
    """Gold test: rows and kept sentences; the arXiv subset is taken by the caller."""
    _, test = load_val("test")
    npz = ROOT / "data/gold/nli_test_scores.npz"
    typed = np.stack([evc(npz, h) for h in TYPES], 1)
    pool = [i for i, r in enumerate(test) if not r["junk"] and typed[i].max() >= GOLD_TAU]
    verdict = {r["i"]: r["keep"] for r in jl(ROOT / "docs/experiments/stage_c_rubric_test_strict.jsonl")}
    rows = [dict(pid=r["pid"], idx=r["idx"], text=r["sentence"], gap=r["label"] != "none", dom=r["dom"]) for r in test]
    kept = {(rows[pool[i]]["pid"], rows[pool[i]]["idx"]) for i, k in verdict.items() if k}
    return rows, kept, len(pool)


def prf(tp, fp, gaps):
    p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / gaps if gaps else 0.0)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(rows, kept):
    tp = sum(r["gap"] and (r["pid"], r["idx"]) in kept for r in rows)
    fp = sum((not r["gap"]) and (r["pid"], r["idx"]) in kept for r in rows)
    return prf(tp, fp, sum(r["gap"] for r in rows))


def paired(rows, a, b, n=2000):
    """F1 of a minus F1 of b with a 95% interval from resampling papers (same procedure as stage_c_nli_qwen.paired)."""
    by_paper = {}
    for r in rows:
        by_paper.setdefault(r["pid"], []).append(r)

    def counts(p, kept):
        tp = fp = gaps = 0
        for r in by_paper[p]:
            k = (r["pid"], r["idx"]) in kept
            gaps += r["gap"]
            tp += k and r["gap"]
            fp += k and not r["gap"]
        return tp, fp, gaps

    table = {p: (counts(p, a), counts(p, b)) for p in by_paper}
    papers, rng, diffs = list(by_paper), random.Random(0), []
    for _ in range(n):
        pick = rng.choices(papers, k=len(papers))
        diffs.append(prf(*[sum(table[p][0][j] for p in pick) for j in range(3)])[2]
                     - prf(*[sum(table[p][1][j] for p in pick) for j in range(3)])[2])
    diffs.sort()
    point = prf(*[sum(table[p][0][j] for p in papers) for j in range(3)])[2] - \
        prf(*[sum(table[p][1][j] for p in papers) for j in range(3)])[2]
    return point, diffs[int(0.025 * n)], diffs[int(0.975 * n)]


def signed(x, digits=3):
    return f"{x:+.{digits}f}".replace("-", "$-$").replace("+", "$+$")


def main():
    data = {}
    for name in ("sal", "fws"):
        rows, kept, n_cand = bench_funnel(name)
        if (n_cand, len(kept)) != EXPECT[name]:
            sys.exit(f"{name}: rebuilt funnel has {n_cand} candidates and keeps {len(kept)}, expected {EXPECT[name]}")
        data[name] = (rows, kept)
    rows, kept, n_cand = gold_funnel()
    if (n_cand, len(kept)) != EXPECT["gold"]:
        sys.exit(f"gold: rebuilt funnel has {n_cand} candidates and keeps {len(kept)}, expected {EXPECT['gold']}")
    arxiv = [r for r in rows if r["dom"] == "arxiv_cs"]
    data["arxiv"] = (arxiv, {k for k in kept if k in {(r["pid"], r["idx"]) for r in arxiv}})

    out, lines = {}, []
    for name, label in SETS:
        rows, funnel = data[name]
        src = json.loads((SRC / f"kept_{name}.json").read_text(encoding="utf-8"))
        ours = [[r["pid"], r["idx"], sha(r["text"]), int(r["gap"])] for r in rows]
        if ours != src["rows"]:
            sys.exit(f"{name}: the baselines were scored on different test rows than this branch's test set")
        fp, fr, ff = score(rows, funnel)
        block = {"task5b_commit": src["task5b_commit"], "rows": len(rows), "papers": len({r["pid"] for r in rows}),
                 "positives": sum(r["gap"] for r in rows),
                 "funnel_high_precision": {"kept": len(funnel), "PRF": [round(fp, 4), round(fr, 4), round(ff, 4)]}, "methods": {}}
        shown = [m for m in ORDER if m in src["methods"]]
        first = f"\\multirow{{{len(shown) + 1}}}{{*}}{{\\parbox{{2.4cm}}{{\\raggedright {label}}}}}"
        for m, info in src["methods"].items():
            kept_m = {(src["rows"][i][0], src["rows"][i][1]) for i in info["positive_rows"]}
            p, r, f1 = score(rows, kept_m)
            d, lo, hi = paired(rows, funnel, kept_m)
            block["methods"][m] = {"kept": len(kept_m), "PRF": [round(p, 4), round(r, 4), round(f1, 4)],
                                   "funnel_minus_method": {"f1_diff": round(d, 3), "ci95": [round(lo, 3), round(hi, 3)]},
                                   "threshold": info["threshold"], "source_sha256": info["source_sha256"]}
        for k, m in enumerate(shown):
            b = block["methods"][m]
            d = b["funnel_minus_method"]
            sig = d["ci95"][0] > 0 or d["ci95"][1] < 0
            diff = signed(d["f1_diff"])
            diff = (f"\\textbf{{{diff}}}" if sig else diff) + f" ({signed(d['ci95'][0])}, {signed(d['ci95'][1])})"
            lines.append(f"{first if k == 0 else ''} & {LABELS[m]} & {b['kept']} & {100 * b['PRF'][0]:.1f} & "
                         f"{100 * b['PRF'][1]:.1f} & {b['PRF'][2]:.3f} & {diff} \\\\")
        lines.append(f" & \\textbf{{Funnel, high precision}} & {len(funnel)} & {100 * fp:.1f} & {100 * fr:.1f} & "
                     f"\\textbf{{{ff:.3f}}} & -- \\\\")
        lines.append(r"\midrule")
        out[name] = block
        print(name, json.dumps(block["funnel_high_precision"]), {m: (b["PRF"], b["funnel_minus_method"]) for m, b in block["methods"].items()})
    head = (r"\textbf{Test set} & \textbf{Method} & \textbf{Kept} & \textbf{Precision} & \textbf{Recall} & \textbf{$F_1$} & "
            r"\textbf{Funnel $-$ method, $F_1$}" + " \\\\\n")
    spec = r"@{}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}Xrrrrr@{}"
    body = "\n".join(lines[:-1]) + "\n"
    (TAB / "baselines_table.tex").write_text(
        r"\begin{tabularx}{\fulllength}{" + spec + "}\n" + r"\toprule" + "\n" + head + r"\midrule" + "\n" + body
        + r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n", encoding="utf-8")
    (SRC / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
