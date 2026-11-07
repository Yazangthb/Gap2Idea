# -*- coding: utf-8 -*-
"""Stage B with the fine-tuned NLI model (finetune_nli.py) against the zero-shot model, same rules.

  python eval_nli_ft.py FT_DIR      FT_DIR holds nli_ft_*_scores.npz and train_log.json from the GPU box

Gold: max over the three short hypotheses, Stage A junk filter, tau = highest with recall >= 90% on gold
validation (all of it), scored on test. The frozen funnel's tau 0.354 (chosen on half of validation) is also
shown for the zero-shot model. SAL and FWS: one hypothesis each, their own Stage A and dev-set tau, as in
nli_sal_fws.py. Writes docs/experiments/nli_finetune.json.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from nli_variants import load_val  # noqa: E402
from nli_sal_fws import load_all, stage_a, threshold  # noqa: E402

H = ("lim_c", "fw_c", "gap_c")
T = ("limitation", "future_work", "research_gap")
FROZEN_TAU = 0.354


def evc(path, h):
    a = np.load(path)[f"sent__{h}"]
    return a[:, 0] / (a[:, 0] + a[:, 2] + 1e-9)


def prf(keep, pos):
    tp = int((keep & pos).sum())
    p = tp / keep.sum() if keep.sum() else 0.0
    r = tp / pos.sum()
    return {"candidates": int(keep.sum()), "P": round(p, 4), "R": round(r, 4), "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def gold(files):
    out = {}
    data = {}
    for part in ("val", "test"):
        _, rows = load_val(part)
        data[part] = (np.array([r["label"] for r in rows]), np.array([not r["junk"] for r in rows]),
                      np.array([r["dom"] for r in rows]))
    for model, f in files.items():
        s = {part: np.stack([evc(f(part), h) for h in H], 1).max(1) for part in ("val", "test")}
        y, ok, _ = data["val"]
        ranked = np.sort(s["val"][ok & (y != "none")])[::-1]
        tau = float(ranked[int(math.ceil(0.9 * (y != "none").sum())) - 1])
        taus = {"val_rule": tau} | ({"frozen": FROZEN_TAU} if model == "zero_shot" else {})
        for name, t in taus.items():
            y, ok, dom = data["test"]
            keep = ok & (s["test"] >= t)
            res = {"tau": round(t, 4), "test": prf(keep, y != "none"),
                   "test_recall_by_type": {k: round(float((keep & (y == k)).sum() / (y == k).sum()), 4) for k in T},
                   "test_by_domain": {d: prf(keep & (dom == d), (y != "none") & (dom == d)) for d in ("arxiv_cs", "biomed_sal")}}
            vy, vok, _ = data["val"]
            res["val"] = prf(vok & (s["val"] >= t), vy != "none")
            out[f"{model}@{name}"] = res
    return out


def benchmarks(ft_dir):
    data = load_all()
    out = {}
    for bench, hyp in (("sal", "lim_c"), ("fws", "fw_c")):
        for model, f in (("zero_shot", lambda n: ROOT / f"data/gold/nli_{n}_scores.npz"),
                         ("fine_tuned", lambda n: ft_dir / f"nli_ft_{n}_scores.npz")):
            dev, test = data[f"{bench}_dev"], data[f"{bench}_test"]
            tau = threshold(dev, evc(f(f"{bench}_dev"), hyp), f"{bench}_dev")
            res = {"tau": round(tau, 4)}
            for name, rows in ((f"{bench}_dev", dev), (f"{bench}_test", test)):
                sc = evc(f(name), hyp)
                keep = np.array([stage_a(name, r) for r in rows]) & (sc >= tau)
                res[name.split("_")[1]] = prf(keep, np.array([r["label"] == 1 for r in rows]))
            out[f"{bench}:{model}"] = res
    return out


def main():
    ft_dir = Path(sys.argv[1])
    files = {"zero_shot": lambda part: ROOT / f"data/gold/nli_{part}_scores.npz",
             "fine_tuned": lambda part: ft_dir / f"nli_ft_gold_{part}_scores.npz"}
    result = {"train_log": json.loads((ft_dir / "train_log.json").read_text(encoding="utf-8")),
              "gold": gold(files), "benchmarks": benchmarks(ft_dir)}
    (ROOT / "docs/experiments/nli_finetune.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
