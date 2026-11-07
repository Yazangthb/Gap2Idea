# -*- coding: utf-8 -*-
"""Full funnel with the fine-tuned Stage B NLI model, compared with the whole-paper baseline (task 5a).

Everything except the Stage B model is the published funnel: Stage A per dataset; tau = highest with recall >= 90%
on the dataset's own development data (gold validation, SAL dev, FWS dev), now on the fine-tuned scores
(data/gold/nli_ft/); Stage C = Qwen3.8-27B, the frozen rubrics (narrowed to limitations on SAL, future work on FWS),
20 candidates per call with +-1 sentence context, per-item answers, three modes. Qwen token usage is logged.

  python funnel_nli_ft.py smoke    first 2 batches of each run
  python funnel_nli_ft.py run      every run (resumable), then score
  python funnel_nli_ft.py score    scores and comparisons from the saved verdicts

Comparisons: the whole-paper baseline (task 5a setup A; SAL and gold rubrics, FWS validate_fws) from its saved
verdicts on origin/task-5a-qwen-baselines, and the published zero-shot funnel in the same mode.
"""
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from pathlib import Path

from nli_variants import ROOT, load_val
from nli_sal_fws import future_scope, load_all, pool_of, stage_a, threshold
from stage_c_nli_qwen import TYPES, candidates, paired
from stage_c_prompt_vs_model import BATCH_SCHEMA, parse_json_response, qwen_client
from stage_c_rubric_val import BATCH, EXTRA, RUBRICS
from stage_c_sal40_modes import limitation_scope

FT = ROOT / "data/gold/nli_ft"
# Overrides for the LLM-size runs (defaults reproduce the Qwen3.8-27B, fine-tuned Stage B run):
#   QWEN_MODEL     served model name;  STAGE_B  "ft" (fine-tuned NLI) or "zero_shot" (the published funnel)
#   FUNNEL_OUTDIR  where verdicts go;  BASELINE_DIR  local whole-paper verdicts instead of the task 5a branch
MODEL = os.environ.get("QWEN_MODEL", "qwen3.8-27b")
STAGE_B = os.environ.get("STAGE_B", "ft")
OUTDIR = Path(os.environ.get("FUNNEL_OUTDIR", ROOT / "docs/experiments/funnel_nli_ft"))
BASELINE_DIR = os.environ.get("BASELINE_DIR")
ZS_GOLD_TAU = 0.354  # the published funnel's frozen gold threshold
MODES = {"high_recall": "keep_biased", "balanced": "neutral", "high_precision": "strict"}
BASELINE_REF = "origin/task-5a-qwen-baselines"


def evc(path, h):
    a = np.load(path)[f"sent__{h}"]
    return a[:, 0] / (a[:, 0] + a[:, 2] + 1e-9)


# ---------------------------------------------------------------- datasets
def gold_set():
    """Gold test rows and candidates; tau from the fine-tuned validation scores."""
    if STAGE_B == "zero_shot":
        tau, test_file = ZS_GOLD_TAU, ROOT / "data/gold/nli_test_scores.npz"
    else:
        _, val = load_val("val")
        vs = np.stack([evc(FT / "nli_ft_gold_val_scores.npz", h) for h in TYPES], 1).max(1)
        y = np.array([r["label"] != "none" for r in val])
        ok = np.array([not r["junk"] for r in val])
        tau = float(np.sort(vs[ok & y])[::-1][int(math.ceil(0.9 * y.sum())) - 1])
        test_file = FT / "nli_ft_gold_test_scores.npz"
    _, rows = load_val("test")
    typed = np.stack([evc(test_file, h) for h in TYPES], 1)
    pool = candidates(rows, typed, tau)
    keyed = [dict(pid=r["pid"], idx=r["idx"], label=r["label"], dom=r["dom"]) for r in rows]
    return {"tau": tau, "rows": keyed, "pool": pool, "keys": [(c["id"], c["idx"]) for c in pool],
            "prompt": lambda mode: RUBRICS[mode]}


def bench_set(bench):
    """SAL-40 or FWS test rows (keyed as in data/splits, which follow the same order) and candidates."""
    data = load_all()
    hyp, gap_type, scope = {"sal": ("lim_c", "limitation", limitation_scope),
                            "fws": ("fw_c", "future_work", future_scope)}[bench]
    score_file = ((lambda n: ROOT / f"data/gold/nli_{n}_scores.npz") if STAGE_B == "zero_shot"
                  else (lambda n: FT / f"nli_ft_{n}_scores.npz"))
    tau = threshold(data[f"{bench}_dev"], evc(score_file(f"{bench}_dev"), hyp), f"{bench}_dev")
    rows = data[f"{bench}_test"]
    split = [json.loads(x) for x in (ROOT / f"data/splits/{bench}_test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(split) == len(rows) and all(a["sent"] == b["sentence"] for a, b in zip(rows, split))
    s = evc(score_file(f"{bench}_test"), hyp)
    ids = [i for i, r in enumerate(rows) if stage_a(f"{bench}_test", r) and s[i] >= tau]
    keyed = [dict(pid=b["paper_id"], idx=b["idx"], label=gap_type if b["label"] == 1 else "none", dom=bench) for b in split]
    return {"tau": tau, "rows": keyed, "pool": pool_of(rows, ids, gap_type),
            "keys": [(split[i]["paper_id"], split[i]["idx"]) for i in ids],
            "prompt": lambda mode: scope(RUBRICS[mode])}


SETS = {"gold": gold_set, "sal": lambda: bench_set("sal"), "fws": lambda: bench_set("fws")}


# ---------------------------------------------------------------- Stage C
def judge(client, pool, start, prompt, only=None):
    idx = list(only) if only is not None else list(range(start, min(start + BATCH, len(pool))))
    items = [{"i": i, "type": pool[i]["gap_type"], "context": pool[i]["paragraph_text"][:600],
              "sentence": pool[i]["gap_sentence"][:400]} for i in idx]
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=256 + 400 * len(items),
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
        response_format={"type": "json_schema", "json_schema": {"name": "gap_verdicts", "schema": BATCH_SCHEMA, "strict": True}},
        extra_body=EXTRA)
    got = {int(r["id"]): bool(r["keep"]) for r in parse_json_response(resp.choices[0].message.content).get("results", [])
           if int(r["id"]) in idx}
    return got, resp.usage.prompt_tokens, resp.usage.completion_tokens


def path(name, mode):
    return OUTDIR / f"{name}_{mode}.jsonl"


def jl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def run_set(name, s, client, limit=None):
    for mode, rubric in MODES.items():
        p = path(name, mode)
        done = {r["start"] for r in jl(p)}
        starts = list(range(0, len(s["pool"]), BATCH))[:limit]
        prompt = s["prompt"](rubric)
        with p.open("a", encoding="utf-8") as fh:
            for k, start in enumerate(starts, 1):
                if start in done:
                    continue
                n = min(BATCH, len(s["pool"]) - start)
                rec, t0, best = None, time.time(), None
                for _attempt in range(2):
                    try:
                        got, tin, tout = judge(client, s["pool"], start, prompt)
                        if len(got) == n:
                            rec = {"start": start, "keep": [got[i] for i in range(start, start + n)],
                                   "prompt_tokens": tin, "completion_tokens": tout}
                            break
                        best = (got, tin, tout)
                        print(f"{name} {mode} batch {start}: {len(got)}/{n} verdicts", flush=True)
                    except Exception as e:  # reported; never counted as a verdict
                        print(f"{name} {mode} batch {start}: {type(e).__name__}: {e}", flush=True)
                if rec is None and best is not None:
                    # Some items lack a verdict (seen with Qwen3.5-4B): ask for those items alone, as LLMGapFilter
                    # falls back for missing verdicts; anything still missing is dropped and recorded.
                    got, tin, tout = best
                    missing = [i for i in range(start, start + n) if i not in got]
                    try:
                        more, t2, o2 = judge(client, s["pool"], start, prompt, only=missing)
                        got = {**got, **more}
                        tin, tout = tin + t2, tout + o2
                    except Exception as e:  # reported; the items stay missing
                        print(f"{name} {mode} batch {start} fallback: {type(e).__name__}: {e}", flush=True)
                    still = [i for i in range(start, start + n) if i not in got]
                    rec = {"start": start, "keep": [got.get(i, False) for i in range(start, start + n)],
                           "prompt_tokens": tin, "completion_tokens": tout, "fallback": missing, "dropped_missing": still}
                if rec is None:
                    sys.exit(f"{name} {mode} batch {start} failed twice. Rerun to resume.")
                rec["seconds"] = round(time.time() - t0, 2)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"{name} {mode} {k}/{len(starts)} kept {sum(rec['keep'])}/{n} [{rec['seconds']}s]", flush=True)


# ---------------------------------------------------------------- scoring
def prf(rows, kept):
    tp = sum((r["pid"], r["idx"]) in kept and r["label"] != "none" for r in rows)
    fp = sum((r["pid"], r["idx"]) in kept and r["label"] == "none" for r in rows)
    g = sum(r["label"] != "none" for r in rows)
    p, q = (tp / (tp + fp) if tp + fp else 0.0), tp / g
    return [round(p, 4), round(q, 4), round(2 * p * q / (p + q), 4) if p + q else 0.0]


def kept_of(name, s, mode):
    recs = jl(path(name, mode))
    flags = {r["start"] + j: k for r in recs for j, k in enumerate(r["keep"])}
    if len(flags) != len(s["pool"]):
        sys.exit(f"{name} {mode}: {len(flags)}/{len(s['pool'])} verdicts; run first")
    return {s["keys"][i] for i, k in flags.items() if k}, recs


def baseline(name, mode):
    prompt = MODES[mode]  # task 5a names prompts keep_biased / neutral / strict
    file = {"gold": f"gold_test_A_{prompt}", "sal": f"sal_test_A_{prompt}", "fws": "fws_test_A_validate_fws"}[name]
    if BASELINE_DIR:
        txt = (Path(BASELINE_DIR) / f"{file.replace('_test_', '_test_', 1)}.jsonl").read_text(encoding="utf-8")
    else:
        txt = subprocess.run(["git", "show", f"{BASELINE_REF}:docs/experiments/baseline_qwen/{file}.jsonl"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
    recs = jl_text(txt)
    return {tuple(k) for r in recs for k in r["kept"]}, recs


def jl_text(t):
    return [json.loads(x) for x in t.splitlines() if x.strip()]


def published(name, s_rows, mode):
    """The published zero-shot funnel's kept sentences, same mode."""
    if name == "gold":
        _, rows = load_val("test")
        npz = np.load(ROOT / "data/gold/nli_test_scores.npz")
        typed = np.stack([npz[f"sent__{h}"][:, 0] / (npz[f"sent__{h}"][:, 0] + npz[f"sent__{h}"][:, 2] + 1e-9)
                          for h in TYPES], 1)
        pool = candidates(rows, typed, 0.354)
        v = {r["i"]: r["keep"] for r in jl(ROOT / f"docs/experiments/stage_c_rubric_test_{MODES[mode]}.jsonl")}
        return {(pool[i]["id"], pool[i]["idx"]) for i, k in v.items() if k}
    data = load_all()
    bench_rows = data[f"{name}_test"]
    hyp = {"sal": "lim_c", "fws": "fw_c"}[name]
    tau = threshold(data[f"{name}_dev"], evc(ROOT / f"data/gold/nli_{name}_dev_scores.npz", hyp), f"{name}_dev")
    sc = evc(ROOT / f"data/gold/nli_{name}_test_scores.npz", hyp)
    ids = [i for i, r in enumerate(bench_rows) if stage_a(f"{name}_test", r) and sc[i] >= tau]
    v = {r["i"]: r["keep"] for r in jl(ROOT / f"docs/experiments/stage_c_sal40_modes_nli_{name}_test_{mode}.jsonl")}
    return {(s_rows[ids[i]]["pid"], s_rows[ids[i]]["idx"]) for i, k in v.items() if k}


def score():
    out = {}
    for name, make in SETS.items():
        s = make()
        rows = s["rows"]
        sub = {"all": rows} if name != "gold" else {"all": rows, "arxiv": [r for r in rows if r["dom"] == "arxiv_cs"]}
        block = {"tau": round(s["tau"], 4), "candidates": len(s["pool"])}
        for mode in MODES:
            kept, recs = kept_of(name, s, mode)
            base, brecs = baseline(name, mode)
            pub = published(name, rows, mode)
            res = {"input_tokens": sum(r["prompt_tokens"] for r in recs), "output_tokens": sum(r["completion_tokens"] for r in recs),
                   "seconds": round(sum(r["seconds"] for r in recs), 1), "calls": len(recs)}
            for label, R in sub.items():
                keys = {(r["pid"], r["idx"]) for r in R}
                res[label] = {"funnel_ft": prf(R, kept), "published_funnel": prf(R, pub),
                              "ft_minus_published": paired(R, kept & keys, pub & keys)}
                if name != "gold" or label == "arxiv":
                    res[label]["whole_paper"] = prf(R, base)
                    res[label]["ft_minus_whole_paper"] = paired(R, kept & keys, base & keys)
                    res[label]["whole_paper_cost"] = {"input_tokens": sum(r["prompt_tokens"] for r in brecs),
                                                      "output_tokens": sum(r["completion_tokens"] for r in brecs),
                                                      "seconds": round(sum(r["seconds"] for r in brecs), 1)}
            if name == "gold":  # arXiv-only Stage C cost, by each batch's share of arXiv candidates
                dom = {(r["pid"], r["idx"]): r["dom"] for r in rows}
                arx = {i for i, k in enumerate(s["keys"]) if dom[k] == "arxiv_cs"}
                share = lambda r: sum(i in arx for i in range(r["start"], r["start"] + len(r["keep"]))) / len(r["keep"])
                res["arxiv_cost"] = {"input_tokens": round(sum(r["prompt_tokens"] * share(r) for r in recs)),
                                     "output_tokens": round(sum(r["completion_tokens"] * share(r) for r in recs)),
                                     "seconds": round(sum(r["seconds"] * share(r) for r in recs), 1)}
            block[mode] = res
        out[name] = block
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("smoke", "run", "score"):
        sys.exit(__doc__)
    if cmd != "score":
        OUTDIR.mkdir(parents=True, exist_ok=True)
        client = qwen_client()
        for name, make in SETS.items():
            s = make()
            print(f"{name}: tau {s['tau']:.4f}, {len(s['pool'])} candidates", flush=True)
            run_set(name, s, client, 2 if cmd == "smoke" else None)
    if cmd != "smoke":
        score()


if __name__ == "__main__":
    main()
