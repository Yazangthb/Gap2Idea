# -*- coding: utf-8 -*-
"""Funnel vs whole-paper prompting with two Stage C model sizes (Qwen3.8-27B, Qwen3.5-4B), from saved verdicts.

Setups compared with the same model's whole-paper baseline (task 5a setup A):
  published   zero-shot NLI Stage B, 20 candidates per call (the paper's funnel)
  ft_batch20  fine-tuned NLI Stage B, 20 per call
  ft_paper    fine-tuned NLI Stage B, one call per paper, ids-only answer

  python llm_size_compare.py      writes docs/experiments/llm_size/results.json and prints the table
"""
import json
import subprocess
from pathlib import Path

import funnel_nli_ft as f

ROOT = f.ROOT
OUT = ROOT / "docs/experiments/llm_size"
SMALL = OUT / "qwen3.5-4b"
EVAL = {"gold": "arxiv", "sal": "all", "fws": "all"}  # gold: the whole-paper baseline covers arXiv only


def jl_text(t):
    return [json.loads(x) for x in t.splitlines() if x.strip()]


def whole(model, name, mode):
    prompt = f.MODES[mode]
    file = {"gold": f"gold_test_A_{prompt}", "sal": f"sal_test_A_{prompt}", "fws": "fws_test_A_validate_fws"}[name]
    if model == "27b":
        txt = subprocess.run(["git", "show", f"{f.BASELINE_REF}:docs/experiments/baseline_qwen/{file}.jsonl"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8", check=True).stdout
    else:
        txt = (SMALL / "whole_paper" / f"{file}.jsonl").read_text(encoding="utf-8")
    recs = jl_text(txt)
    return {tuple(k) for r in recs if r["status"] == "ok" for k in r["kept"]}, recs


def batch20(folder, stage_b, name, mode):
    f.STAGE_B = stage_b
    s = f.SETS[name]()
    recs = f.jl(folder / f"{name}_{mode}.jsonl")
    flags = {r["start"] + j: k for r in recs for j, k in enumerate(r["keep"])}
    assert len(flags) == len(s["pool"]), (folder, name, mode, len(flags), len(s["pool"]))
    return {s["keys"][i] for i, k in flags.items() if k}, recs


def per_paper(folder, name, mode):
    recs = f.jl(folder / f"{name}_paper_{mode}.jsonl")
    return {tuple(r["keys"][i]) for r in recs for i in r["positives"]}, recs


def cost(recs, arxiv_only=False):
    if arxiv_only:
        recs = [r for r in recs if not str(r.get("unit", "")).startswith("PMC")] if recs and "unit" in recs[0] else recs
    return {"calls": len(recs), "input": sum(r.get("prompt_tokens", 0) for r in recs),
            "output": sum(r.get("completion_tokens", 0) for r in recs), "seconds": round(sum(r.get("seconds", 0) for r in recs), 1)}


def published_27b(name, rows, mode):
    return f.published(name, rows, mode), []


def main():
    f.STAGE_B = "ft"
    out = {}
    for name in ("gold", "sal", "fws"):
        rows = f.SETS[name]()["rows"]
        R = [r for r in rows if r["dom"] == "arxiv_cs"] if EVAL[name] == "arxiv" else rows
        keys = {(r["pid"], r["idx"]) for r in R}
        for mode in f.MODES:
            for model in ("27b", "4b"):
                base, brecs = whole(model, name, mode)
                runs = {}
                if model == "27b":
                    runs["published"] = published_27b(name, rows, mode)
                    runs["ft_batch20"] = batch20(ROOT / "docs/experiments/funnel_nli_ft", "ft", name, mode)
                    runs["ft_paper"] = per_paper(ROOT / "docs/experiments/stage_c_per_paper_ft", name, mode)
                else:
                    for label, folder, sb in (("published", SMALL / "funnel_zs_batch20", "zero_shot"),
                                              ("ft_batch20", SMALL / "funnel_ft_batch20", "ft")):
                        if (folder / f"{name}_{mode}.jsonl").exists():
                            runs[label] = batch20(folder, sb, name, mode)
                    if (SMALL / "per_paper_ft" / f"{name}_paper_{mode}.jsonl").exists():
                        runs["ft_paper"] = per_paper(SMALL / "per_paper_ft", name, mode)
                block = {"whole_paper": {"PRF": f.prf(R, base), "cost": cost(brecs)}}
                for label, (kept, recs) in runs.items():
                    block[label] = {"PRF": f.prf(R, kept), "minus_whole": f.paired(R, kept & keys, base & keys),
                                    "cost": cost(recs, arxiv_only=(name == "gold"))}
                    fb = [r for r in recs if r.get("fallback")]
                    if fb:
                        block[label]["fallback_batches"] = len(fb)
                        block[label]["dropped_missing"] = sum(len(r.get("dropped_missing", [])) for r in fb)
                out[f"{name}/{mode}/{model}"] = block
                line = f"{name:4s} {mode:14s} {model:3s} whole F1 {block['whole_paper']['PRF'][2]:.3f}"
                for label in ("published", "ft_batch20", "ft_paper"):
                    if label in block:
                        d = block[label]["minus_whole"]
                        line += f" | {label} {block[label]['PRF'][2]:.3f} ({d['f1_diff']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}])"
                print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
