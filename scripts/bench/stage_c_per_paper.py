# -*- coding: utf-8 -*-
"""Stage C with one call per paper: all of a paper's Stage A+B candidates, in reading order.

Task 5a gave Qwen whole papers (one call per paper) and beat the funnel on arXiv, while the funnel
sent 20 candidates per call, each with its own context. This tests the funnel with the same call
shape: the candidates NLI kept (short NLI, tau 0.354, unchanged), grouped by paper.

Variants (both answer with ids only, {"positives": [...]}, ids 0..n-1 per call):
  paper       numbered candidates only, no context
  paper_ctx1  numbered candidates plus their +-1 neighbouring sentences as unnumbered context lines;
              overlapping windows are merged and "..." marks removed text

Protocol, fixed before any call: both variants run on the gold VALIDATION split in all three
modes. The variant with the higher mean F1 over the three modes is chosen and run once on the
TEST split. Test results of the other variant are never computed.

  python stage_c_per_paper.py estimate                      token counts per variant, no calls
  python stage_c_per_paper.py run   --split val|test --variant V
  python stage_c_per_paper.py score --split val|test --variant V
"""
import argparse
import json
import sys
import time

import numpy as np

from nli_variants import ROOT, load_val
from stage_c_nli_qwen import SCORES as TEST_SCORES, TAUS, TYPES, candidates, evc, paired, score_set
from stage_c_prompt_vs_model import RUBRIC, parse_json_response, qwen_client
from stage_c_rubric_val import EXTRA, RUBRICS, SCORES as VAL_SCORES

OUTDIR = ROOT / "docs/experiments/stage_c_per_paper"
ITEMS = ('Each item has: "i" (integer id), "type" (the predicted gap type), "sentence" (the candidate to judge), and '
         '"context" (surrounding text). Judge the "sentence", using "context" only to disambiguate.')
NEW_ITEMS = {
    "paper": ('The input is the candidate sentences of ONE scientific paper, in reading order, one per line as '
              '"[id] sentence". Sentences between candidates were removed. Judge each candidate on its own, using the '
              'other candidates only to disambiguate.'),
    "paper_ctx1": ('The input is excerpts of ONE scientific paper, in reading order. Candidate sentences are numbered, '
                   'one per line as "[id] sentence". Unnumbered lines are neighbouring sentences given only as context, '
                   'and "..." marks removed text. Judge only the numbered sentences, using the context only to '
                   'disambiguate.'),
}
OUTPUT = ('Return JSON of the form {"results":[{"id":<i>,"keep":<bool>}, ...]} with EXACTLY one entry per input '
          'item, where id is the item\'s "i" and keep=true means YES.')
NEW_OUTPUT = ('Return JSON of the form {"positives":[<id>, ...]}: the ids of every numbered sentence you answer YES '
              'for, in increasing order. Use only ids that appear in the input. If none qualifies, return '
              '{"positives":[]}.')
POS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["positives"],
              "properties": {"positives": {"type": "array", "items": {"type": "integer"}}}}
MODES = ("keep_biased", "neutral", "strict")


def prompt(mode, variant):
    text = RUBRICS[mode]
    for old, new in ((ITEMS, NEW_ITEMS[variant]), (OUTPUT, NEW_OUTPUT)):
        if text.count(old) != 1:
            sys.exit(f"Rubric text changed; not found: {old[:50]}")
        text = text.replace(old, new)
    return text


def load(split):
    _, rows = load_val({"val": "val", "test": "test"}[split])
    npz = np.load(VAL_SCORES if split == "val" else TEST_SCORES)
    typed = np.stack([evc(npz, h) for h in TYPES], 1)
    pool = candidates(rows, typed, TAUS["r90"])
    want = {"val": None, "test": 2522}[split]
    if want and len(pool) != want:
        sys.exit(f"Expected {want} candidates, got {len(pool)}")
    return rows, pool


def units(rows, pool, variant):
    """One call per paper. keys[k] is the (paper, idx) of numbered line k."""
    sents = {}
    for r in rows:
        sents.setdefault(r["pid"], {})[r["idx"]] = r["sentence"]
    by_paper = {}
    for c in pool:
        by_paper.setdefault(c["id"], []).append(c["idx"])
    for pid, idxs in by_paper.items():
        idxs = sorted(idxs)
        cand = set(idxs)
        if variant == "paper":
            show = idxs
        else:
            show = sorted({j for i in idxs for j in (i - 1, i, i + 1) if j in sents[pid]})
        lines, keys, prev = [], [], None
        for j in show:
            if prev is not None and j != prev + 1 and variant == "paper_ctx1":
                lines.append("...")
            if j in cand:
                lines.append(f"[{len(keys)}] {sents[pid][j]}")
                keys.append([pid, j])
            else:
                lines.append(sents[pid][j])
            prev = j
        yield {"unit": pid, "keys": keys, "user": "\n".join(lines)}


def path(split, variant, mode):
    return OUTDIR / f"{split}_{variant}_{mode}.jsonl"


def jl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] if p.exists() else []


def ask(client, system, u):
    n = len(u["keys"])
    resp = client.chat.completions.create(
        model="qwen3.8-27b", temperature=0.0, max_tokens=64 + 8 * n,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": u["user"]}],
        response_format={"type": "json_schema", "json_schema": {"name": "positives", "schema": POS_SCHEMA, "strict": True}},
        extra_body=EXTRA)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("answer hit max_tokens")
    got = parse_json_response(choice.message.content).get("positives")
    if not isinstance(got, list) or not all(isinstance(i, int) for i in got):
        raise ValueError(f"bad positives: {str(got)[:100]}")
    return (sorted({i for i in got if 0 <= i < n}), sorted({i for i in got if not 0 <= i < n}),
            resp.usage.prompt_tokens, resp.usage.completion_tokens)


def run(args):
    rows, pool = load(args.split)
    client = qwen_client()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        p = path(args.split, args.variant, mode)
        done = {r["unit"] for r in jl(p)}
        system = prompt(mode, args.variant)
        todo = list(units(rows, pool, args.variant))
        with p.open("a", encoding="utf-8") as fh:
            for k, u in enumerate(todo, 1):
                if u["unit"] in done:
                    continue
                t0, rec = time.time(), None
                for _attempt in range(2):
                    try:
                        pos, bad, tin, tout = ask(client, system, u)
                        rec = {"unit": u["unit"], "keys": u["keys"], "positives": pos, "invalid": bad,
                               "prompt_tokens": tin, "completion_tokens": tout}
                        break
                    except Exception as e:  # reported; never counted as a verdict
                        print(f"{mode} {u['unit']}: {type(e).__name__}: {e}", flush=True)
                if rec is None:
                    sys.exit(f"{mode} {u['unit']}: failed twice. Rerun to resume.")
                rec["seconds"] = round(time.time() - t0, 2)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"{args.split} {args.variant} {mode} {k}/{len(todo)} kept {len(pos)}/{len(u['keys'])} "
                      f"in {tin} out {tout}", flush=True)
    score(args)


def score(args):
    rows, pool = load(args.split)
    n_units = len(list(units(rows, pool, args.variant)))
    out = {"split": args.split, "variant": args.variant, "candidates": len(pool), "calls_per_mode": n_units}
    kept_sets = {}
    for mode in MODES:
        recs = jl(path(args.split, args.variant, mode))
        if len(recs) != n_units:
            sys.exit(f"{mode}: {len(recs)}/{n_units} calls saved; run first")
        kept = {tuple(r["keys"][i]) for r in recs for i in r["positives"]}
        kept_sets[mode] = kept
        out[mode] = {"after": score_set(rows, kept),
                     "input_tokens": sum(r["prompt_tokens"] for r in recs),
                     "output_tokens": sum(r["completion_tokens"] for r in recs),
                     "invalid_ids": sum(len(r["invalid"]) for r in recs),
                     "max_call_input": max(r["prompt_tokens"] for r in recs)}
        earlier = ROOT / f"docs/experiments/stage_c_rubric_{'val' if args.split == 'val' else 'test'}_{mode}.jsonl"
        if earlier.exists():
            v = {r["i"]: r["keep"] for r in jl(earlier)}
            old = {(pool[i]["id"], pool[i]["idx"]) for i, k in v.items() if k}
            out[mode]["funnel_20_per_call"] = score_set(rows, old)
            out[mode]["minus_funnel_20_per_call"] = paired(rows, kept, old)
    f1s = [out[m]["after"]["PRF"][2] for m in MODES]
    out["mean_f1"] = round(sum(f1s) / 3, 4)
    (OUTDIR / f"{args.split}_{args.variant}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def estimate(_args):
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    for split in ("val", "test"):
        rows, pool = load(split)
        for variant in NEW_ITEMS:
            us = list(units(rows, pool, variant))
            sys_tok = len(enc.encode(prompt("neutral", variant)))
            user = [len(enc.encode(u["user"])) for u in us]
            print(f"{split} {variant}: {len(us)} calls, {len(pool)} candidates, input ~{sum(user) + sys_tok * len(us):,} "
                  f"(cl100k), largest call ~{max(user) + sys_tok:,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("estimate", "run", "score"))
    ap.add_argument("--split", choices=("val", "test"))
    ap.add_argument("--variant", choices=tuple(NEW_ITEMS))
    args = ap.parse_args()
    if args.cmd != "estimate" and not (args.split and args.variant):
        sys.exit("run and score need --split and --variant")
    if args.cmd == "run" and args.split == "test":
        chosen = OUTDIR / "chosen_variant.txt"
        if not chosen.exists() or chosen.read_text().strip() != args.variant:
            sys.exit("Test runs only the variant chosen on validation (chosen_variant.txt)")
    {"estimate": estimate, "run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
