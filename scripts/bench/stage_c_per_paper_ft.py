# -*- coding: utf-8 -*-
"""Stage C with one call per paper on the fine-tuned Stage B candidates, the same call shape as the whole-paper
baseline (task 5a setup A): all of a paper's remaining sentences in one call, numbered "[id] sentence" in reading
order, ids-only answer. Two fixed variants, both reported:

  paper      the numbered candidates only
  paper_abs  the paper's abstract first, as unnumbered context, then the numbered candidates
             (arXiv abstracts from the arXiv API, data/gold/arxiv_test_abstracts.json; SAL abstracts are the
             ABSTRACT section; FWS has no abstracts, so this variant skips FWS)

Candidates, thresholds and rubrics as in funnel_nli_ft.py (fine-tuned NLI, recall >= 90% rule on dev data,
rubrics narrowed per benchmark). Qwen token usage and call time are logged.

  python stage_c_per_paper_ft.py run --variant paper|paper_abs [--port-env NAME]
  python stage_c_per_paper_ft.py score
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from funnel_nli_ft import MODEL, MODES, SETS, baseline, jl, paired, prf, published
from funnel_nli_ft import path as batch20_path
from gold_label_full import load_corpus
from nli_sal_fws import future_scope, load_all
from nli_variants import ROOT
from stage_c_prompt_vs_model import parse_json_response, qwen_client
from stage_c_rubric_val import EXTRA, RUBRICS
from stage_c_sal40_modes import limitation_scope

OUTDIR = Path(os.environ.get("PERPAPER_OUTDIR", ROOT / "docs/experiments/stage_c_per_paper_ft"))
ITEMS = ('Each item has: "i" (integer id), "type" (the predicted gap type), "sentence" (the candidate to judge), and '
         '"context" (surrounding text). Judge the "sentence", using "context" only to disambiguate.')
NEW_ITEMS = {
    "paper": ('The input is the candidate sentences of ONE scientific paper, in reading order, one per line as '
              '"[id] sentence". Sentences between candidates were removed. Judge each candidate on its own, using the '
              'other candidates only to disambiguate.'),
    "paper_abs": ('The input is ONE scientific paper: first its abstract, after the line "ABSTRACT:", given only as '
                  'context; then its candidate sentences in reading order, after the line "CANDIDATES:", one per line '
                  'as "[id] sentence". Sentences between candidates were removed. Judge each candidate on its own, '
                  'using the abstract and the other candidates only to disambiguate.'),
}
OUTPUT = ('Return JSON of the form {"results":[{"id":<i>,"keep":<bool>}, ...]} with EXACTLY one entry per input '
          'item, where id is the item\'s "i" and keep=true means YES.')
NEW_OUTPUT = ('Return JSON of the form {"positives":[<id>, ...]}: the ids of every numbered sentence you answer YES '
              'for, in increasing order. Use only ids that appear in the input. If none qualifies, return '
              '{"positives":[]}.')
POS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["positives"],
              "properties": {"positives": {"type": "array", "items": {"type": "integer"}}}}
SCOPE = {"gold": lambda t: t, "sal": limitation_scope, "fws": future_scope}


def prompt(name, mode, variant):
    text = SCOPE[name](RUBRICS[MODES[mode]])
    for old, new in ((ITEMS, NEW_ITEMS[variant]), (OUTPUT, NEW_OUTPUT)):
        if text.count(old) != 1:
            sys.exit(f"{name} {mode}: rubric text changed; not found: {old[:50]}")
        text = text.replace(old, new)
    return text


def abstracts():
    out = json.loads((ROOT / "data/gold/arxiv_test_abstracts.json").read_text(encoding="utf-8"))
    sal = {}
    for r in load_all()["sal_test"]:
        if r["sec"] == "ABSTRACT":
            sal.setdefault(r["pmcid"], []).append(r["sent"])
    for p in load_corpus():  # gold's biomedical papers, by their BioC sections
        if p.get("secs"):
            sents = [s for s, sec in zip(p["sents"], p["secs"]) if sec == "ABSTRACT"]
            if sents:
                sal.setdefault(p["paper_id"], sents)
    out.update({k: " ".join(v) for k, v in sal.items()})
    return out


def units(s, variant, abs_text):
    by_paper = {}
    for i, k in enumerate(s["keys"]):
        by_paper.setdefault(k[0], []).append((k[1], s["pool"][i]["gap_sentence"]))
    for pid, items in by_paper.items():
        items.sort()
        lines = [f"[{j}] {sent}" for j, (_, sent) in enumerate(items)]
        if variant == "paper_abs":
            if pid not in abs_text:
                sys.exit(f"no abstract for {pid}")
            user = "ABSTRACT:\n" + abs_text[pid] + "\n\nCANDIDATES:\n" + "\n".join(lines)
        else:
            user = "\n".join(lines)
        yield {"unit": pid, "keys": [[pid, idx] for idx, _ in items], "user": user}


def path(name, variant, mode):
    return OUTDIR / f"{name}_{variant}_{mode}.jsonl"


def ask(client, system, u):
    n = len(u["keys"])
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=64 + 8 * n,
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


def names_for(variant):
    return [n for n in SETS if not (variant == "paper_abs" and n == "fws")]


def run(args):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    client = qwen_client()
    abs_text = abstracts() if args.variant == "paper_abs" else {}
    for name in names_for(args.variant):
        s = SETS[name]()
        todo = list(units(s, args.variant, abs_text))
        for mode in MODES:
            p = path(name, args.variant, mode)
            done = {r["unit"] for r in jl(p)}
            system = prompt(name, mode, args.variant)
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
                            print(f"{name} {mode} {u['unit']}: {type(e).__name__}: {e}", flush=True)
                    if rec is None:
                        sys.exit(f"{name} {mode} {u['unit']}: failed twice. Rerun to resume.")
                    rec["seconds"] = round(time.time() - t0, 2)
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    print(f"{name} {args.variant} {mode} {k}/{len(todo)} kept {len(pos)}/{len(u['keys'])} "
                          f"in {tin} out {tout} {rec['seconds']}s", flush=True)


def cost(recs):
    return {"calls": len(recs), "input_tokens": sum(r["prompt_tokens"] for r in recs),
            "output_tokens": sum(r["completion_tokens"] for r in recs), "seconds": round(sum(r["seconds"] for r in recs), 1)}


def score(_args):
    out = {}
    for name in SETS:
        s = SETS[name]()
        rows = s["rows"]
        evals = {"all": rows}
        if name == "gold":
            evals["arxiv"] = [r for r in rows if r["dom"] == "arxiv_cs"]
        block = {}
        for mode in MODES:
            base, brecs = baseline(name, mode)
            pub = published(name, rows, mode)
            b20 = jl(batch20_path(name, mode))
            k20 = {s["keys"][r["start"] + j] for r in b20 for j, k in enumerate(r["keep"]) if k}
            res = {"whole_paper_cost": cost(brecs)}
            for variant in NEW_ITEMS:
                p = path(name, variant, mode)
                if not p.exists():
                    continue
                recs = jl(p)
                kept = {tuple(r["keys"][i]) for r in recs for i in r["positives"]}
                v = {"cost_all": cost(recs)}
                if name == "gold":
                    v["cost_arxiv"] = cost([r for r in recs if not r["unit"].startswith("PMC")])
                for label, R in evals.items():
                    keys = {(r["pid"], r["idx"]) for r in R}
                    e = {"P_R_F1": prf(R, kept), "minus_batch20_ft": paired(R, kept & keys, k20 & keys),
                         "minus_published": paired(R, kept & keys, pub & keys)}
                    if name != "gold" or label == "arxiv":
                        e["minus_whole_paper"] = paired(R, kept & keys, base & keys)
                    v[label] = e
                res[variant] = v
            for label, R in evals.items():
                keys = {(r["pid"], r["idx"]) for r in R}
                res[f"reference_{label}"] = {"batch20_ft": prf(R, k20), "published": prf(R, pub)}
                if name != "gold" or label == "arxiv":
                    res[f"reference_{label}"]["whole_paper"] = prf(R, base)
            block[mode] = res
        out[name] = block
    (OUTDIR / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "score"))
    ap.add_argument("--variant", choices=tuple(NEW_ITEMS))
    args = ap.parse_args()
    if args.cmd == "run" and not args.variant:
        sys.exit("run needs --variant")
    {"run": run, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
