# -*- coding: utf-8 -*-
"""Task 5a: Qwen without the funnel. Every sentence goes to the same Qwen the funnel uses.

Plan, prompts and frozen settings: docs/experiments/baseline_qwen.md. Commit that file before any
test call; `run --split test` refuses to start while it has uncommitted changes.

  check                         offline: load data, build every prompt, count calls and tokens,
                                rescore the saved funnel verdicts. No Qwen calls.
  run   --dataset D --setup S --prompt P --split dev|test [--limit-papers N]
                                one run, resumable; verdicts go to docs/experiments/baseline_qwen/
  run-all --split dev|test [--limit-papers N]
                                every run in RUN_PLAN, one at a time (also resumes)
  score --split dev|test [--write-plan]
                                P/R/F1 with paper-level CIs and paired differences vs the funnel;
                                writes results_<split>.md, and with --write-plan the plan's table

Datasets: sal (SAL-40, human limitation labels), gold (arXiv gold, LLM labels, arxiv_cs only),
fws (FWS, human future-work labels, lowercased). Setups: A whole paper in one call, B chunks,
C batches of 20. Prompts: keep_biased, neutral, strict (RUBRICS); fws also validate_fws.
Server: QWEN_BASE_URL and QWEN_API_KEY from the environment or .env.
"""
import argparse
import ast
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "scripts" / "bench"
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "src"))

PLAN = ROOT / "docs/experiments/baseline_qwen.md"
OUTDIR = ROOT / "docs/experiments/baseline_qwen"
MODEL = "qwen3.8-27b"
EXTRA = {"seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
RETRIES = 2  # extra attempts after the first; then the call is recorded as failed
DRAWS = 2000

# Frozen in baseline_qwen.md. chunk=None means the whole paper in one call.
SETUPS = {"A": {"chunk": None, "max_tokens": 4096},
          "B": {"chunk": 100, "max_tokens": 2048},
          "C": {"chunk": 20, "max_tokens": 1024}}
RUBRIC_NAMES = ("keep_biased", "neutral", "strict")
DATASETS = {"sal": {"setups": "ABC", "prompts": RUBRIC_NAMES},
            "gold": {"setups": "ABC", "prompts": RUBRIC_NAMES},
            "fws": {"setups": "AC", "prompts": ("validate_fws",) + RUBRIC_NAMES}}
GAP_TYPES = ("limitation", "future_work", "research_gap")

# The frozen run list, in run order. FWS future_scope runs would be added here (open question 1).
RUN_PLAN = ([("sal", s, p) for s in "ABC" for p in RUBRIC_NAMES]
            + [("gold", s, p) for s in "ABC" for p in RUBRIC_NAMES]
            + [("fws", s, "validate_fws") for s in "AC"])

POS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["positives"],
              "properties": {"positives": {"type": "array", "items": {"type": "integer"}}}}


# ---------------------------------------------------------------- prompts
def _load_defs(path, names, ns):
    """Run only the named top-level assignments and functions of a script, not its imports.

    stage_c_rubric_val.py and stage_c_sal40_modes.py import result files that are not in the
    repo, so importing them fails. This takes the rubric text and the scope functions verbatim.
    """
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        name = node.name if isinstance(node, ast.FunctionDef) else (
            node.targets[0].id if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) else None)
        if name in names:
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
    missing = [n for n in names if n not in ns]
    if missing:
        sys.exit(f"{path.name}: definitions not found: {missing}")


def funnel_prompts():
    ns = {"sys": sys}
    _load_defs(BENCH / "stage_c_prompt_vs_model.py", ["RUBRIC"], ns)
    _load_defs(BENCH / "stage_c_rubric_val.py", ["KEEP_BIAS", "NEUTRAL", "STRICT", "RUBRICS"], ns)
    _load_defs(BENCH / "stage_c_sal40_modes.py", ["swap", "limitation_scope"], ns)
    _load_defs(BENCH / "nli_sal_fws.py", ["future_scope"], ns)
    if ns["KEEP_BIAS"] not in ns["RUBRIC"]:
        sys.exit("Rubric text changed; keep-bias clause not found")
    return ns


HEAD = ("You are a precision filter for a research-gap extraction pipeline. Judge each candidate sentence:",
        "You are reading sentences from scientific papers. Judge each sentence:")
ITEMS = ('Each item has: "i" (integer id), "type" (the predicted gap type), "sentence" (the candidate to judge), '
         'and "context" (surrounding text). Judge the "sentence", using "context" only to disambiguate.',
         'The input is consecutive sentences from scientific papers, in reading order, one per line as "[id] sentence". '
         'Judge every sentence on its own, using the neighbouring sentences only to disambiguate.')
OUTPUT = ('Return JSON of the form {"results":[{"id":<i>,"keep":<bool>}, ...]} with EXACTLY one entry per input '
          'item, where id is the item\'s "i" and keep=true means YES.',
          'Return JSON of the form {"positives":[<id>, ...]}: the ids of every sentence you answer YES for, in '
          'increasing order. Use only ids that appear in the input. If no sentence qualifies, return {"positives":[]}.')
FWS_IO = ("\n\nYou will receive a NUMBERED list. For EACH item, decide keep=true if the TARGET states the AUTHORS' "
          "OWN limitation, assumption, scope restriction, or future-work direction, and keep=false otherwise "
          "(including limitations of prior or other work). Return JSON of the form "
          '{"results":[{"id":<n>,"keep":<bool>}, ...]} with EXACTLY one entry per input item, using the same ids, '
          "in order.",
          "\n\nThe input is consecutive sentences from scientific papers, in reading order, one per line as \"[id] sentence\". "
          "For EACH sentence, decide YES if it states the AUTHORS' OWN limitation, assumption, scope restriction, or "
          "future-work direction, and NO otherwise (including limitations of prior or other work). Return JSON of "
          'the form {"positives":[<id>, ...]}: the ids of every sentence you answer YES for, in increasing order. '
          'Use only ids that appear in the input. If no sentence qualifies, return {"positives":[]}.')


def build_prompt(dataset, prompt_name, ns=None):
    """The funnel's Stage C prompt with only the input and output wording changed."""
    ns = ns or funnel_prompts()
    swap = ns["swap"]
    if prompt_name == "validate_fws":
        from gap2idea.pipeline.gap_llm_filter import LLMGapFilter
        system = LLMGapFilter(backend="api", mode="validate_fws", client=object())._batch_messages(["x"])[0]["content"]
        return swap(system, *FWS_IO)
    text = ns["RUBRICS"][prompt_name]
    if dataset == "sal":
        text = ns["limitation_scope"](text)
    elif dataset == "fws":
        text = ns["future_scope"](text)
    for old, new in (HEAD, ITEMS, OUTPUT):
        text = swap(text, old, new)
    return text


# ---------------------------------------------------------------- data
def _jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def load_rows(dataset, split):
    """Rows in reading order: pid, idx, sentence, label (a gap type or 'none')."""
    if dataset in ("sal", "fws"):
        part = {"dev": "dev", "test": "test"}[split]
        rows = _jsonl(ROOT / f"data/splits/{dataset}_{part}.jsonl")
        pos = "limitation" if dataset == "sal" else "future_work"
        out = [dict(pid=r["paper_id"], idx=r["idx"], sentence=r["sentence"], label=pos if r["label"] == 1 else "none")
               for r in rows]
        if dataset == "fws" and any(r["sentence"] != r["sentence"].lower() for r in out):
            sys.exit("FWS split is not lowercased")
        return out
    from nli_variants import load_val
    _, rows = load_val({"dev": "val", "test": "test"}[split])
    return [dict(pid=r["pid"], idx=r["idx"], sentence=r["sentence"], label=r["label"])
            for r in rows if r["dom"] == "arxiv_cs"]


def papers_of(rows):
    out = {}
    for r in rows:
        out.setdefault(r["pid"], []).append(r)
    return out


def units(rows, setup):
    """Calls for one run, in reading order. A and B stay inside one paper; C takes the next 20
    sentences of the whole set, across paper boundaries, as the funnel's Stage C batches do.
    Ids are numbered 0..n-1 within each call and mapped back to (paper, idx)."""
    chunk = SETUPS[setup]["chunk"]
    groups = [rows] if setup == "C" else list(papers_of(rows).values())
    for sents in groups:
        size = chunk or len(sents)
        for s in range(0, len(sents), size):
            part = sents[s:s + size]
            yield {"unit": f"{part[0]['pid']}:{part[0]['idx']}", "keys": [[r["pid"], r["idx"]] for r in part],
                   "user": "\n".join(f"[{i}] {r['sentence']}" for i, r in enumerate(part))}


# ---------------------------------------------------------------- calls
def qwen_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    from openai import OpenAI
    url, key = os.environ.get("QWEN_BASE_URL", "").strip(), os.environ.get("QWEN_API_KEY", "").strip()
    if not url or not key:
        sys.exit("Set QWEN_BASE_URL and QWEN_API_KEY (environment or .env)")
    return OpenAI(base_url=url, api_key=key, timeout=900, max_retries=0)


def ask(client, system, unit, max_tokens):
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": unit["user"]}],
        response_format={"type": "json_schema", "json_schema": {"name": "positives", "schema": POS_SCHEMA, "strict": True}},
        extra_body=EXTRA)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("output hit max_tokens")
    got = json.loads(choice.message.content)["positives"]
    if not isinstance(got, list) or not all(isinstance(i, int) for i in got):
        raise ValueError(f"bad positives: {str(got)[:100]}")
    n = len(unit["keys"])
    return (sorted({i for i in got if 0 <= i < n}), sorted({i for i in got if not 0 <= i < n}),
            resp.usage.prompt_tokens, resp.usage.completion_tokens)


def run_path(dataset, split, setup, prompt):
    return OUTDIR / f"{dataset}_{split}_{setup}_{prompt}.jsonl"


def plan_is_committed():
    rel = PLAN.relative_to(ROOT).as_posix()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout
    logged = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel], cwd=ROOT, capture_output=True, text=True).stdout
    return not dirty.strip() and bool(logged.strip())


def run(args):
    if args.prompt not in DATASETS[args.dataset]["prompts"] or args.setup not in DATASETS[args.dataset]["setups"]:
        sys.exit(f"{args.dataset} allows setups {DATASETS[args.dataset]['setups']} and prompts {DATASETS[args.dataset]['prompts']}")
    if args.split == "test" and not plan_is_committed():
        sys.exit(f"Commit {PLAN.name} before any test call")
    rows = load_rows(args.dataset, args.split)
    todo = list(units(rows, args.setup))
    if args.limit_papers:
        keep = set(list(papers_of(rows))[:args.limit_papers])
        todo = [u for u in todo if any(k[0] in keep for k in u["keys"])]
    system = build_prompt(args.dataset, args.prompt)
    path = run_path(args.dataset, args.split, args.setup, args.prompt)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["unit"] for r in _jsonl(path)} if path.exists() else set()
    client, max_tokens = qwen_client(), SETUPS[args.setup]["max_tokens"]
    print(f"{path.name}: {len(todo)} calls, {len(done)} already saved", flush=True)
    with path.open("a", encoding="utf-8") as fh:
        for n, u in enumerate(todo, 1):
            if u["unit"] in done:
                continue
            rec, t0 = {"unit": u["unit"], "keys": u["keys"]}, time.time()
            for attempt in range(1 + RETRIES):
                try:
                    pos, bad, tin, tout = ask(client, system, u, max_tokens)
                    rec.update(status="ok", positives=pos, kept=[u["keys"][i] for i in pos], invalid_ids=bad,
                               prompt_tokens=tin, completion_tokens=tout)
                    break
                except Exception as e:  # recorded; never scored as an empty answer
                    rec.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
            rec.update(attempts=attempt + 1, seconds=round(time.time() - t0, 2))
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            detail = (f"kept {len(rec['kept'])}/{len(u['keys'])}, {rec['prompt_tokens']} tok" if rec["status"] == "ok"
                      else f"FAILED after {rec['attempts']} tries: {rec['error']}")
            print(f"  [{time.strftime('%H:%M:%S')}] {n}/{len(todo)} {u['unit']} {rec['seconds']:.1f}s {detail}", flush=True)
    print(f"done {path.name}", flush=True)


def run_all(args):
    """Every run in RUN_PLAN, one at a time. Finished calls are skipped, so this also resumes."""
    t0 = time.time()
    for k, (dataset, setup, prompt) in enumerate(RUN_PLAN, 1):
        print(f"\n=== run {k}/{len(RUN_PLAN)}: {dataset} {setup} {prompt} "
              f"(elapsed {(time.time() - t0) / 60:.1f} min) ===", flush=True)
        run(argparse.Namespace(dataset=dataset, setup=setup, prompt=prompt, split=args.split,
                               limit_papers=args.limit_papers))


# ---------------------------------------------------------------- scoring
def prf(tp, fp, gaps):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / gaps if gaps else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def per_paper(rows, kept):
    table = {}
    for r in rows:
        t = table.setdefault(r["pid"], [0, 0, 0])
        gap, k = r["label"] != "none", (r["pid"], r["idx"]) in kept
        t[0] += k and gap
        t[1] += k and not gap
        t[2] += gap
    return table


def boot(rows, kept, other=None, n=DRAWS):
    """Percentile CIs from resampling papers (random.Random(0), as in stage_c_nli_qwen.paired)."""
    a = per_paper(rows, kept)
    b = per_paper(rows, other) if other is not None else None
    papers, rng, draws = list(a), random.Random(0), []
    def total(tab, pick):
        return prf(*[sum(tab[p][j] for p in pick) for j in range(3)])
    for _ in range(n):
        pick = rng.choices(papers, k=len(papers))
        x = total(a, pick)
        draws.append(x[2] - total(b, pick)[2] if b else x)
    def ci(vals):
        vals = sorted(vals)
        return [round(vals[int(0.025 * n)], 3), round(vals[int(0.975 * n)], 3)]
    if b:
        return {"f1_diff": round(total(a, papers)[2] - total(b, papers)[2], 3), "ci95": ci(draws)}
    return {k: ci([d[j] for d in draws]) for j, k in enumerate(("precision", "recall", "f1"))}


def score_run(rows, path):
    recs = _jsonl(path)
    kept = {(k[0], k[1]) for r in recs if r["status"] == "ok" for k in r["kept"]}
    p, r, f = prf(*[sum(v[j] for v in per_paper(rows, kept).values()) for j in range(3)])
    ok = [x for x in recs if x["status"] == "ok"]
    res = {"PRF": [round(p, 3), round(r, 3), round(f, 3)], "ci95": boot(rows, kept), "kept": len(kept),
           "calls": len(recs), "failed_calls": len(recs) - len(ok),
           "failed_sentences": sum(len(x["keys"]) for x in recs if x["status"] != "ok"),
           "empty_answers": sum(not x["positives"] for x in ok),
           "invalid_ids": sum(len(x["invalid_ids"]) for x in ok),
           "input_tokens": sum(x["prompt_tokens"] for x in ok), "output_tokens": sum(x["completion_tokens"] for x in ok),
           "seconds": round(sum(x["seconds"] for x in recs), 1),
           "complete": len(recs) == len(list(units(rows, path.stem.split("_")[2])))}
    if path.stem.startswith("gold_"):
        res["recall_by_type"] = {t: round(sum((x["pid"], x["idx"]) in kept for x in rows if x["label"] == t)
                                          / max(1, sum(x["label"] == t for x in rows)), 3) for t in GAP_TYPES}
    return kept, res


def funnel_kept(dataset, rows):
    """Saved funnel verdicts as {(paper, idx)}, keyed by a short name. Only files in the repo."""
    out = {}
    if dataset == "sal":
        f = ROOT / "docs/experiments/stage_c_sal40_qwen.json"
        if f.exists():
            v = json.loads(f.read_text(encoding="utf-8"))["qwen_verdicts"]
            out["pubmedbert+qwen_rct_loose"] = {(rows[int(i)]["pid"], rows[int(i)]["idx"]) for i, k in v.items() if k}
    if dataset == "gold":
        f = ROOT / "docs/experiments/stage_c_nli_r90_verdicts.jsonl"
        if f.exists():
            out["nli_r90+qwen_keep_biased"] = {(r["id"], r["idx"]) for r in _jsonl(f) if r["keep"]}
    return out


def score(args):
    report = {}
    for dataset in DATASETS:
        rows = load_rows(dataset, args.split)
        funnels = funnel_kept(dataset, rows) if args.split == "test" else {}
        block = {"funnel": {k: {"PRF": [round(x, 3) for x in prf(*[sum(v[j] for v in per_paper(rows, kept).values())
                                                                     for j in range(3)])]}
                            for k, kept in funnels.items()}}
        for path in sorted(OUTDIR.glob(f"{dataset}_{args.split}_*.jsonl")):
            kept, res = score_run(rows, path)
            res["vs_funnel"] = {k: boot(rows, kept, fk) for k, fk in funnels.items()}
            block[path.stem] = res
        report[dataset] = block
    out = OUTDIR / f"scores_{args.split}.json"
    out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    table = results_table(report)
    (OUTDIR / f"results_{args.split}.md").write_text(table, encoding="utf-8")
    if args.write_plan:
        text = PLAN.read_text(encoding="utf-8")
        start, end = text.index(RESULTS_START) + len(RESULTS_START), text.index(RESULTS_END)
        PLAN.write_text(text[:start] + "\n" + table + text[end:], encoding="utf-8")
    print(table)


RESULTS_START, RESULTS_END = "<!-- results:start -->", "<!-- results:end -->"
NAMES = {"sal": "SAL-40", "gold": "arXiv gold", "fws": "FWS"}


def results_table(report):
    """The task's table: P, R, F1 with CIs, F1 difference from the funnel, cost, failures."""
    pct = lambda v, ci: f"{100 * v:.1f}% ({100 * ci[0]:.1f}–{100 * ci[1]:.1f})"
    f1 = lambda v, ci: f"{v:.3f} ({ci[0]:.3f}–{ci[1]:.3f})"
    lines = ["| Dataset | Setup | Prompt | Precision | Recall | F1 | F1 minus funnel | Input tokens | Run time | "
             "Failed calls | Empty answers |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    types, funnels = [], []
    for dataset, block in report.items():
        for name, res in block["funnel"].items():
            funnels.append(f"- {NAMES[dataset]}, `{name}`: P {res['PRF'][0]:.3f}, R {res['PRF'][1]:.3f}, F1 {res['PRF'][2]:.3f}")
        for stem, res in block.items():
            if stem == "funnel":
                continue
            _, _, setup, prompt = stem.split("_", 3)
            (p, r, f), ci = res["PRF"], res["ci95"]
            diff = "<br>".join(f"{d['f1_diff']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f}) vs `{k}`"
                               for k, d in res["vs_funnel"].items()) or "pending"
            failed = f"{res['failed_calls']} ({res['failed_sentences']} sentences)" if res["failed_calls"] else "0"
            mark = "" if res["complete"] else " (incomplete)"
            lines.append(f"| {NAMES[dataset]} | {setup}{mark} | {prompt} | {pct(p, ci['precision'])} | {pct(r, ci['recall'])} | "
                         f"{f1(f, ci['f1'])} | {diff} | {res['input_tokens']:,} | {res['seconds'] / 60:.1f} min | "
                         f"{failed} | {res['empty_answers']} |")
            if "recall_by_type" in res:
                t = res["recall_by_type"]
                types.append(f"| {setup} | {prompt} | " + " | ".join(f"{100 * t[g]:.1f}%" for g in GAP_TYPES) + " |")
    out = "\n".join(lines) + "\n"
    if funnels:
        out += "\nFunnel, rescored from its saved verdicts:\n" + "\n".join(funnels) + "\n"
    if types:
        out += ("\narXiv gold, recall by gap type:\n\n| Setup | Prompt | Limitation | Future work | Research gap |\n"
                "|---|---|---|---|---|\n" + "\n".join(types) + "\n")
    return out


# ---------------------------------------------------------------- offline check
def check(args):
    ns = funnel_prompts()
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        count = lambda s: len(enc.encode(s))
    except ImportError:
        count = lambda s: len(s) // 4
    (OUTDIR / "prompts").mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset, spec in DATASETS.items():
        for split in ("dev", "test"):
            rows = load_rows(dataset, split)
            info = {"papers": len(papers_of(rows)), "sentences": len(rows),
                    "positives": sum(r["label"] != "none" for r in rows)}
            if split == "test":
                for prompt in spec["prompts"]:
                    text = build_prompt(dataset, prompt, ns)
                    (OUTDIR / "prompts" / f"{dataset}_{prompt}.txt").write_text(text, encoding="utf-8")
                psize = count(build_prompt(dataset, spec["prompts"][0], ns))
                for setup in spec["setups"]:
                    us = list(units(rows, setup))
                    longest = max(count(u["user"]) for u in us)
                    info[setup] = {"calls": len(us), "input_tokens_one_prompt": sum(count(u["user"]) for u in us) + psize * len(us),
                                   "longest_call_tokens": longest + psize}
                info["funnel_rescored"] = {k: [round(x, 3) for x in prf(*[sum(v[j] for v in per_paper(rows, kept).values())
                                                                          for j in range(3)])]
                                           for k, kept in funnel_kept(dataset, rows).items()}
            summary[f"{dataset}_{split}"] = info
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    ra = sub.add_parser("run-all")
    ra.add_argument("--split", required=True, choices=("dev", "test"))
    ra.add_argument("--limit-papers", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=list(DATASETS))
    r.add_argument("--setup", required=True, choices=list(SETUPS))
    r.add_argument("--prompt", required=True, choices=RUBRIC_NAMES + ("validate_fws",))
    r.add_argument("--split", required=True, choices=("dev", "test"))
    r.add_argument("--limit-papers", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--split", required=True, choices=("dev", "test"))
    s.add_argument("--write-plan", action="store_true", help="also put the table in baseline_qwen.md")
    args = ap.parse_args()
    {"check": check, "run": run, "run-all": run_all, "score": score}[args.cmd](args)


if __name__ == "__main__":
    main()
