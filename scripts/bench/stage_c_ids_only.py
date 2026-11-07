# -*- coding: utf-8 -*-
"""Stage C on the gold TEST split with an ids-only answer: {"positives": [<i>, ...]}.

Same candidates (short NLI, tau 0.354, 2,522), same items, batches of 20, the same three rubrics and
the same server settings as stage_c_rubric_test.py. Only the answer format changes: the earlier
format returned one {"id", "keep"} entry per item, so output grew with the number of candidates
(task 5a found it about 11x the whole-paper baseline's output). Here the model lists only the
ids it keeps. Token usage is logged per call, so input and output are measured, not estimated.

  python stage_c_ids_only.py smoke        # first 3 batches of each rubric, prints verdicts vs earlier
  python stage_c_ids_only.py run          # all batches, resumable; then compares with the earlier runs
  python stage_c_ids_only.py score        # comparison only, from the saved files
"""
import json
import sys
import time

from nli_variants import ROOT
from stage_c_nli_qwen import TAUS, candidates, paired, score_set, setup
from stage_c_prompt_vs_model import parse_json_response, qwen_client
from stage_c_rubric_val import BATCH, EXTRA, RUBRICS

OUT = ROOT / "docs/experiments/stage_c_ids_only.json"
OLD_OUT = ('Return JSON of the form {"results":[{"id":<i>,"keep":<bool>}, ...]} with EXACTLY one entry per input '
           'item, where id is the item\'s "i" and keep=true means YES.')
NEW_OUT = ('Return JSON of the form {"positives":[<i>, ...]}: the "i" of every item you answer YES for, in increasing '
           'order. Use only ids that appear in the input. If no item qualifies, return {"positives":[]}.')
POS_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["positives"],
              "properties": {"positives": {"type": "array", "items": {"type": "integer"}}}}
MAX_TOKENS = 1024


def prompts():
    out = {}
    for name, text in RUBRICS.items():
        if text.count(OLD_OUT) != 1:
            sys.exit(f"{name}: answer-format sentence not found; rubric text changed")
        out[name] = text.replace(OLD_OUT, NEW_OUT)
    return out


def judge(client, pool, start, prompt):
    idx = range(start, min(start + BATCH, len(pool)))
    items = [{"i": i, "type": pool[i]["gap_type"], "context": pool[i]["paragraph_text"][:600],
              "sentence": pool[i]["gap_sentence"][:400]} for i in idx]
    resp = client.chat.completions.create(
        model="qwen3.8-27b", temperature=0.0, max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
        response_format={"type": "json_schema", "json_schema": {"name": "positives", "schema": POS_SCHEMA, "strict": True}},
        extra_body=EXTRA)
    choice = resp.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("answer hit max_tokens")
    got = parse_json_response(choice.message.content).get("positives")
    if not isinstance(got, list) or not all(isinstance(i, int) for i in got):
        raise ValueError(f"bad positives: {str(got)[:100]}")
    return {"start": start, "n": len(idx), "positives": sorted({i for i in got if i in idx}),
            "invalid": sorted({i for i in got if i not in idx}),
            "prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens}


def ckpt(name):
    return ROOT / f"docs/experiments/stage_c_ids_only_{name}.jsonl"


def load(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()] if path.exists() else []


def run_one(name, prompt, pool, client, limit=None):
    done = {r["start"] for r in load(ckpt(name))}
    starts = list(range(0, len(pool), BATCH))[:limit]
    failed, t0 = [], time.time()
    with ckpt(name).open("a", encoding="utf-8") as fh:
        for k, start in enumerate(starts, 1):
            if start in done:
                continue
            rec, t1 = None, time.time()
            for _attempt in range(2):
                try:
                    rec = judge(client, pool, start, prompt)
                    break
                except Exception as e:  # reported; never counted as a verdict
                    print(f"{name} batch {start}: {type(e).__name__}: {e}", flush=True)
            if rec is None:
                failed.append(start)
                continue
            rec["seconds"] = round(time.time() - t1, 2)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"{name} {k}/{len(starts)} kept {len(rec['positives'])}/{rec['n']} out {rec['completion_tokens']} tok "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    return failed


def earlier(name):
    return {r["i"]: r["keep"] for r in load(ROOT / f"docs/experiments/stage_c_rubric_test_{name}.jsonl")}


def compare(rows, pool, limit=None):
    result = {"candidates": len(pool), "answer_format": NEW_OUT}
    kept_sets = {}
    for name in RUBRICS:
        recs = load(ckpt(name))
        if limit is None and len(recs) != -(-len(pool) // BATCH):
            sys.exit(f"{name}: {len(recs)} batches saved; run first")
        covered = sorted(i for r in recs for i in range(r["start"], r["start"] + r["n"]))
        pos = {i for r in recs for i in r["positives"]}
        now = {i: i in pos for i in covered}
        old = earlier(name)
        flips = [i for i in covered if now[i] != old[i]]
        res = {"batches": len(recs), "verdicts": len(covered),
               "changed_vs_earlier": len(flips),
               "changed_on_gaps": sum(pool[i]["label"] != "none" for i in flips),
               "kept_now": sum(now.values()), "kept_earlier": sum(old[i] for i in covered),
               "input_tokens": sum(r["prompt_tokens"] for r in recs),
               "output_tokens": sum(r["completion_tokens"] for r in recs),
               "invalid_ids": sum(len(r["invalid"]) for r in recs),
               "seconds": round(sum(r["seconds"] for r in recs), 1)}
        arx = [r for r in recs if all(pool[i]["dom"] == "arxiv_cs" for i in range(r["start"], r["start"] + r["n"]))]
        res["arxiv_only_batches"] = {"batches": len(arx), "output_tokens": sum(r["completion_tokens"] for r in arx)}
        if limit is None:
            kept = {(pool[i]["id"], pool[i]["idx"]) for i in covered if now[i]}
            kept_old = {(pool[i]["id"], pool[i]["idx"]) for i in covered if old[i]}
            kept_sets[name] = kept
            res["after"] = score_set(rows, kept)
            res["earlier_after"] = score_set(rows, kept_old)
            res["minus_earlier"] = paired(rows, kept, kept_old)
        result[name] = res
    if limit is None:
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1), flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd not in ("smoke", "run", "score"):
        sys.exit(__doc__)
    rows, typed, _ = setup()
    pool = candidates(rows, typed, TAUS["r90"])
    if len(pool) != 2522:
        sys.exit(f"Expected the 2,522 candidates of the frozen run, got {len(pool)}")
    dom = {r["pid"]: r["dom"] for r in rows}
    for c in pool:
        c["dom"] = dom[c["id"]]
    limit = 3 if cmd == "smoke" else None
    if cmd != "score":
        client = qwen_client()
        for name, prompt in prompts().items():
            failed = run_one(name, prompt, pool, client, limit)
            if failed:
                sys.exit(f"{name}: failed batches {failed}. Rerun to resume.")
    compare(rows, pool, limit)


if __name__ == "__main__":
    main()
