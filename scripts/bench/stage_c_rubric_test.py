# -*- coding: utf-8 -*-
"""Three Stage C rubrics on the gold TEST split, each reported as a fixed operating point.

Rubric wording, hypotheses and tau 0.354 were frozen before this run (stage_c_rubric_val.md).
Nothing here chooses between rubrics. Keep-biased is judged again from scratch, so comparing it
with the earlier verdicts (stage_c_nli_r90_verdicts.jsonl) also measures run-to-run stability.
"""
import json
import sys
import time

from nli_variants import ROOT
from stage_c_heldout import selection_keys
from stage_c_nli_qwen import TAUS, candidates, paired, score_set, setup
from stage_c_prompt_vs_model import qwen_client
from stage_c_rubric_val import BATCH, RUBRICS, judge

OUT = ROOT / "docs/experiments/stage_c_rubric_test.json"
EARLIER = ROOT / "docs/experiments/stage_c_nli_r90_verdicts.jsonl"


def run_one(name, prompt, pool, client):
    ckpt = ROOT / f"docs/experiments/stage_c_rubric_test_{name}.jsonl"
    done = {}
    if ckpt.exists():
        for raw in ckpt.read_text(encoding="utf-8").splitlines():
            r = json.loads(raw)
            done[r["i"]] = r["keep"]
    failed, t0 = [], time.time()
    with ckpt.open("a", encoding="utf-8") as fh:
        for start in range(0, len(pool), BATCH):
            idx = range(start, min(start + BATCH, len(pool)))
            if all(i in done for i in idx):
                continue
            got = {}
            for _attempt in range(2):
                try:
                    got = judge(client, pool, start, prompt)
                    if len(got) == len(idx):
                        break
                except Exception as e:  # reported; never counted as a verdict
                    print(f"{name} batch {start}: {type(e).__name__}: {e}", flush=True)
            if len(got) < len(idx):
                failed.append(start)
            for i, keep in sorted(got.items()):
                done[i] = keep
                fh.write(json.dumps({"i": i, "keep": keep}) + "\n")
            fh.flush()
            print(f"{name} {min(start + BATCH, len(pool))}/{len(pool)} [{time.time() - t0:.0f}s]", flush=True)
    return done, failed


def main():
    rows, typed, _ = setup()
    pool = candidates(rows, typed, TAUS["r90"])
    if len(pool) != 2522:
        sys.exit(f"Expected the 2,522 candidates of the frozen run, got {len(pool)}")
    seen = selection_keys()
    client = qwen_client()
    result, kept_sets = {"candidates": len(pool)}, {}
    for name, prompt in RUBRICS.items():
        done, failed = run_one(name, prompt, pool, client)
        if failed or len(done) < len(pool):
            sys.exit(f"{name}: incomplete, failed batches {failed}. Rerun to resume.")
        kept = {(pool[i]["id"], pool[i]["idx"]) for i, k in done.items() if k}
        kept_sets[name] = kept
        gaps_in = sum(c["label"] != "none" for c in pool)
        tp = sum(k and pool[i]["label"] != "none" for i, k in done.items())
        fp = sum(k and pool[i]["label"] == "none" for i, k in done.items())
        result[name] = {"keeps_gaps": round(tp / gaps_in, 3),
                        "removes_negatives": round(1 - fp / (len(pool) - gaps_in), 3),
                        "after": score_set(rows, kept),
                        "excluding_196_selection_sentences": score_set(rows, kept, seen)}
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")

    earlier = {}
    for raw in EARLIER.read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        earlier[r["i"]] = r["keep"]
    now = {}
    for raw in (ROOT / "docs/experiments/stage_c_rubric_test_keep_biased.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(raw)
        now[r["i"]] = r["keep"]
    flips = [i for i in now if now[i] != earlier[i]]
    result["keep_biased_rerun_vs_earlier"] = {
        "verdicts": len(now), "changed": len(flips),
        "changed_on_gaps": sum(pool[i]["label"] != "none" for i in flips)}
    result["neutral_minus_keep_biased"] = paired(rows, kept_sets["neutral"], kept_sets["keep_biased"])
    result["strict_minus_keep_biased"] = paired(rows, kept_sets["strict"], kept_sets["keep_biased"])
    result["strict_minus_neutral"] = paired(rows, kept_sets["strict"], kept_sets["neutral"])
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
