# -*- coding: utf-8 -*-
"""Is Stage C's recall loss on the gold corpus caused by the model or the prompt?

Judges the exact 80 candidates Sonnet judged on 2026-10-26 (balanced candidates,
random.seed(11) shuffle) with yandexgpt-5-lite, changing one thing at a time:

  control    the 2026-10-26 Yandex setup: LLMGapFilter mode=validate, no context, batch 40
  rubric_20  Sonnet's rubric and item format, batch 20
  rubric_80  Sonnet's rubric and item format, all 80 in one call (what Sonnet saw)

The rubric conditions call the API directly because LLMGapFilter._batch_messages
appends a fixed "keep=false for limitations of prior or other work" instruction,
which contradicts part (c) of the rubric.

Sonnet's per-sentence verdicts are rebuilt from its reported 12 FP and 1 FN.
Every verdict is saved to docs/experiments/stage_c_prompt_vs_model.json.
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")
os.environ["LLM_PROVIDER"] = "yandex"
from gap2idea.pipeline.gap_llm_filter import BATCH_SCHEMA, LLMGapFilter
from gap2idea.pipeline.llm import get_llm_client, parse_json_response

OUT = ROOT / "docs" / "experiments" / "stage_c_prompt_vs_model.json"
QWEN_HOST = os.environ.get("QWEN_HOST", "")  # Qwen3.8-27B-FP8 on vLLM; pass "qwen" as the argument

# The decision text given to the Sonnet subagent, verbatim. Only the file-handling
# and output-format instructions differ, because the API returns JSON directly.
RUBRIC = """You are a precision filter for a research-gap extraction pipeline. Judge each candidate sentence: does it genuinely state a research gap?

Each item has: "i" (integer id), "type" (the predicted gap type), "sentence" (the candidate to judge), and "context" (surrounding text). Judge the "sentence", using "context" only to disambiguate.

Decision — answer YES (keep) if the sentence states ANY of:
  (a) a limitation / weakness / caveat / bias / constraint / assumption of the authors' OWN work (e.g., small sample, no blinding, single setting, short follow-up, confounding, limited generalizability, unvalidated measure, "we did not / could not ...", "a limitation is ...");
  (b) an explicit FUTURE-WORK direction the authors propose ("in future work we will ...", "future research should ...", "it would be interesting to ...");
  (c) an unsolved problem / missing capability / open research gap in the field or prior work ("it remains unknown ...", "no existing method ...", "current approaches struggle with ...").

Answer NO (drop) if the sentence is instead: a result or finding, a method/experimental-setup description, a positive or contribution claim ("we propose/show/achieve ..."), background or related-work description, a citation or reference fragment, acknowledgements / funding / author contributions / competing interests, a figure or table caption, a dataset/metric definition, or a generic non-gap statement.

Be KEEP-BIASED: when you are genuinely uncertain whether a sentence is a real gap, answer YES. Only answer NO when it clearly is NOT a gap by the rules above.

Return JSON of the form {"results":[{"id":<i>,"keep":<bool>}, ...]} with EXACTLY one entry per input item, where id is the item's "i" and keep=true means YES."""

# Prefixes of Sonnet's reported verdicts, from the 2026-10-26 transcript.
SONNET_FP = [
    "Therefore, this metric would not be comparable",
    "For example, if the canonicalization network",
    "Here, discovery of subject-",
    "The results of the models built in this study, cannot",
    "Intuitively, this weight sharing",
    "This might be due to a number of reasons.",
    "Long Task Horizon:",
    "For this reason, we argue that FTL",
    "In contrast, the infinite-variance case",
    "Previous observations have highlighted that robotic policies",
    "The models are giving inaccurate predictions",
    "However, since we do not assume knowledge of task boundaries",
]
SONNET_FN = ["In South Africa, a similar relative increase"]


def load_sample():
    cands = json.loads((ROOT / "data/gold/funnel_v2_candidates_balanced.json").read_text(encoding="utf-8"))
    random.seed(11)
    random.shuffle(cands)
    return cands[:80]


def sonnet_verdicts(pool):
    verdicts = []
    for c in pool:
        s = c["gap_sentence"]
        if c["label"] == "none":
            verdicts.append(any(s.startswith(p) for p in SONNET_FP))
        else:
            verdicts.append(not any(s.startswith(p) for p in SONNET_FN))
    return verdicts


def qwen_client():
    """Settings come from .env: QWEN_HOST, plus QWEN_API_KEY or QWEN_SSH_USER and QWEN_KEY_PATH.

    A key read over SSH is kept in memory only.
    """
    from openai import OpenAI
    if not QWEN_HOST:
        sys.exit("Set QWEN_HOST in .env")
    key = os.environ.get("QWEN_API_KEY", "").strip()
    if not key:
        user, path = os.environ.get("QWEN_SSH_USER"), os.environ.get("QWEN_KEY_PATH")
        if not user or not path:
            sys.exit("Set QWEN_API_KEY, or QWEN_SSH_USER and QWEN_KEY_PATH, in .env")
        key = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", f"{user}@{QWEN_HOST}", f"cat {path}"],
            capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    if not key:
        sys.exit("Could not read the model key")
    return OpenAI(base_url=f"http://{QWEN_HOST}:8001/v1", api_key=key, timeout=900, max_retries=1)


def run_control(pool, client, model, extra=None):
    filt = LLMGapFilter(backend="api", model=model, mode="validate", client=client)
    filt.extra_body = extra
    rows = [{"id": i, "gap_type": c["gap_type"], "gap_sentence": c["gap_sentence"],
             "paragraph_text": c["paragraph_text"]} for i, c in enumerate(pool)]
    t0 = time.time()
    kept = {r["id"] for r in filt.filter_gaps(rows, protect_rules=False)}
    return [i in kept for i in range(len(pool))], filt.n_calls, time.time() - t0, 0


def run_rubric(pool, client, model, batch, extra=None, max_tokens=None):
    verdicts = [None] * len(pool)
    calls, t0 = 0, time.time()
    for start in range(0, len(pool), batch):
        idx = range(start, min(start + batch, len(pool)))
        items = [{"i": i, "type": pool[i]["gap_type"], "context": pool[i]["paragraph_text"][:600],
                  "sentence": pool[i]["gap_sentence"][:400]} for i in idx]
        resp = client.chat.completions.create(
            model=model, temperature=0.0, max_tokens=max_tokens or min(8192, 256 + 400 * len(items)),
            messages=[{"role": "system", "content": RUBRIC},
                      {"role": "user", "content": json.dumps(items, ensure_ascii=False)}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "gap_verdicts", "schema": BATCH_SCHEMA, "strict": True}},
            **({"extra_body": extra} if extra else {}))
        calls += 1
        for r in parse_json_response(resp.choices[0].message.content).get("results", []):
            if int(r["id"]) in idx:
                verdicts[int(r["id"])] = bool(r["keep"])
    missing = sum(v is None for v in verdicts)
    return verdicts, calls, time.time() - t0, missing


def score(pool, verdicts):
    """A missing verdict counts as dropped, and is reported separately."""
    gaps = sum(c["label"] != "none" for c in pool)
    tp = sum(bool(v) and c["label"] != "none" for c, v in zip(pool, verdicts))
    fp = sum(bool(v) and c["label"] == "none" for c, v in zip(pool, verdicts))
    per_type = {}
    for t in ("limitation", "future_work", "research_gap"):
        of_type = [(c, v) for c, v in zip(pool, verdicts) if c["label"] == t]
        per_type[t] = f"{sum(bool(v) for _, v in of_type)}/{len(of_type)}"
    return {"kept": tp + fp, "tp": tp, "fp": fp, "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "gaps_kept": round(tp / gaps, 3), "per_type": per_type}


def main():
    pool = load_sample()
    gaps = sum(c["label"] != "none" for c in pool)
    if gaps != 19:
        sys.exit(f"Sample mismatch: expected the 19 gold gaps of the Oct 26 sample, got {gaps}")
    sonnet = sonnet_verdicts(pool)
    sonnet_score = score(pool, sonnet)
    if (sonnet_score["tp"], sonnet_score["fp"]) != (18, 12):
        sys.exit(f"Sonnet verdicts not rebuilt: {sonnet_score}")
    print("sample OK: 80 candidates, 19 gaps; Sonnet rebuilt:", sonnet_score, flush=True)

    results = {"sonnet": {"verdicts": sonnet, **sonnet_score}}
    if len(sys.argv) > 1 and sys.argv[1] == "qwen":
        client, model, out = qwen_client(), "qwen3.8-27b", OUT.with_name(OUT.stem + "_qwen.json")
        off = {"seed": 0, "chat_template_kwargs": {"enable_thinking": False}}
        on = {"seed": 0, "chat_template_kwargs": {"enable_thinking": True}}
        runs = [("control", lambda: run_control(pool, client, model, off)),
                ("rubric_20", lambda: run_rubric(pool, client, model, 20, off)),
                ("rubric_80", lambda: run_rubric(pool, client, model, 80, off)),
                ("rubric_20_thinking", lambda: run_rubric(pool, client, model, 20, on, 16384))]
    else:
        client, out = get_llm_client("yandex"), OUT
        model = f"gpt://{os.environ['YANDEX_FOLDER_ID'].strip()}/yandexgpt-5-lite/latest"
        runs = [("control", lambda: run_control(pool, client, model)),
                ("rubric_20", lambda: run_rubric(pool, client, model, 20)),
                ("rubric_80", lambda: run_rubric(pool, client, model, 80))]
    for name, run in runs:
        verdicts, calls, secs, missing = run()
        agree = sum(bool(a) == b for a, b in zip(verdicts, sonnet))
        results[name] = {"verdicts": verdicts, "calls": calls, "seconds": round(secs, 1), "missing": missing,
                         "agree_with_sonnet": agree, **score(pool, verdicts)}
        print(name, {k: v for k, v in results[name].items() if k != "verdicts"}, flush=True)

    results["sample"] = [{"id": c["id"], "idx": c["idx"], "label": c["label"], "gap_type": c["gap_type"],
                          "gap_sentence": c["gap_sentence"]} for c in pool]
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
