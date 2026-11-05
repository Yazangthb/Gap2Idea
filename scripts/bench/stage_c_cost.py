# -*- coding: utf-8 -*-
"""Estimated LLM input tokens: NLI + Qwen funnel vs sending everything to the LLM. No model calls.

Counts the prompt and items as they are actually sent, with tiktoken cl100k_base, the vendor-neutral
tokenizer of the earlier 3.74x estimate. Output tokens are left out. Stage A and B use no LLM.
  whole_paper     one call per paper: rubric + numbered sentences
  every_sentence  every sentence judged at batch 20, in the funnel's item format
  funnel          Stage C on the candidates, batch 20, per mode (modes differ only in prompt length)
"""
import json
import math

import tiktoken

from nli_sal_fws import KEEP_SECTIONS, OUT as NLI_OUT, SCORE_FILES, evc, future_scope, load_all, stage_a
from stage_c_nli_qwen import TAUS, TYPES, candidates, setup
from stage_c_prompt_vs_model import ROOT
from stage_c_rubric_val import BATCH, RUBRICS
from stage_c_sal40_modes import MODES, limitation_scope

OUT = ROOT / "docs/experiments/stage_c_cost.json"
ENC = tiktoken.get_encoding("cl100k_base")


def tok(text):
    return len(ENC.encode(text, disallowed_special=()))


def batch_tokens(items, prompt):
    """items: (type, context, sentence), batched in reading order exactly as the funnel sends them."""
    total, fixed = 0, tok(prompt)
    for start in range(0, len(items), BATCH):
        chunk = items[start:start + BATCH]
        payload = [{"i": start + k, "type": t, "context": c[:600], "sentence": s[:400]} for k, (t, c, s) in enumerate(chunk)]
        total += fixed + tok(json.dumps(payload, ensure_ascii=False))
    return total, math.ceil(len(items) / BATCH)


def paper_tokens(papers, prompt):
    fixed = tok(prompt)
    return sum(fixed + tok("\n".join(f"[{k}] {s}" for k, s in enumerate(sents))) for sents in papers.values())


def report(name, rows, candidate_ids, gap_type, scope, types=None):
    """rows: dicts with pmcid, sent. types: optional predicted type per candidate position."""
    def window(i):
        return " ".join(rows[j]["sent"] for j in range(i - 1, i + 2)
                        if 0 <= j < len(rows) and rows[j]["pmcid"] == rows[i]["pmcid"])
    papers = {}
    for r in rows:
        papers.setdefault(r["pmcid"], []).append(r["sent"])
    every = [(gap_type, window(i), rows[i]["sent"]) for i in range(len(rows))]
    cands = [((types[k] if types else gap_type), window(i), rows[i]["sent"]) for k, i in enumerate(candidate_ids)]
    main_prompt = scope(RUBRICS["strict"])
    whole = paper_tokens(papers, main_prompt)
    every_total, every_calls = batch_tokens(every, main_prompt)
    out = {"papers": len(papers), "sentences": len(rows), "candidates": len(cands),
           "whole_paper": {"tokens": whole, "calls": len(papers)},
           "every_sentence": {"tokens": every_total, "calls": every_calls}}
    for mode, rubric in MODES.items():
        total, calls = batch_tokens(cands, scope(RUBRICS[rubric]))
        out[mode] = {"tokens": total, "calls": calls, "per_paper": round(total / len(papers)),
                     "whole_paper_times": round(whole / total, 2), "every_sentence_times": round(every_total / total, 2)}
    print(name, json.dumps(out), flush=True)
    return out


def main():
    data = load_all()
    t = json.loads(NLI_OUT.read_text(encoding="utf-8"))["thresholds"]
    same = lambda r: r
    result = {"tokenizer": "cl100k_base", "input_tokens_only": True}

    sal = data["sal_test"]
    s = evc(SCORE_FILES["sal_test"], "lim_c")
    ids = [i for i, r in enumerate(sal) if stage_a("sal_test", r) and s[i] >= t["sal"]["tau"]]
    result["sal40_nli"] = report("sal40_nli", sal, ids, "limitation", limitation_scope)
    bert = [i for i, r in enumerate(sal) if r["sec"] in KEEP_SECTIONS and r["preds"]["PubMedBERT-FT"] == 1]
    result["sal40_pubmedbert"] = report("sal40_pubmedbert", sal, bert, "limitation", limitation_scope)

    fws = data["fws_test"]
    s = evc(SCORE_FILES["fws_test"], "fw_c")
    ids = [i for i, r in enumerate(fws) if s[i] >= t["fws"]["tau"]]
    result["fws_nli"] = report("fws_nli", fws, ids, "future_work", future_scope)

    rows, typed, _ = setup()
    pool = candidates(rows, typed, TAUS["r90"])
    for part, dom in (("gold_all", None), ("gold_arxiv", "arxiv_cs")):
        use = [r for r in rows if dom is None or r["dom"] == dom]
        shaped = [{"pmcid": r["pid"], "sent": r["sentence"]} for r in use]
        where = {(r["pid"], r["idx"]): k for k, r in enumerate(use)}
        chosen = [c for c in pool if (c["id"], c["idx"]) in where]
        result[part] = report(part, shaped, [where[(c["id"], c["idx"])] for c in chosen], "research_gap", same,
                              types=[c["gap_type"] for c in chosen])
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
