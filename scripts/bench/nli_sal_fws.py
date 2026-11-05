# -*- coding: utf-8 -*-
"""NLI Stage B + Qwen Stage C on SAL (human labels) and FWS (human labels), three modes.

Rules fixed before any test score was computed:
  Hypothesis   SAL "This is a limitation." FWS "This is future work." Each benchmark labels one type.
  Stage A      SAL: DISCUSS and CONCL sections, as in the SAL fixture. FWS: none, because it holds conclusion sections only.
  Threshold    Highest tau with recall >= 90% on the benchmark's dev data, the rule used on gold.
  Stage C      The frozen rubrics, narrowed to the benchmark's one label. Batch 20, +/-1 sentence context.
Test papers are scored once. SAL dev also gets Stage C, as a check on papers no prompt has seen.

  export DIR    write label-free sentence files for nli_score_remote.py
  taus          print thresholds and dev Stage B scores (no test data)
  run           Stage C on SAL dev, SAL test and FWS test, then all scores
"""
import ast
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_rct_pool import bioc_sections, section_of
from stage_c_prompt_vs_model import ROOT, qwen_client
from stage_c_rubric_val import KEEP_BIAS, RUBRICS
from stage_c_sal40_modes import GOLD, MODES, limitation_scope, paired, run_one, scored, swap


SAL_FIXTURE = ROOT / "stage_c_package/sal_criteria/sal_test_allmodels.json"
SAL_DIR = ROOT / "data/external/sal_rct"
SCORE_FILES = {name: ROOT / f"data/gold/nli_{name}_scores.npz" for name in ("sal_dev", "sal_test", "fws_dev", "fws_test")}
OUT = ROOT / "docs/experiments/nli_sal_fws.json"
KEEP_SECTIONS = {"DISCUSS", "CONCL"}
MIN_WORDS = 3
RECALL_RULE = 0.90


def future_scope(rubric):
    start, end = rubric.index("  (a) a limitation"), rubric.index("  (b) an explicit")
    text = rubric[:start] + rubric[end:]
    start, end = text.index("  (c) an unsolved problem"), text.index("Answer NO (drop)")
    text = text[:start].rstrip() + "\n\n" + text[end:]
    text = swap(text, "does it genuinely state a research gap?", "does it genuinely state a future-work direction proposed by the authors?")
    text = swap(text, "if the sentence states ANY of:", "if the sentence states:")
    text = swap(text, "if the sentence is instead: ", "if the sentence is instead: a limitation of the authors' own work without a proposed direction, an open problem in the field or prior work, ")
    if KEEP_BIAS in rubric:
        text = swap(text, "whether a sentence is a real gap", "whether a sentence is a real future-work direction")
        text = swap(text, "clearly is NOT a gap", "clearly is NOT a future-work direction")
    if "Be STRICT" in rubric:
        text = swap(text, "explicitly states the limitation, the proposed direction, or the unsolved problem",
                    "explicitly states the proposed direction")
        text = swap(text, "names no specific gap", "names no specific direction")
    return text


def parse_lim(x):
    return int(any(ast.literal_eval(str(x))))


def sal_rows(split):
    """Same rows, sections and labels as the SAL fixture. Checked against it for the test split."""
    df = pd.read_csv(SAL_DIR / f"{split}.csv")
    cache = ROOT / "data/bioc_cache"
    passages, rows = {}, []
    for _, r in df.iterrows():
        sent = str(r.sentences).strip()
        if len(sent.split()) < MIN_WORDS:
            continue
        pid = r.pmid
        if pid not in passages:
            passages[pid] = bioc_sections(pid, cache)
        rows.append({"pmcid": pid, "sent": sent, "sec": section_of(sent, passages[pid]), "label": parse_lim(r.limitation_annotation)})
    return rows


def check_sal_test():
    fixture = json.load(open(SAL_FIXTURE, encoding="utf-8"))
    rows = sal_rows("test")
    same_sec = sum(a["sent"] == b["sent"] and a["sec"] == b["sec"] for a, b in zip(rows, fixture))
    report = {"rows": len(rows), "fixture_rows": len(fixture), "same_sentence_and_section": same_sec,
              "limitations": sum(r["label"] for r in rows), "fixture_limitations": sum(r["label"] for r in fixture),
              "stage_a_kept": sum(r["sec"] in KEEP_SECTIONS for r in rows),
              "stage_a_limitations": sum(r["sec"] in KEEP_SECTIONS and r["label"] for r in rows)}
    print(json.dumps(report), flush=True)
    if len(rows) != len(fixture) or same_sec < 0.99 * len(fixture) or report["limitations"] != 205:
        sys.exit("Rebuilt SAL test does not match the fixture, so dev would not be built the same way")
    return fixture


def fws_rows():
    """Test: the selection in stage_c_package/breadth_funnel.py. Dev: the next papers in the same shuffle."""
    train = {json.loads(line)["sentence"].strip() for line in
             (ROOT / "data/scibert_prep/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    df = pd.read_csv(ROOT / "data/external/fws_corpus.csv")
    df = df[~df.text.str.strip().isin(train)].reset_index(drop=True)
    papers = df.id.unique().tolist()
    random.seed(0)
    random.shuffle(papers)
    groups = {pid: sub for pid, sub in df.groupby("id", sort=False)}
    picked, used = {"test": [], "dev": []}, 0
    for pid in papers:
        part = "test" if used < 1500 else "dev"
        if part == "dev" and sum(len(groups[p]) for p in picked["dev"]) >= 3000:
            break
        picked[part].append(pid)
        used += len(groups[pid])
    out = {}
    for part, ids in picked.items():
        # Every FWS positive is all lowercase and nearly every negative is not, so casing would reveal the label.
        out[part] = [{"pmcid": pid, "sent": str(s).lower(), "label": int(y)}
                     for pid in ids for s, y in zip(groups[pid].text, groups[pid].label)]
    if len(picked["test"]) != 209 or len(out["test"]) != 1506:
        sys.exit(f"FWS test selection changed: {len(picked['test'])} papers, {len(out['test'])} sentences")
    return out


def load_all():
    fws = fws_rows()
    return {"sal_dev": sal_rows("dev"), "sal_test": json.load(open(SAL_FIXTURE, encoding="utf-8")),
            "fws_dev": fws["dev"], "fws_test": fws["test"]}


def stage_a(name, row):
    if name.startswith("sal"):
        return row["sec"] in KEEP_SECTIONS
    return True  # FWS holds conclusion sections only, and the junk filter rejects its positives (checked on dev)


def evc(path, hyp):
    a = np.load(path)[f"sent__{hyp}"]
    return a[:, 0] / (a[:, 0] + a[:, 2] + 1e-9)


def threshold(rows, scores, name):
    """Highest tau whose overall recall, counting Stage A losses, stays at or above the rule."""
    passed = np.array([stage_a(name, r) for r in rows])
    y = np.array([r["label"] == 1 for r in rows])
    ranked = np.sort(scores[passed & y])[::-1]
    need = int(np.ceil(RECALL_RULE * y.sum()))
    if len(ranked) < need:
        sys.exit(f"{name}: Stage A alone keeps {len(ranked)} of {y.sum()} positives, below the recall rule")
    return float(ranked[need - 1])


def export():
    check_sal_test()
    target = Path(sys.argv[2])
    for name, rows in load_all().items():
        (target / f"nli_{name}.json").write_text(json.dumps([{"sentence": r["sent"]} for r in rows], ensure_ascii=False), encoding="utf-8")
        print(name, "rows", len(rows), "papers", len({r["pmcid"] for r in rows}), "positives", sum(r["label"] == 1 for r in rows), flush=True)


def taus(data=None):
    data = data or load_all()
    out = {}
    for bench, hyp in (("sal", "lim_c"), ("fws", "fw_c")):
        rows = data[f"{bench}_dev"]
        s = evc(SCORE_FILES[f"{bench}_dev"], hyp)
        tau = threshold(rows, s, f"{bench}_dev")
        kept = {i for i, r in enumerate(rows) if stage_a(f"{bench}_dev", r) and s[i] >= tau}
        out[bench] = {"hypothesis": hyp, "tau": round(tau, 4), "dev": scored(rows, kept, lambda i, rows=rows: rows[i]["label"] == 1)}
        print(bench, json.dumps(out[bench]), flush=True)
    return out


def pool_of(rows, ids, gap_type):
    def window(i):
        return " ".join(rows[j]["sent"] for j in range(i - 1, i + 2)
                        if 0 <= j < len(rows) and rows[j]["pmcid"] == rows[i]["pmcid"])
    return [{"gap_type": gap_type, "gap_sentence": rows[i]["sent"], "paragraph_text": window(i)} for i in ids]


def run_set(name, rows, tau, hyp, gap_type, scope, client, result):
    s = evc(SCORE_FILES[name], hyp)
    ids = [i for i, r in enumerate(rows) if stage_a(name, r) and s[i] >= tau]
    label = lambda i: rows[i]["label"] == 1
    after_a = {i for i, r in enumerate(rows) if stage_a(name, r)}
    block = {"sentences": len(rows), "papers": len({r["pmcid"] for r in rows}), "positives": sum(r["label"] == 1 for r in rows),
             "stage_a": scored(rows, after_a, label), "before_stage_c": scored(rows, set(ids), label), "kept": {}}
    pool = pool_of(rows, ids, gap_type)
    for mode, rubric_name in MODES.items():
        done, failed = run_one(f"nli_{name}_{mode}", scope(RUBRICS[rubric_name]), pool, client)
        if failed or len(done) < len(pool):
            sys.exit(f"{name} {mode}: incomplete, failed batches {failed}. Rerun to resume.")
        kept = {ids[i] for i, k in done.items() if k}
        block["kept"][mode] = kept
        block[mode] = scored(rows, kept, label)
    result[name] = {k: v for k, v in block.items() if k != "kept"}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return block["kept"]


def run():
    fixture = check_sal_test()
    data = load_all()
    data["sal_test"] = fixture
    result = {"thresholds": taus(data)}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    client = qwen_client()
    t = result["thresholds"]
    run_set("sal_dev", data["sal_dev"], t["sal"]["tau"], "lim_c", "limitation", limitation_scope, client, result)
    kept = run_set("sal_test", fixture, t["sal"]["tau"], "lim_c", "limitation", limitation_scope, client, result)
    run_set("fws_test", data["fws_test"], t["fws"]["tau"], "fw_c", "future_work", future_scope, client, result)

    # SAL test: comparison with the PubMedBERT funnel, and the kept sentences against the LLM gold.
    gold = {}
    for raw in GOLD.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            g = json.loads(raw)
            gold[(g["paper_id"], g["idx"])] = g["label"]
    seen, llm = {}, []
    for r in fixture:
        k = seen.get(r["pmcid"], 0)
        seen[r["pmcid"]] = k + 1
        llm.append(gold.get((r["pmcid"], k), "none"))
    sal = lambda i: fixture[i]["label"] == 1
    bert_ids = sorted(i for i, r in enumerate(fixture) if r["sec"] in KEEP_SECTIONS and r["preds"]["PubMedBERT-FT"] == 1)
    for mode in MODES:
        ckpt = ROOT / f"docs/experiments/stage_c_sal40_modes_limitation_{mode}.jsonl"
        verdicts = [json.loads(x) for x in ckpt.read_text(encoding="utf-8").splitlines()]
        bert = {bert_ids[v["i"]] for v in verdicts if v["keep"]}
        result["sal_test"][mode]["vs_pubmedbert_same_mode"] = paired(fixture, kept[mode], bert, sal)
        result["sal_test"][mode]["llm_limitation"] = scored(fixture, kept[mode], lambda i: llm[i] == "limitation")
        result["sal_test"][mode]["llm_any_gap"] = scored(fixture, kept[mode], lambda i: llm[i] != "none")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    {"export": export, "taus": taus, "run": run}[sys.argv[1]]()
