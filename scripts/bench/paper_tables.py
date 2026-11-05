# -*- coding: utf-8 -*-
"""Write the manuscript's result-table rows straight from the saved result files, so no number is retyped.

Reads docs/experiments/{stage_c_modes,nli_sal_fws,stage_c_cost}.json (gitignored, local) and writes
manuscript_mdpi/tables/*_rows.tex.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "docs/experiments"
OUT = ROOT / "manuscript_mdpi/tables"
MODES = [("high_precision", "High precision"), ("balanced", "Balanced"), ("high_recall", "High recall")]
BS = "\\"


def num(n):
    return f"{n:,}".replace(",", "{,}")


def pct(s, k):
    v, (lo, hi) = s[k]["value"], s[k]["ci95"]
    return f"{100 * v:.1f} ({100 * lo:.1f}--{100 * hi:.1f})"


def f1(s):
    v, (lo, hi) = s["F1"]["value"], s["F1"]["ci95"]
    return f"{v:.3f} ({lo:.3f}--{hi:.3f})"


def main():
    m = json.loads((EXP / "stage_c_modes.json").read_text(encoding="utf-8"))
    n = json.loads((EXP / "nli_sal_fws.json").read_text(encoding="utf-8"))
    c = json.loads((EXP / "stage_c_cost.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    sets = [("SAL-40 test (human labels)", lambda k: n["sal_test"][k], n["sal_test"]["before_stage_c"]),
            ("SAL dev (human labels)", lambda k: n["sal_dev"][k], n["sal_dev"]["before_stage_c"]),
            ("FWS test (human labels)", lambda k: n["fws_test"][k], n["fws_test"]["before_stage_c"]),
            ("Gold test, all (LLM labels)", lambda k: m[k]["all"], None),
            ("Gold test, arXiv (LLM labels)", lambda k: m[k]["arxiv"], None),
            ("Gold test, biomedical (LLM labels)", lambda k: m[k]["biomedical"], None)]
    rows = []
    for name, get, before in sets:
        block = ([("Before Stage C", before)] if before else []) + [(label, get(k)) for k, label in MODES]
        for i, (label, s) in enumerate(block):
            head = f"{BS}multirow{{{len(block)}}}{{*}}{{{BS}parbox{{2.4cm}}{{{BS}raggedright {name}}}}}" if i == 0 else ""
            rows.append(f"{head} & {label} & {num(s['kept'])} & {pct(s, 'P')} & {pct(s, 'R')} & {f1(s)} {BS}{BS}")
        rows.append(f"{BS}midrule")
    (OUT / "modes_rows.tex").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")

    rows = []
    for k, label in MODES:
        r, s = m[k]["recall_by_type"], m[k]["no_research_gap"]
        rows.append(f"{label} & {100 * r['limitation']:.1f} & {100 * r['future_work']:.1f} & {100 * r['research_gap']:.1f} & "
                    f"{100 * s['P']['value']:.1f} & {100 * s['R']['value']:.1f} & {s['F1']['value']:.3f} {BS}{BS}")
    (OUT / "types_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for k, label in (("sal40_nli", "SAL-40, NLI"), ("sal40_pubmedbert", "SAL-40, PubMedBERT"), ("fws_nli", "FWS, NLI"),
                     ("gold_all", "Gold test, NLI"), ("gold_arxiv", "Gold test (arXiv), NLI")):
        d = c[k]
        h = d["high_precision"]
        rows.append(f"{label} & {num(d['candidates'])} & {h['calls']} & {num(h['tokens'])} & {num(d['whole_paper']['tokens'])} & "
                    f"{num(d['every_sentence']['tokens'])} & {h['whole_paper_times']:.2f} & {h['every_sentence_times']:.2f} {BS}{BS}")
    (OUT / "cost_rows.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")

    rows = []
    for k, label in MODES:
        d = n["sal_test"][k]["vs_pubmedbert_same_mode"]
        rows.append(f"{label} & {100 * d['precision_diff']:+.1f} ({100 * d['precision_ci95'][0]:+.1f}, {100 * d['precision_ci95'][1]:+.1f}) & "
                    f"{d['f1_diff']:+.3f} ({d['f1_ci95'][0]:+.3f}, {d['f1_ci95'][1]:+.3f}) & "
                    f"{100 * n['sal_test'][k]['llm_limitation']['P']['value']:.1f} & {100 * n['sal_test'][k]['llm_any_gap']['P']['value']:.1f} {BS}{BS}")
    text = "\n".join(rows).replace("+0.000", "0.000").replace("-", "$-$").replace("+", "$+$")
    (OUT / "salcmp_rows.tex").write_text(text + "\n", encoding="utf-8")
    print("wrote", sorted(p.name for p in OUT.glob("*_rows.tex")))


if __name__ == "__main__":
    main()
