# -*- coding: utf-8 -*-
"""Tables and figure for the funnel-versus-whole-paper section, straight from the saved verdicts.

Reads docs/experiments/llm_size/results.json (llm_size_compare.py) and the verdict files it scores, and writes
  manuscript_mdpi/tables/whole_table.tex      F1 of whole paper and published funnel, paired difference, per model
  manuscript_mdpi/tables/wholecost_rows.tex  measured input/output tokens and call time per setup
  manuscript_mdpi/tables/variants_rows.tex   F1 of the Stage B / Stage C variants, per model
  manuscript_mdpi/figures/whole.pdf          F1 in high-precision mode, whole paper vs funnel, per model
A model appears only when its whole-paper and published-funnel runs are complete.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import funnel_nli_ft as f  # noqa: E402
from paper_figures import INK2, MUTED, grid  # noqa: E402  (also sets the shared rcParams)

ROOT = f.ROOT
EXP = ROOT / "docs/experiments"
TAB, FIG = ROOT / "manuscript_mdpi/tables", ROOT / "manuscript_mdpi/figures"
RES = json.loads((EXP / "llm_size/results.json").read_text(encoding="utf-8"))
MODELS = [("27b", "Qwen3.8-27B"), ("9b", "Qwen3.5-9B"), ("4b", "Qwen3.5-4B")]
MODES = [("high_precision", "High precision"), ("balanced", "Balanced"), ("high_recall", "High recall")]
SETS = [("gold", "arXiv (28 papers, LLM labels)"), ("sal", "SAL-40 (human labels)"), ("fws", "FWS (human labels)")]
FOLDER = {"27b": None, "9b": EXP / "llm_size/qwen3.5-9b", "4b": EXP / "llm_size/qwen3.5-4b"}


def present(model):
    return all(f"{n}/{m}/{model}" in RES and "published" in RES[f"{n}/{m}/{model}"] for n, _ in SETS for m, _ in MODES)


MODELS = [m for m in MODELS if present(m[0])]


def diff(d, bold_sig=True):
    s = f"{d['f1_diff']:+.3f}".replace("-", "$-$").replace("+", "$+$")
    ci = f"{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}".replace("-", "$-$").replace("+", "$+$")
    sig = d["ci95"][0] > 0 or d["ci95"][1] < 0
    return (f"\\textbf{{{s}}}" if sig and bold_sig else s) + f" {{\\scriptsize[{ci}]}}"


def whole_rows():
    lines = []
    for n, label in SETS:
        for k, (m, mlabel) in enumerate(MODES):
            first = f"\\multirow{{3}}{{*}}{{\\parbox{{2.3cm}}{{\\raggedright {label}}}}}" if k == 0 else ""
            cells = []
            for model, _ in MODELS:
                b = RES[f"{n}/{m}/{model}"]
                if n == "fws" and k > 0:
                    w = "''"  # FWS whole-paper run has one prompt for all modes
                else:
                    w = f"{b['whole_paper']['PRF'][2]:.3f}"
                cells += [w, f"{b['published']['PRF'][2]:.3f}", diff(b["published"]["minus_whole"])]
            lines.append(f"{first} & {mlabel} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1]) + "\n"


def share_cost(recs, keys_of, arxiv):
    """Cost of the arXiv part of batched gold runs: each batch weighted by its share of arXiv candidates."""
    tin = tout = sec = 0.0
    for r in recs:
        ks = keys_of(r)
        s = sum(k in arxiv for k in ks) / len(ks)
        tin += r.get("prompt_tokens", 0) * s
        tout += r.get("completion_tokens", 0) * s
        sec += r.get("seconds", 0) * s
    return tin, tout, sec


def costs():
    """{(set, setup, model): (input, output, seconds)} for high-precision mode, arXiv only on gold."""
    rows = f.SETS["gold"]()["rows"]
    arxiv = {(r["pid"], r["idx"]) for r in rows if r["dom"] == "arxiv_cs"}
    out = {}
    for n, _ in SETS:
        m = "high_precision"
        for model, _ in MODELS:
            b = RES[f"{n}/{m}/{model}"]
            c = b["whole_paper"]["cost"]
            out[(n, "whole", model)] = (c["input"], c["output"], c["seconds"])
            # published funnel: logged for the smaller models; the 27B run predates token logging
            if FOLDER[model] is not None:
                f.STAGE_B = "zero_shot"
                s = f.SETS[n]()
                recs = f.jl(FOLDER[model] / "funnel_zs_batch20" / f"{n}_{m}.jsonl")
                if n == "gold":
                    kof = lambda r, s=s: [s["keys"][r["start"] + j] for j in range(len(r["keep"]))]
                    out[(n, "published", model)] = share_cost(recs, kof, arxiv)
                else:
                    out[(n, "published", model)] = (sum(r["prompt_tokens"] for r in recs),
                                                    sum(r["completion_tokens"] for r in recs),
                                                    sum(r["seconds"] for r in recs))
            pp = b.get("ft_paper")
            if pp:
                c = pp["cost"]
                out[(n, "per_paper", model)] = (c["input"], c["output"], c["seconds"])
    f.STAGE_B = "ft"
    return out


def k(x):
    return f"{x / 1000:.0f}k" if x >= 10000 else f"{x / 1000:.1f}k"


def cost_rows(c):
    lines = []
    setups = [("whole", "Whole paper"), ("published", "Funnel, 20 per call"), ("per_paper", "Funnel, one call per paper")]
    for n, label in SETS:
        short = {"gold": "arXiv", "sal": "SAL-40", "fws": "FWS"}[n]
        for j, (s, slabel) in enumerate(setups):
            first = f"\\multirow{{3}}{{*}}{{{short}}}" if j == 0 else ""
            inputs = [c[(n, s, mm)][0] for mm, _ in MODELS if (n, s, mm) in c]
            cells = [k(max(inputs)) if inputs else "--"]
            for mm, _ in MODELS:
                v = c.get((n, s, mm))
                cells += [k(v[1]) if v else "n/a", f"{v[2]:.0f}" if v else "n/a"]
            lines.append(f"{first} & {slabel} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1]) + "\n"


def variants_rows():
    lines = []
    setups = [("whole_paper", "Whole paper"), ("published", "Funnel (zero-shot NLI, 20 per call)"),
              ("ft_batch20", "Fine-tuned NLI, 20 per call"), ("ft_paper", "Fine-tuned NLI, one call per paper")]
    for n, label in SETS:
        short = {"gold": "arXiv", "sal": "SAL-40", "fws": "FWS"}[n]
        for j, (s, slabel) in enumerate(setups):
            first = f"\\multirow{{4}}{{*}}{{{short}}}" if j == 0 else ""
            cells = []
            for model, _ in MODELS:
                vals = []
                for m, _ in MODES:
                    b = RES[f"{n}/{m}/{model}"]
                    vals.append(f"{b[s]['PRF'][2]:.3f}" if s in b else "--")
                cells.append(" / ".join(vals))
            lines.append(f"{first} & {slabel} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule")
    return "\n".join(lines[:-1]) + "\n"


def figure():
    colors = {"whole": "#9a9893", "funnel": "#2a78d6"}
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.1), sharey=True)
    for ax, (n, label) in zip(axes, SETS):
        xs = range(len(MODELS))
        w = [RES[f"{n}/high_precision/{mm}"]["whole_paper"]["PRF"][2] for mm, _ in MODELS]
        p = [RES[f"{n}/high_precision/{mm}"]["published"]["PRF"][2] for mm, _ in MODELS]
        ax.bar([x - 0.19 for x in xs], w, 0.36, color=colors["whole"], label="Whole paper")
        ax.bar([x + 0.19 for x in xs], p, 0.36, color=colors["funnel"], label="Funnel")
        for x, (a, b, (mm, _)) in enumerate(zip(w, p, MODELS)):
            d = RES[f"{n}/high_precision/{mm}"]["published"]["minus_whole"]
            sig = d["ci95"][0] > 0 or d["ci95"][1] < 0
            ax.text(x, max(a, b) + 0.03, f"{d['f1_diff']:+.2f}" + ("*" if sig else ""), ha="center", fontsize=7, color=INK2)
        ax.set_xticks(list(xs), [lab.replace("Qwen", "") for _, lab in MODELS])
        ax.set_title(label, color=INK2)
        ax.set_ylim(0, 1)
        grid(ax, "y")
    axes[0].set_ylabel("$F_1$, high-precision mode")
    axes[0].legend(frameon=False, loc="upper left", ncol=2, bbox_to_anchor=(0, 1.02))
    fig.savefig(FIG / "whole.pdf")


def wrap(colspec, head, body):
    return (r"\begin{tabularx}{\fulllength}{" + colspec + "}\n" + r"\toprule" + "\n" + head + r"\midrule" + "\n"
            + body + r"\bottomrule" + "\n" + r"\end{tabularx}" + "\n")


def main():
    TAB.mkdir(exist_ok=True)
    n = len(MODELS)
    names = [lab for _, lab in MODELS]
    nl = r" \\" + "\n"
    head = (" & & " + " & ".join(r"\multicolumn{3}{c}{\textbf{" + x + "}}" for x in names) + nl
            + "".join(r"\cmidrule(lr){" + f"{3 + 3 * i}-{5 + 3 * i}" + "}" for i in range(n)) + "\n"
            + r"\textbf{Set} & \textbf{Mode} & " + " & ".join(["Whole", "Funnel", r"Funnel $-$ whole"] * n) + nl)
    spec = r"@{}>{\raggedright\arraybackslash}p{2.3cm}l" + r"rr>{\raggedleft\arraybackslash}X" * n + "@{}"
    (TAB / "whole_table.tex").write_text(wrap(spec, head, whole_rows()), encoding="utf-8")

    head = (" & & & " + " & ".join(r"\multicolumn{2}{c}{\textbf{" + x + "}}" for x in names) + nl
            + "".join(r"\cmidrule(lr){" + f"{4 + 2 * i}-{5 + 2 * i}" + "}" for i in range(n)) + "\n"
            + r"\textbf{Set} & \textbf{Setup} & \textbf{Input} & " + " & ".join(["Output", "Seconds"] * n) + nl)
    spec = r"@{}l>{\raggedright\arraybackslash}Xr" + "rr" * n + "@{}"
    (TAB / "wholecost_table.tex").write_text(wrap(spec, head, cost_rows(costs())), encoding="utf-8")

    head = r"\textbf{Set} & \textbf{Setup} & " + " & ".join(r"\textbf{" + x + "}" for x in names) + nl
    spec = r"@{}l>{\raggedright\arraybackslash}X" + "c" * n + "@{}"
    (TAB / "variants_table.tex").write_text(wrap(spec, head, variants_rows()), encoding="utf-8")
    figure()
    print("models:", [m for m, _ in MODELS])


if __name__ == "__main__":
    main()
