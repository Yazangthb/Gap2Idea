"""
Read runs/{ai,ml,math}/artifacts/ideas.tsv, render deploy/evaluator_form.gs
with the 15 real pipeline-generated ideas (5 per domain).

Run from repo root:
    python scripts/deploy/build_evaluator_form_gs.py
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pandas as pd

DOMAIN_LABELS = {"ai": "AI", "ml": "Classical ML", "math": "Mathematics"}
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
OUT = ROOT / "deploy" / "evaluator_form.gs"


def truncate(s: str, n: int) -> str:
    s = (s or "").strip().replace("\r", " ").replace("\n", " ")
    if len(s) <= n:
        return s
    return s[: n - 1].rsplit(" ", 1)[0] + "…"


def js_string(s: str) -> str:
    """Escape a Python string for safe interpolation into a JS string literal."""
    return (
        s.replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def load_domain(domain_dir: str) -> list[dict]:
    """Read ideas.tsv + idea_eval.tsv, return top 5 by judge composite score."""
    AXES = ["novelty", "specificity", "feasibility", "evidence_grounding"]
    ideas = pd.read_csv(RUNS / domain_dir / "artifacts" / "ideas.tsv", sep="\t")
    eval_path = RUNS / domain_dir / "artifacts" / "idea_eval.tsv"
    if eval_path.exists():
        evals = pd.read_csv(eval_path, sep="\t")
        m = ideas.merge(evals[["title"] + AXES], on="title", how="left")
        m["composite"] = m[AXES].mean(axis=1)
        picked = m.sort_values("composite", ascending=False).head(5)
    else:
        picked = ideas.head(5)
    out: list[dict] = []
    for _, r in picked.iterrows():
        out.append({
            "title": truncate(str(r.get("title", "")), 110),
            "rq": truncate(str(r.get("research_question", "")), 280),
            "method": truncate(str(r.get("method_sketch", "")), 380),
            "evalp": truncate(str(r.get("evaluation_plan", "")), 320),
            "risks": truncate(str(r.get("assumptions_and_risks", "")), 280),
        })
    return out


def main() -> None:
    blocks = []
    n = 0
    for key in ("ai", "ml", "math"):
        for idea in load_domain(key):
            n += 1
            blocks.append({"domain": DOMAIN_LABELS[key], "n": n, **idea})

    # JS array literal
    js_ideas = []
    for b in blocks:
        js_ideas.append(
            "  { domain: '%s', n: %d, title: '%s', rq: '%s', method: '%s', evalp: '%s', risks: '%s' }"
            % (
                b["domain"], b["n"],
                js_string(b["title"]),
                js_string(b["rq"]),
                js_string(b["method"]),
                js_string(b["evalp"]),
                js_string(b["risks"]),
            )
        )
    ideas_block = ",\n".join(js_ideas)

    script = textwrap.dedent("""\
        /**
         * Gap2Idea — evaluator-form generator (auto-built from pipeline output).
         *
         * Source: runs/{ai,ml,math}/artifacts/ideas.tsv (5 ideas per domain).
         * Regenerate by running scripts/deploy/build_evaluator_form_gs.py from repo root.
         *
         * Deploy:
         *   1. Visit https://script.google.com and click "New project".
         *   2. Paste this entire file in.
         *   3. Click "Run" (▶) on `createEvaluatorForm`. Accept Drive + Forms scopes.
         *   4. Open Executions (clock icon) → click the latest run → Logs show the
         *      Edit URL (private) and Share URL (give to evaluators).
         */

        const IDEAS = [
        __IDEAS__
        ];

        function createEvaluatorForm() {
          const form = FormApp.create('Gap2Idea — Research Idea Evaluation');
          form.setDescription(
            'Score 15 research ideas across AI, classical ML, and mathematics. ' +
            'Each idea has 3 axes (Novelty / Feasibility / Potential impact), 1-5 Likert. ' +
            'Expected time: 12-15 minutes. Responses are anonymous unless you fill in your name.'
          );
          form.setCollectEmail(false);
          form.setShowLinkToRespondAgain(false);
          form.setProgressBar(true);

          // --- Page 1: evaluator metadata
          form.addTextItem()
            .setTitle('Name / pseudonym (optional)')
            .setRequired(false);
          form.addMultipleChoiceItem()
            .setTitle('Which domain is closest to your expertise?')
            .setChoiceValues(['AI / LLMs', 'Classical ML / statistics', 'Mathematics', 'Other'])
            .setRequired(false);

          // --- One page per domain
          let currentDomain = '';
          for (const idea of IDEAS) {
            if (idea.domain !== currentDomain) {
              form.addPageBreakItem()
                .setTitle('Section: ' + idea.domain)
                .setHelpText('Five ideas in ' + idea.domain + '. Score each on three axes.');
              currentDomain = idea.domain;
            }
            const help =
              'Research question: ' + idea.rq + '\\n\\n' +
              'Method sketch: ' + idea.method + '\\n\\n' +
              'Evaluation plan: ' + idea.evalp + '\\n\\n' +
              'Assumptions and risks: ' + idea.risks;
            form.addSectionHeaderItem()
              .setTitle('Idea ' + idea.n + ' — ' + idea.title)
              .setHelpText(help);
            form.addScaleItem()
              .setTitle('Idea ' + idea.n + ' — Novelty')
              .setBounds(1, 5)
              .setLabels('Not novel', 'Highly novel')
              .setRequired(true);
            form.addScaleItem()
              .setTitle('Idea ' + idea.n + ' — Feasibility')
              .setBounds(1, 5)
              .setLabels('Very hard', 'Clearly feasible')
              .setRequired(true);
            form.addScaleItem()
              .setTitle('Idea ' + idea.n + ' — Potential impact')
              .setBounds(1, 5)
              .setLabels('Marginal', 'Field-shifting')
              .setRequired(true);
          }

          // --- Final page: free-text
          form.addPageBreakItem().setTitle('Wrap-up');
          form.addParagraphTextItem()
            .setTitle('Any general comments? (optional)')
            .setRequired(false);

          Logger.log('============================================================');
          Logger.log('Edit URL    (keep private): ' + form.getEditUrl());
          Logger.log('Share URL   (send to evaluators): ' + form.getPublishedUrl());
          Logger.log('============================================================');
        }
    """).replace("__IDEAS__", ideas_block)

    OUT.write_text(script, encoding="utf-8")
    print(f"Wrote {OUT} with {len(blocks)} ideas")


if __name__ == "__main__":
    main()
