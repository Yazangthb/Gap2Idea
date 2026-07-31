"""Produce a draft paper for each of the 6 v2 ideas.

Two paths:

  math ideas  → draft_full_paper with a "deepen the exposition" system-prompt
                augmentation; emphasis on motivation, theoretical context,
                related work prose, and a careful experimental plan.

  AI ideas    → first run the multi-agent sanity stage (Prototyper →
                Reviewer → sandboxed Runner → Analyst + Skeptic +
                Statistician → Verdict-Synthesiser, budget=benchmark) to
                produce an actual code-plus-stdout trace; then draft the
                paper with that trace injected as evidence, asking the
                drafter to include a "Preliminary experiment" subsection
                that reports what the toy run found.

Outputs per idea:
  artifacts/paper_drafts/<slug>.json   — full PAPER_DRAFT_SCHEMA dict
  artifacts/paper_drafts/<slug>_sanity.json   — sanity trace (AI only)
  artifacts/paper_drafts/<slug>.md     — human-readable rendered draft
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from gap2idea.pipeline.llm import get_llm_client, parse_json_response
from gap2idea.pipeline.paper_drafter import (
    DEFAULT_DRAFT_MODEL,
    PAPER_DRAFT_SCHEMA,
    SYSTEM_PROMPT as DRAFT_SYSTEM_PROMPT,
    draft_full_paper,
)
from gap2idea.pipeline.sanity import run_sanity_check
from gap2idea.utils import retry


# ----------------------------------------------------------------------
# System-prompt augmentations
# ----------------------------------------------------------------------

MATH_SYSTEM_EXTRA = (
    "\n\nThis paper is a MATHEMATICS paper. Be explicitly formal:\n"
    "  • Use LaTeX math throughout. Inline math goes between $...$; displayed "
    "    equations use $$...$$ or \\begin{equation}...\\end{equation}.\n"
    "  • Every formal object MUST be named with notation in the text: groups "
    "    $G$, measures $\\mu$, operators $T$, kernels $K(x,y)$, function "
    "    spaces $L^p(\\mathbb{R}^n)$, exponents $\\alpha, \\beta$, constants "
    "    $C, C_n, C(\\lambda, \\Lambda)$, etc.\n"
    "  • The introduction.motivation field must contain at least ONE displayed "
    "    formula stating the classical result (e.g. $\\|T f\\|_{L^p} \\le C \\|f\\|_{L^p}$) "
    "    and at least one inline formula stating the open problem.\n"
    "  • The method.approach field MUST be structured as a chain of named "
    "    theorems / lemmas / propositions. State each with:\n"
    "      Theorem (Name). Let X be ... . Suppose ... . Then ... .\n"
    "    Use Markdown bold or LaTeX `\\textbf{Theorem.}` for the labels. "
    "    Include the precise hypotheses and the precise conclusion using "
    "    real math symbols ($\\le$, $\\ge$, $\\to$, $\\Rightarrow$, etc.).\n"
    "  • The method.architecture_or_algorithm field MUST contain the actual "
    "    proof-sketch chain: \"By Lemma 1 we control X. Inserting this in (2.3) "
    "    yields $...$. Combining with Lemma 2 and the Cauchy-Schwarz inequality "
    "    we obtain $...$.\" Use displayed equations for the key estimates.\n"
    "  • The motivation section should give 3 paragraphs of theoretical "
    "    context: classically known result (with formula), the open problem "
    "    (with formula), why it matters within the relevant mathematical "
    "    area, and what techniques have historically been brought to bear.\n"
    "  • The related work section should treat each entry as a paragraph of "
    "    prose, not a one-line gloss — explain what each cited work proves, "
    "    state its main estimate or theorem in math, and how the proposal "
    "    builds on or differs from it.\n"
    "  • The experimental_setup section corresponds to NUMERICAL or COMPUTER-"
    "    ALGEBRA verification of the theory: name a concrete computer-algebra "
    "    package, an explicit parameter range (give exact intervals, e.g. "
    "    $n \\in [10^3, 10^6]$, $d \\in (\\log n)^{0.5..2}$), and explain "
    "    what such a check would or would not establish.\n"
    "  • The expected_results section should describe the QUALITATIVE shape of "
    "    the bound or theorem expected, using formulas: e.g. \"a bound of the "
    "    form $|T f|_{L^p} \\le C(p, n) \\beta_\\mu(B)^{1/2} \\|f\\|_{L^p}$\", "
    "    never numerical results.\n"
    "  • All citations must be by paper_id from the input evidence/prior_art "
    "    list — never invent paper IDs."
)

AI_PRELIM_EXPT_INSTRUCTION = (
    "\n\nA preliminary toy experiment has already been carried out via a "
    "sandboxed multi-agent pipeline (Prototyper → Reviewer → Runner → Analyst → "
    "Skeptic → Statistician → Verdict-Synthesiser) and the trace is included "
    "under `prior_art` with paper_id starting `SANITY:`. You MUST:\n"
    "  • Add a paragraph headed \"Preliminary experiment\" at the END of the "
    "    `experimental_setup.implementation_notes` field that summarises what "
    "    that toy experiment did, what numbers it produced, and the agent panel's "
    "    verdict. This is REAL measured output from a sandboxed run, so concrete "
    "    numbers from the parsed_results array are OK (and required) here.\n"
    "  • Treat the toy experiment as a sanity check ONLY. The headline contributions "
    "    of the paper remain at the planning stage; the toy run does NOT replace "
    "    the full evaluation described in the rest of the section."
)


# ----------------------------------------------------------------------
# Drafter with a customised system prompt
# ----------------------------------------------------------------------

@retry(tries=3, base_delay=2.0)
def _call_drafter_custom(client, prompt: str, system_extra: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DRAFT_SYSTEM_PROMPT + system_extra},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "paper_draft", "schema": PAPER_DRAFT_SCHEMA, "strict": True,
            },
        },
        temperature=0.4,
    )
    return parse_json_response(resp.choices[0].message.content)


def draft_with_extra(
    idea: dict,
    evidence: list[dict],
    prior_art: list[dict],
    system_extra: str,
    model: str = DEFAULT_DRAFT_MODEL,
) -> dict:
    client = get_llm_client()
    payload = {
        "idea": {
            "title": idea.get("title", ""),
            "research_question": idea.get("research_question", ""),
            "method_sketch": idea.get("method_sketch", ""),
            "evaluation_plan": idea.get("evaluation_plan", ""),
            "expected_contribution": idea.get("expected_contribution", ""),
            "assumptions_and_risks": idea.get("assumptions_and_risks", ""),
            "named_baseline": idea.get("named_baseline", ""),
            "falsifiable_prediction": idea.get("falsifiable_prediction", ""),
            "themes": [t for t in (idea.get("label_a"), idea.get("label_b")) if t],
        },
        "evidence": evidence or [],
        "prior_art": prior_art or [],
    }
    user_prompt = (
        "Draft a full paper plan for the following idea.\n"
        "Return JSON matching the schema. Remember: NO fabricated numbers, "
        "NO fabricated citations.\n\n"
        "INPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    )
    return _call_drafter_custom(client, user_prompt, system_extra, model=model)


# ----------------------------------------------------------------------
# Topic classifier (same heuristic as the picker)
# ----------------------------------------------------------------------

def is_math(idea: dict) -> bool:
    cid = int(idea.get("cluster_a", 0))
    if cid < 0:
        return True
    text = " ".join(str(idea.get(k, "")) for k in
                     ("title", "research_question", "method_sketch"))
    text_l = text.lower()
    if "helmholtz" in text_l and "stability" in text_l:
        return True
    return False


# ----------------------------------------------------------------------
# Render to readable markdown
# ----------------------------------------------------------------------

def render_md(idea: dict, draft: dict, sanity: dict | None = None) -> str:
    lines: list[str] = []
    lines.append(f"# {idea.get('title','(untitled)')}\n")
    lines.append(f"*Panel composite: {idea.get('panel_composite','?')} · "
                  f"α: {idea.get('panel_agreement','?')} · "
                  f"cluster: {idea.get('cluster_a','?')}*\n")
    lines.append("\n## Abstract\n")
    lines.append(draft.get("abstract", "(missing)") + "\n")

    intro = draft.get("introduction", {})
    lines.append("\n## 1. Introduction\n")
    lines.append("### 1.1 Motivation\n")
    lines.append(intro.get("motivation", "(missing)") + "\n")
    lines.append("\n### 1.2 Contributions\n")
    for c in intro.get("contributions", []):
        lines.append(f"- {c}")
    lines.append("\n\n### 1.3 Paper structure\n")
    lines.append(intro.get("paper_structure", "(missing)") + "\n")

    lines.append("\n## 2. Related work\n")
    for rw in draft.get("related_work", []):
        lines.append(f"**[{rw.get('paper_id','?')}] {rw.get('title','?')}**  ")
        lines.append(f"Relevance: {rw.get('relevance','')}  ")
        lines.append(f"Difference: {rw.get('how_we_differ','')}\n")

    method = draft.get("method", {})
    lines.append("\n## 3. Method\n")
    lines.append("### 3.1 Overview\n" + method.get("overview","(missing)") + "\n")
    lines.append("\n### 3.2 Approach\n" + method.get("approach","(missing)") + "\n")
    lines.append("\n### 3.3 Architecture / algorithm\n"
                  + method.get("architecture_or_algorithm","(missing)") + "\n")
    lines.append("\n### 3.4 Training setup\n"
                  + method.get("training_setup","(missing)") + "\n")

    exp = draft.get("experimental_setup", {})
    lines.append("\n## 4. Experimental setup\n")
    lines.append("\n### 4.1 Datasets\n")
    for d in exp.get("datasets", []):
        lines.append(f"- {d}")
    lines.append("\n\n### 4.2 Baselines\n")
    for b in exp.get("baselines", []):
        lines.append(f"- {b}")
    lines.append("\n\n### 4.3 Metrics\n")
    for m in exp.get("metrics", []):
        lines.append(f"- {m}")
    lines.append("\n\n### 4.4 Implementation notes\n")
    lines.append(exp.get("implementation_notes","(missing)") + "\n")

    if sanity:
        lines.append("\n### 4.5 Preliminary experiment (multi-agent sanity stage)\n")
        lines.append(f"- **Tier**: {sanity.get('sanity_tier')}")
        lines.append(f"- **Ran to completion**: {sanity.get('sanity_ran')}")
        lines.append(f"- **Verdict**: {sanity.get('sanity_supported')}")
        lines.append(f"- **Signal strength**: {sanity.get('sanity_signal'):.2f}" if isinstance(sanity.get('sanity_signal'), (int, float)) else f"- **Signal strength**: {sanity.get('sanity_signal')}")
        lines.append(f"- **Confound score**: {sanity.get('sanity_confound_score'):.2f}" if isinstance(sanity.get('sanity_confound_score'), (int, float)) else f"- **Confound score**: {sanity.get('sanity_confound_score')}")
        if sanity.get("sanity_effect_size") is not None:
            lines.append(f"- **Effect size**: {sanity.get('sanity_effect_size')}")
        lines.append("")
        lines.append(f"**Notes**: {sanity.get('sanity_notes','')}\n")
        tr = sanity.get("_trace") or {}
        e3 = (tr.get("e3") or {})
        run = (e3.get("sandbox") or {})
        if run.get("parsed_results"):
            lines.append("\nParsed sandbox output:")
            lines.append("```")
            for line in run["parsed_results"][:8]:
                lines.append(json.dumps(line))
            lines.append("```\n")
        code = e3.get("final_code") or ""
        if code:
            lines.append("\nSanity-stage code (truncated to first 2000 chars):")
            lines.append("```python")
            lines.append(code[:2000] + ("\n# ...truncated" if len(code) > 2000 else ""))
            lines.append("```\n")

    lines.append("\n### 4.6 Human work required\n")
    lines.append(exp.get("human_work_required","(missing)") + "\n")

    lines.append("\n## 5. Expected results\n")
    lines.append(draft.get("expected_results","(missing)") + "\n")

    disc = draft.get("discussion", {})
    lines.append("\n## 6. Discussion\n")
    lines.append("### 6.1 Limitations\n" + disc.get("limitations","(missing)") + "\n")
    lines.append("\n### 6.2 Ethical considerations\n"
                  + disc.get("ethical_considerations","(missing)") + "\n")
    lines.append("\n### 6.3 Future work\n" + disc.get("future_work","(missing)") + "\n")

    lines.append("\n## 7. Conclusion\n")
    lines.append(draft.get("conclusion","(missing)") + "\n")

    return "\n".join(lines)


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", str(s)).strip("-").lower()
    return s[:60] if len(s) > 60 else s


async def drive_one(idea: dict, out_dir: Path) -> None:
    title = str(idea.get("title", "(untitled)"))
    slug = slugify(title)
    domain = "math" if is_math(idea) else "ai"
    print(f"\n=== [{domain}] {title[:70]}")

    evidence = []
    raw = idea.get("evidence_used_json")
    if isinstance(raw, str) and raw.strip():
        try:
            evidence = json.loads(raw)
        except Exception:
            evidence = []

    prior_art: list[dict] = []
    sanity_verdict: dict | None = None

    if domain == "ai":
        print("   running multi-agent sanity stage (budget=benchmark)...")
        try:
            # run_sanity_check gates on idea["confidence"]; the flat row from
            # ideas_v2.tsv stores it under idea_confidence. Mirror it so the
            # gate doesn't short-circuit before the E1 deliberation runs.
            if not idea.get("confidence"):
                idea["confidence"] = float(idea.get("idea_confidence", 0.0) or 0.0)
            sanity_verdict = await run_sanity_check(
                idea, budget="benchmark", critique_history=None,
            )
            print(f"   sanity: ran={sanity_verdict.get('sanity_ran')} "
                  f"supported={sanity_verdict.get('sanity_supported')} "
                  f"signal={sanity_verdict.get('sanity_signal')}")
            # Persist sanity record
            (out_dir / f"{slug}_sanity.json").write_text(
                json.dumps(sanity_verdict, ensure_ascii=False, indent=2,
                           default=str),
                encoding="utf-8",
            )
            # Inject sanity into prior_art so the drafter can cite + summarise it
            tr = sanity_verdict.get("_trace") or {}
            e3 = (tr.get("e3") or {})
            run = (e3.get("sandbox") or {})
            prior_art.append({
                "paperId": "SANITY:0",
                "paper_id": "SANITY:0",
                "title": "Sanity-stage toy experiment (auto-generated)",
                "abstract": (
                    f"Tier {sanity_verdict.get('sanity_tier')} multi-agent "
                    f"sanity stage. ran_to_completion="
                    f"{run.get('ran_to_completion')}. parsed_results="
                    + json.dumps(run.get("parsed_results", [])[:6])
                    + f". Verdict: {sanity_verdict.get('sanity_supported')} "
                      f"(signal={sanity_verdict.get('sanity_signal')}, "
                      f"confound={sanity_verdict.get('sanity_confound_score')}). "
                    + f"Notes: {sanity_verdict.get('sanity_notes','')}"
                ),
            })
        except Exception as e:
            print(f"   sanity FAILED: {e}", file=sys.stderr)

    extra_prompt = (
        MATH_SYSTEM_EXTRA if domain == "math" else AI_PRELIM_EXPT_INSTRUCTION
    )
    print("   drafting paper...")
    try:
        if domain == "math":
            draft = draft_full_paper(idea, evidence=evidence, prior_art=prior_art)
            # Re-call with the math extension prompt for a richer second pass
            draft = draft_with_extra(idea, evidence, prior_art, MATH_SYSTEM_EXTRA)
        else:
            draft = draft_with_extra(idea, evidence, prior_art,
                                     AI_PRELIM_EXPT_INSTRUCTION)
    except Exception as e:
        print(f"   drafter FAILED: {e}", file=sys.stderr)
        return

    (out_dir / f"{slug}.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = render_md(idea, draft, sanity=sanity_verdict)
    (out_dir / f"{slug}.md").write_text(md, encoding="utf-8")
    print(f"   wrote {slug}.json + {slug}.md "
          f"({'+sanity' if sanity_verdict else ''})")


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ideas_path = REPO / "artifacts" / "ideas_v2.tsv"
    out_dir = REPO / "artifacts" / "paper_drafts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ideas_path, sep="\t")
    print(f"loaded {len(df)} ideas from {ideas_path}")

    # Sort: AI first (sanity stages run sequentially), then math
    df["_is_math"] = df.apply(lambda r: is_math(r.to_dict()), axis=1)
    df = df.sort_values("_is_math", kind="stable")

    for _, row in df.iterrows():
        await drive_one(row.to_dict(), out_dir)


if __name__ == "__main__":
    asyncio.run(main())
