"""Idea export — LaTeX templates, server-side LaTeX→PDF compilation, and
library-summary PDFs.

Three outputs:

  • `idea_to_latex(idea, template=...)`  -> `.tex` source. Choose one of the
        bundled templates (`minimal` · `standard` · `ieee`) or supply a custom
        Jinja2 template string via `template_source`.

  • `render_idea_to_pdf(idea, ...)`      -> rendered PDF bytes (compiled by
        `tectonic` or `pdflatex` if either is on PATH). Falls back to raising
        a clear error so the caller can offer the .tex download instead.

  • `ideas_to_pdf_bytes(ideas_df)`       -> a multi-page library-summary PDF
        rendered with reportlab (pure Python, no system deps).

Best practices:
  • Jinja2 environment with custom `escape_tex` filter (handles every LaTeX
    special character). `ChoiceLoader` over `FileSystemLoader(bundled)` +
    `DictLoader(user_uploads)` so user-uploaded templates take precedence.
  • LaTeX compilation runs in a `TemporaryDirectory`, with `nonstopmode` and
    a 60 s timeout so a runaway source can't hang the UI.
  • Variables passed to user-supplied templates are documented in
    `TEMPLATE_VARIABLES` so authors can write their own templates safely.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import pandas as pd
from jinja2 import (
    ChoiceLoader,
    DictLoader,
    Environment,
    FileSystemLoader,
    select_autoescape,
)

from gap2idea.utils import get_logger

log = get_logger(__name__)


# ===========================================================================
# Jinja environment + template registry
# ===========================================================================

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"

# Mapping of friendly template name -> filename in templates/
BUNDLED_TEMPLATES: dict[str, str] = {
    "minimal":  "idea_paper_minimal.tex.j2",
    "standard": "idea_paper_standard.tex.j2",
    "ieee":     "idea_paper_ieee.tex.j2",
}
DEFAULT_TEMPLATE = "standard"

# For documentation in the CLI / Streamlit UI — these are the variables a
# custom Jinja template can use, with their Python types.
TEMPLATE_VARIABLES: dict[str, str] = {
    "title":                 "str",
    "label_a":               "str — primary theme label",
    "label_b":               "str — second theme (bridge mode only, else empty)",
    "mode":                  "str — bridge | within | method-gap | orchestrated",
    "research_question":     "str",
    "method_sketch":         "str",
    "evaluation_plan":       "str",
    "expected_contribution": "str",
    "assumptions_and_risks": "str",
    "evidence_used":         "list[{paper_id: str, gap_sentence: str}]",
    "novelty_score":         "float | None — 1 - max_cosine(idea, S2_hit)",
    "closest_paper_title":   "str",
    "closest_paper_year":    "str | int",
    "date":                  "str — YYYY-MM-DD UTC",
    "panel_consensus":       "dict | None — {composite, novelty, specificity, "
                             "feasibility, evidence_grounding, agreement, n_judges}",
    "full_paper":            "dict | None — full paper draft produced by "
                             "paper_drafter.draft_full_paper. Has keys: abstract, "
                             "introduction{motivation,contributions,paper_structure}, "
                             "related_work[{paper_id,title,relevance,how_we_differ}], "
                             "method{overview,approach,architecture_or_algorithm,training_setup}, "
                             "experimental_setup{datasets,baselines,metrics,implementation_notes,"
                             "human_work_required}, expected_results, "
                             "discussion{limitations,ethical_considerations,future_work}, conclusion.",
    "(filter) escape_tex":   "callable — escape LaTeX special chars in any value",
}

# LaTeX special-character escape table.
_TEX_ESCAPE = {
    "&":  r"\&",
    "%":  r"\%",
    "$":  r"\$",
    "#":  r"\#",
    "_":  r"\_",
    "{":  r"\{",
    "}":  r"\}",
    "~":  r"\textasciitilde{}",
    "^":  r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}
_TEX_ESCAPE_RE = re.compile("|".join(re.escape(c) for c in _TEX_ESCAPE))


def escape_tex(value: object) -> str:
    """Escape LaTeX special characters in a string.

    Robust against None and non-string inputs (coerces via str()).
    """
    if value is None:
        return ""
    s = str(value)
    return _TEX_ESCAPE_RE.sub(lambda m: _TEX_ESCAPE[m.group(0)], s)


def _build_env(extra_templates: dict[str, str] | None = None) -> Environment:
    """Construct a Jinja env. `extra_templates` is an optional in-memory
    mapping of `template_name -> source` (used for user uploads); these
    take precedence over bundled files.
    """
    loaders = []
    if extra_templates:
        loaders.append(DictLoader(extra_templates))
    loaders.append(FileSystemLoader(str(_TEMPLATES_DIR)))
    env = Environment(
        loader=ChoiceLoader(loaders),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["escape_tex"] = escape_tex
    return env


def list_templates() -> list[str]:
    """Return the names of bundled templates available to `idea_to_latex`."""
    return list(BUNDLED_TEMPLATES.keys())


# ===========================================================================
# LaTeX
# ===========================================================================

def _idea_render_context(
    idea: dict,
    panel_consensus: dict | None,
    full_paper: dict | None = None,
) -> dict:
    """Build the variable dict passed to any LaTeX template."""
    evidence_used: list[dict] = []
    raw = idea.get("evidence_used_json") or idea.get("evidence_used")
    if isinstance(raw, str) and raw.strip():
        try:
            evidence_used = json.loads(raw)
        except json.JSONDecodeError:
            evidence_used = []
    elif isinstance(raw, list):
        evidence_used = raw

    novelty_score = idea.get("novelty_score")
    try:
        novelty_score = float(novelty_score) if novelty_score is not None and not pd.isna(novelty_score) else None
    except (TypeError, ValueError):
        novelty_score = None

    return dict(
        title=idea.get("title") or "(untitled idea)",
        label_a=idea.get("label_a", "") or "",
        label_b=idea.get("label_b", "") or "",
        mode=idea.get("mode", "—") or "—",
        research_question=idea.get("research_question", "") or "",
        method_sketch=idea.get("method_sketch", "") or "",
        evaluation_plan=idea.get("evaluation_plan", "") or "",
        expected_contribution=idea.get("expected_contribution", "") or "",
        assumptions_and_risks=idea.get("assumptions_and_risks", "") or "",
        evidence_used=evidence_used,
        novelty_score=novelty_score,
        closest_paper_title=idea.get("closest_paper_title") or "",
        closest_paper_year=idea.get("closest_paper_year") or "",
        date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        panel_consensus=panel_consensus,
        full_paper=full_paper,
    )


def idea_to_latex(
    idea: dict,
    *,
    template: str = DEFAULT_TEMPLATE,
    template_source: str | None = None,
    panel_consensus: dict | None = None,
    full_paper: dict | None = None,
) -> str:
    """Render one idea into LaTeX.

    Args:
      template:        name of a bundled template (see `list_templates()`).
                       Ignored if `template_source` is given.
      template_source: raw Jinja template string. Wins over `template`.
                       Useful for user-uploaded templates.
      panel_consensus: optional dict from multi-judge eval; templates may
                       render an "automated review scores" block when present.
      full_paper:      optional dict from `paper_drafter.draft_full_paper`.
                       When present, templates render a full paper draft
                       (abstract, related work, expanded method/experiments,
                       discussion, conclusion) instead of the skeleton view.

    The template receives the variables documented in `TEMPLATE_VARIABLES`
    plus the `escape_tex` filter.
    """
    ctx = _idea_render_context(idea, panel_consensus, full_paper=full_paper)

    if template_source is not None:
        env = _build_env(extra_templates={"__custom__": template_source})
        tmpl = env.get_template("__custom__")
    else:
        if template not in BUNDLED_TEMPLATES:
            raise ValueError(
                f"Unknown template '{template}'. Available: {list(BUNDLED_TEMPLATES)}"
            )
        env = _build_env()
        tmpl = env.get_template(BUNDLED_TEMPLATES[template])

    return tmpl.render(**ctx)


def write_idea_latex(
    idea: dict,
    out_path: Path,
    *,
    template: str = DEFAULT_TEMPLATE,
    template_source: str | None = None,
    panel_consensus: dict | None = None,
    full_paper: dict | None = None,
) -> Path:
    """Convenience: render and write a .tex file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        idea_to_latex(idea, template=template, template_source=template_source,
                       panel_consensus=panel_consensus, full_paper=full_paper),
        encoding="utf-8",
    )
    return out_path


# ===========================================================================
# LaTeX -> PDF compilation (optional, requires tectonic or pdflatex on PATH)
# ===========================================================================

class LatexCompilerNotFound(RuntimeError):
    """Raised when neither tectonic nor pdflatex is installed."""


class LatexCompilationError(RuntimeError):
    """Raised when the LaTeX compiler ran but produced no PDF or errored."""


def find_latex_compiler() -> str | None:
    """Return the command of the first available LaTeX compiler, or None.

    Search order: tectonic (single-binary, auto-fetches packages) → pdflatex.
    """
    for cmd in ("tectonic", "pdflatex"):
        if shutil.which(cmd):
            return cmd
    return None


def compile_latex_to_pdf(tex_source: str, *, timeout_s: int = 60) -> bytes:
    """Compile a LaTeX source string to PDF bytes.

    Raises:
      LatexCompilerNotFound  — no compiler on PATH
      LatexCompilationError  — compiler ran but failed (log excerpt included)
    """
    compiler = find_latex_compiler()
    if not compiler:
        raise LatexCompilerNotFound(
            "No LaTeX compiler found on PATH. Install one to enable rendered "
            "PDF export:\n"
            "  • tectonic (recommended, single binary):  https://tectonic-typesetting.github.io/\n"
            "  • TeX Live (full suite, includes pdflatex): https://tug.org/texlive/"
        )

    with tempfile.TemporaryDirectory(prefix="gap2idea_tex_") as tmp_name:
        tmp = Path(tmp_name)
        tex_path = tmp / "doc.tex"
        tex_path.write_text(tex_source, encoding="utf-8")

        if compiler == "tectonic":
            cmd = ["tectonic", "--keep-logs", "--outdir", str(tmp), str(tex_path)]
        else:
            cmd = [
                "pdflatex",
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={tmp}",
                str(tex_path),
            ]
        log.info("Compiling LaTeX with %s ...", compiler)
        try:
            result = subprocess.run(
                cmd, capture_output=True, timeout=timeout_s, cwd=str(tmp),
            )
        except subprocess.TimeoutExpired as e:
            raise LatexCompilationError(
                f"{compiler} timed out after {timeout_s}s — likely an infinite loop or huge document"
            ) from e

        pdf_path = tmp / "doc.pdf"
        if result.returncode != 0 or not pdf_path.exists():
            log_excerpt = (result.stdout + result.stderr).decode("utf-8", errors="replace")[-1500:]
            raise LatexCompilationError(
                f"{compiler} returned {result.returncode} and did not produce a PDF.\n"
                f"Last 1500 chars of output:\n\n{log_excerpt}"
            )
        return pdf_path.read_bytes()


def render_idea_to_pdf(
    idea: dict,
    *,
    template: str = DEFAULT_TEMPLATE,
    template_source: str | None = None,
    panel_consensus: dict | None = None,
    full_paper: dict | None = None,
    timeout_s: int = 60,
) -> bytes:
    """Convenience: render LaTeX + compile to PDF in one call."""
    tex = idea_to_latex(
        idea, template=template, template_source=template_source,
        panel_consensus=panel_consensus, full_paper=full_paper,
    )
    return compile_latex_to_pdf(tex, timeout_s=timeout_s)


def draft_and_render_paper(
    idea: dict,
    *,
    template: str = DEFAULT_TEMPLATE,
    template_source: str | None = None,
    panel_consensus: dict | None = None,
    evidence: list[dict] | None = None,
    prior_art: list[dict] | None = None,
    drafter_model: str | None = None,
) -> tuple[str, dict]:
    """One-call helper: draft a full paper for `idea` (via LLM), then render
    LaTeX. Returns (tex_source, full_paper_dict) so callers can cache the
    expensive LLM output to disk and re-render fast.
    """
    from gap2idea.pipeline.paper_drafter import (
        DEFAULT_DRAFT_MODEL,
        draft_full_paper,
    )
    full_paper = draft_full_paper(
        idea, evidence=evidence, prior_art=prior_art,
        model=drafter_model or DEFAULT_DRAFT_MODEL,
    )
    tex = idea_to_latex(
        idea, template=template, template_source=template_source,
        panel_consensus=panel_consensus, full_paper=full_paper,
    )
    return tex, full_paper


# ===========================================================================
# PDF library summary (reportlab)
# ===========================================================================

def _safe(x, default: str = "—") -> str:
    if x is None:
        return default
    try:
        if pd.isna(x):
            return default
    except (TypeError, ValueError):
        pass
    s = str(x).strip()
    return s or default


def ideas_to_pdf_bytes(ideas_df: pd.DataFrame, title: str = "Gap2Idea — Idea Library") -> bytes:
    """Render a multi-page PDF summary of every idea. Returns bytes.

    Uses ReportLab's Platypus (flowables) — pure Python, no system deps.
    """
    # Lazy import: keeps reportlab off the import path for users who don't export
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title=title, author="Gap2Idea",
    )

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, spaceAfter=8, textColor=colors.HexColor("#1e293b"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#0f172a"))
    h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11, spaceBefore=8, spaceAfter=2, textColor=colors.HexColor("#475569"))
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, textColor=colors.HexColor("#0f172a"))
    label = ParagraphStyle("Label", parent=styles["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#64748b"), spaceAfter=2)
    cite = ParagraphStyle("Cite", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.HexColor("#334155"), leftIndent=8)

    story = []
    story.append(Paragraph(title, h1))
    story.append(Paragraph(
        f"{len(ideas_df)} idea(s) · generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        label,
    ))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceBefore=4, spaceAfter=8))

    # Top summary table (only fields that exist)
    summary_cols = []
    for c in ["mode", "novelty_score", "idea_confidence", "panel_composite", "panel_agreement"]:
        if c in ideas_df.columns:
            summary_cols.append(c)
    if summary_cols:
        agg = {}
        for c in summary_cols:
            s = pd.to_numeric(ideas_df[c], errors="coerce") if c != "mode" else None
            if s is not None and s.notna().any():
                agg[c] = f"{s.mean():.2f}"
        if "mode" in ideas_df.columns:
            counts = ideas_df["mode"].value_counts().to_dict()
            agg["mode breakdown"] = ", ".join(f"{k}: {v}" for k, v in counts.items())
        if agg:
            rows = [[Paragraph("<b>Metric</b>", label), Paragraph("<b>Value</b>", label)]]
            rows += [[Paragraph(k.replace("_", " "), body), Paragraph(_safe(v), body)] for k, v in agg.items()]
            t = Table(rows, colWidths=[5 * cm, 11 * cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(t)
    story.append(Spacer(1, 0.4 * cm))

    # Per-idea card
    for i, (_, r) in enumerate(ideas_df.iterrows(), 1):
        story.append(Paragraph(f"{i}.  {_safe(r.get('title'))}", h2))

        bits = []
        if _safe(r.get("mode"), "") != "":
            bits.append(f"<b>Mode</b> {r.get('mode')}")
        if r.get("label_a"):
            theme_str = _safe(r.get("label_a"))
            if r.get("label_b"):
                theme_str += f" ✕ {_safe(r.get('label_b'))}"
            bits.append(f"<b>Themes</b> {theme_str}")
        if "idea_confidence" in r and pd.notna(r.get("idea_confidence")):
            try:
                bits.append(f"<b>Confidence</b> {float(r['idea_confidence']):.0%}")
            except (ValueError, TypeError):
                pass
        if "novelty_score" in r and pd.notna(r.get("novelty_score")):
            try:
                bits.append(f"<b>Novelty</b> {float(r['novelty_score']):.0%}")
            except (ValueError, TypeError):
                pass
        if "panel_composite" in r and pd.notna(r.get("panel_composite")):
            try:
                bits.append(f"<b>Judge panel</b> {float(r['panel_composite']):.2f}/5")
            except (ValueError, TypeError):
                pass
        if bits:
            story.append(Paragraph(" &nbsp;·&nbsp; ".join(bits), label))

        sections = [
            ("Research question", r.get("research_question")),
            ("Method sketch",      r.get("method_sketch")),
            ("Evaluation plan",    r.get("evaluation_plan")),
            ("Expected contribution", r.get("expected_contribution")),
            ("Assumptions and risks", r.get("assumptions_and_risks")),
        ]
        for hdr, val in sections:
            v = _safe(val, "")
            if not v:
                continue
            story.append(Paragraph(hdr, h3))
            story.append(Paragraph(v, body))

        # Evidence citations
        raw = r.get("evidence_used_json")
        if isinstance(raw, str) and raw.strip():
            try:
                ev = json.loads(raw)
                if ev:
                    story.append(Paragraph("Evidence used", h3))
                    for e in ev:
                        story.append(Paragraph(
                            f"({_safe(e.get('paper_id'))}) — {_safe(e.get('gap_sentence'))}",
                            cite,
                        ))
            except json.JSONDecodeError:
                pass

        # Closest prior work
        if r.get("closest_paper_title"):
            story.append(Paragraph(
                f"<i>Closest prior work:</i> {_safe(r.get('closest_paper_title'))} "
                f"({_safe(r.get('closest_paper_year'))})",
                label,
            ))

        story.append(HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#e2e8f0"), spaceBefore=8, spaceAfter=6))

    # ---- Render ----
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawCentredString(
            A4[0] / 2, 1.2 * cm,
            f"Page {doc.page}   ·   Generated by Gap2Idea   ·   {datetime.now(timezone.utc).strftime('%Y-%m-%d')}",
        )
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def write_ideas_pdf(ideas_df: pd.DataFrame, out_path: Path, title: str = "Gap2Idea — Idea Library") -> Path:
    """Convenience: render and write the library PDF to disk."""
    data = ideas_to_pdf_bytes(ideas_df, title=title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    log.info("Wrote %d-byte PDF to %s", len(data), out_path)
    return out_path
