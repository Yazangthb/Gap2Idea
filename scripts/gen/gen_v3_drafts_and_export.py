"""Draft + LaTeX-compile the v3 (analysis + CV) ideas.

  - analysis ideas → math-style deep-exposition prompt, no sandbox
  - cv ideas       → AI-style + multi-agent sanity stage with budget=benchmark

Then swap artifacts/ideas.tsv to the v3 set, rename draft caches to the
pipeline stem convention, and run gap2idea export-ideas.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gap2idea.pipeline.sanity import run_sanity_check

from gen_paper_drafts import (
    AI_PRELIM_EXPT_INSTRUCTION, MATH_SYSTEM_EXTRA,
    draft_with_extra, render_md, slugify,
)


def pipeline_stem(i: int, title: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in str(title))[:40]
    return f"{i:03d}_{safe}" if safe else f"idea_{i:03d}"


async def drive_one(idea: dict, idx: int, out_dir: Path) -> None:
    title = str(idea.get("title", "(untitled)"))
    domain = str(idea.get("domain", "")).strip().lower()
    is_cv = domain in ("cv", "ai")
    slug = slugify(title)
    pstem = pipeline_stem(idx, title)
    print(f"\n=== [{domain or 'unknown'}] {title[:70]}")

    evidence = []
    raw = idea.get("evidence_used_json")
    if isinstance(raw, str) and raw.strip():
        try:
            evidence = json.loads(raw)
        except Exception:
            evidence = []

    prior_art: list[dict] = []
    sanity_verdict: dict | None = None

    if is_cv:
        # Mirror confidence into the canonical field the gate reads.
        if not idea.get("confidence"):
            idea["confidence"] = float(idea.get("idea_confidence", 0.0) or 0.0)
        print("   running multi-agent sanity stage (budget=benchmark)...")
        try:
            sanity_verdict = await run_sanity_check(
                idea, budget="benchmark", critique_history=None,
            )
            print(f"   sanity: ran={sanity_verdict.get('sanity_ran')} "
                  f"supported={sanity_verdict.get('sanity_supported')} "
                  f"signal={sanity_verdict.get('sanity_signal')}")
            (out_dir / f"{slug}_sanity.json").write_text(
                json.dumps(sanity_verdict, ensure_ascii=False, indent=2,
                           default=str),
                encoding="utf-8",
            )
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

    print("   drafting paper...")
    try:
        extra = AI_PRELIM_EXPT_INSTRUCTION if is_cv else MATH_SYSTEM_EXTRA
        draft = draft_with_extra(idea, evidence, prior_art, extra)
    except Exception as e:
        print(f"   drafter FAILED: {e}", file=sys.stderr)
        return

    # Persist under both slug (legacy) and pipeline stem (so export-ideas finds it)
    (out_dir / f"{slug}.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / f"{pstem}.json").write_text(
        json.dumps(draft, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = render_md(idea, draft, sanity=sanity_verdict)
    (out_dir / f"{slug}.md").write_text(md, encoding="utf-8")
    print(f"   wrote {slug}.json (+ {pstem}.json for the export cache) + {slug}.md")


def swap_ideas_tsv() -> None:
    artifacts = REPO / "artifacts"
    if not (artifacts / "ideas_v2_backup.tsv").exists() and \
       (artifacts / "ideas.tsv").exists():
        shutil.copy(artifacts / "ideas.tsv", artifacts / "ideas_v2_backup.tsv")
        print(f"backed up ideas.tsv -> ideas_v2_backup.tsv")
    shutil.copy(artifacts / "ideas_v3.tsv", artifacts / "ideas.tsv")
    print(f"replaced artifacts/ideas.tsv with ideas_v3.tsv")


def run_export() -> int:
    print("\nrunning gap2idea export-ideas --format rendered-pdf --full-paper ...")
    # Use --full-paper so it picks up our cached JSON drafts.
    cmd = [
        str(REPO / ".venv" / "Scripts" / "gap2idea.exe"),
        "export-ideas",
        "--format", "rendered-pdf",
        "--full-paper",
        "--template", "standard",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    print(proc.stdout[-2000:] if proc.stdout else "")
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:], file=sys.stderr)
    return proc.returncode


def strip_control_chars_from_tex() -> None:
    """The drafter occasionally emits stray control characters (e.g. ESC 0x1B)
    that tectonic refuses. Clean every .tex sibling in artifacts/exports/."""
    exports = REPO / "artifacts" / "exports"
    if not exports.exists():
        return
    pat = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    fixed = 0
    for tex in exports.glob("0*_*.tex"):
        text = tex.read_text(encoding="utf-8", errors="replace")
        cleaned = pat.sub("", text)
        if cleaned != text:
            tex.write_text(cleaned, encoding="utf-8")
            fixed += 1
    if fixed:
        print(f"stripped control chars from {fixed} .tex files")


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ideas_path = REPO / "artifacts" / "ideas_v3.tsv"
    if not ideas_path.exists():
        raise SystemExit("ideas_v3.tsv not present yet; run gen_analysis_cv_ideas.py first")
    out_dir = REPO / "artifacts" / "paper_drafts"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ideas_path, sep="\t")
    print(f"loaded {len(df)} v3 ideas")

    # Process in TSV order so the pipeline-stem indexing matches export-ideas.
    for idx, row in df.iterrows():
        await drive_one(row.to_dict(), int(idx), out_dir)

    swap_ideas_tsv()
    rc = run_export()
    if rc != 0:
        print("export-ideas non-zero exit; attempting tex cleanup + retry")
        strip_control_chars_from_tex()
        # Manually compile any failed PDFs from cleaned .tex
        exports = REPO / "artifacts" / "exports"
        for tex in exports.glob("0*_*.tex"):
            pdf = tex.with_suffix(".pdf")
            # If the matching pdf is older than the .tex, recompile
            if not pdf.exists() or tex.stat().st_mtime > pdf.stat().st_mtime:
                tmp = Path("/tmp") / f"tec_{tex.stem}"
                tmp.mkdir(exist_ok=True)
                (tmp / "doc.tex").write_text(tex.read_text(encoding="utf-8"),
                                              encoding="utf-8")
                proc = subprocess.run(
                    ["tectonic", "doc.tex"], cwd=str(tmp),
                    capture_output=True, text=True,
                )
                if (tmp / "doc.pdf").exists():
                    shutil.copy(tmp / "doc.pdf", pdf)
                    print(f"  rebuilt {pdf.name}")
                else:
                    print(f"  STILL FAILED: {pdf.name}\n{proc.stderr[-500:]}")


if __name__ == "__main__":
    asyncio.run(main())
