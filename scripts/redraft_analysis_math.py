"""Re-draft the 3 analysis papers with the math-heavy prompt and recompile.

Reuses the patched MATH_SYSTEM_EXTRA in gen_paper_drafts.py — that prompt
now insists on real LaTeX math, theorem/lemma chains, displayed equations,
and proof-sketch estimate chains. CV drafts and sanity records are left
untouched.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from gen_paper_drafts import (
    MATH_SYSTEM_EXTRA, draft_with_extra, render_md, slugify,
)


def pipeline_stem(i: int, title: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in str(title))[:40]
    return f"{i:03d}_{safe}" if safe else f"idea_{i:03d}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ideas = pd.read_csv(REPO / "artifacts" / "ideas.tsv", sep="\t")
    out_dir = REPO / "artifacts" / "paper_drafts"
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_rows = [
        (idx, row.to_dict())
        for idx, row in ideas.iterrows()
        if str(row.get("domain", "")).lower() == "analysis"
    ]
    if not analysis_rows:
        raise SystemExit("no analysis ideas in artifacts/ideas.tsv")

    print(f"re-drafting {len(analysis_rows)} analysis ideas with math-heavy prompt...")
    for idx, idea in analysis_rows:
        title = str(idea.get("title", ""))
        slug = slugify(title)
        pstem = pipeline_stem(idx, title)
        print(f"\n[{idx}] {title[:70]}")

        evidence = []
        raw = idea.get("evidence_used_json")
        if isinstance(raw, str) and raw.strip():
            try:
                evidence = json.loads(raw)
            except Exception:
                evidence = []

        try:
            draft = draft_with_extra(idea, evidence, [], MATH_SYSTEM_EXTRA)
        except Exception as e:
            print(f"   drafter FAILED: {e}", file=sys.stderr)
            continue

        (out_dir / f"{slug}.json").write_text(
            json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (out_dir / f"{pstem}.json").write_text(
            json.dumps(draft, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        md = render_md(idea, draft, sanity=None)
        (out_dir / f"{slug}.md").write_text(md, encoding="utf-8")
        n_eq = md.count("$$") // 2
        n_inline = md.count("$") - md.count("$$") * 2
        print(f"   wrote {pstem}.json + {slug}.md  "
              f"({n_inline} inline math · {n_eq} displayed)")

    print("\nre-running gap2idea export-ideas --format rendered-pdf --full-paper ...")
    proc = subprocess.run(
        [
            str(REPO / ".venv" / "Scripts" / "gap2idea.exe"),
            "export-ideas", "--format", "rendered-pdf", "--full-paper",
            "--template", "standard",
        ],
        cwd=str(REPO), capture_output=True, text=True,
    )
    print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-1500:], file=sys.stderr)

    # Strip control chars from any .tex that ended up with embedded ESC bytes
    # and rebuild affected PDFs manually with tectonic.
    pat = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    exports = REPO / "artifacts" / "exports"
    for tex in exports.glob("0*_*.tex"):
        text = tex.read_text(encoding="utf-8", errors="replace")
        cleaned = pat.sub("", text)
        if cleaned != text:
            tex.write_text(cleaned, encoding="utf-8")
            tmp = Path("/tmp") / f"tec_{tex.stem}"
            tmp.mkdir(exist_ok=True)
            (tmp / "doc.tex").write_text(cleaned, encoding="utf-8")
            r = subprocess.run(
                ["tectonic", "doc.tex"], cwd=str(tmp),
                capture_output=True, text=True,
            )
            if (tmp / "doc.pdf").exists():
                shutil.copy(tmp / "doc.pdf", tex.with_suffix(".pdf"))
                print(f"  rebuilt {tex.stem}.pdf after control-char cleanup")


if __name__ == "__main__":
    main()
