# -*- coding: utf-8 -*-
"""Build the files MDPI asks for at submission, from the current manuscript_mdpi/ folder.

Writes manuscript_mdpi/submission_package/ (not committed):
  manuscript.pdf                    the compiled paper
  source.zip                        LaTeX sources needed to compile it (main.tex, sections, tables, figures,
                                    Definitions, references.bib, main.bbl)
  figures.zip                       every figure as a PNG at 600 dpi, named Figure1.png, Figure2.png, ...
                                    (MDPI: "preferably no less than 600 dpi", PNG, JPEG or TIFF, one zip archive)

Needs latexmk/pdflatex on the PATH and the Python package pypdfium2.

  python scripts/publication/make_submission_package.py
"""
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "manuscript_mdpi"
OUT = SRC / "submission_package"
DPI = 600
STANDALONE = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=18.2cm,paperheight=10.1cm,margin=0.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newpxtext,newpxmath}
\usepackage{tikz,xcolor}
\usetikzlibrary{positioning, arrows.meta, shapes.geometric}
\pagestyle{empty}
\begin{document}
\noindent\input{overview.tex}
\end{document}
"""


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed in {cwd}:\n{r.stdout[-2000:]}")


def figures_in_order():
    """Figure files in the order they appear in the paper."""
    order = []
    for line in (SRC / "main.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\input\{(sections/[^}]+)\}", line.strip())
        if not m:
            continue
        text = (SRC / (m.group(1) + ".tex")).read_text(encoding="utf-8")
        for f in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}|\\input\{(figures/[^}]+)\}", text):
            order.append(f[0] or f[1])
    return order


def to_png(pdf_path, png_path):
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        doc[0].render(scale=DPI / 72).to_pil().save(png_path, dpi=(DPI, DPI))
    finally:
        doc.close()  # otherwise Windows cannot delete the temporary folder


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work = Path(tmp) / "paper"
        shutil.copytree(SRC, work, ignore=shutil.ignore_patterns("submission_package", "build", "*.aux", "*.log", "*.out",
                                                                 "*.fls", "*.fdb_latexmk", "*.synctex.gz", "*.blg"))
        run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], work)
        shutil.copy(work / "main.pdf", OUT / "manuscript.pdf")

        figs = figures_in_order()
        with zipfile.ZipFile(OUT / "figures.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for k, fig in enumerate(figs, 1):
                src = SRC / fig
                if src.suffix == ".tex":  # the TikZ overview: compile it on its own page first
                    fdir = Path(tmp) / f"fig{k}"
                    fdir.mkdir()
                    shutil.copy(src, fdir / "overview.tex")
                    (fdir / "fig.tex").write_text(STANDALONE, encoding="utf-8")
                    run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"], fdir)
                    pdf = fdir / "fig.pdf"
                else:
                    pdf = src
                png = Path(tmp) / f"Figure{k}.png"
                to_png(pdf, png)
                z.write(png, png.name)
                print(f"Figure{k}.png  <-  {fig}")

        # The human-study section is shipped only once the results section actually inputs it.
        results = (work / "sections/04_results.tex").read_text(encoding="utf-8").splitlines()
        human_study_used = any("\\input{sections/04b_human_study}" in line and not line.lstrip().startswith("%")
                               for line in results)
        with zipfile.ZipFile(OUT / "source.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for name in ("main.tex", "main.bbl", "references.bib"):
                z.write(work / name, name)
            for folder in ("sections", "tables", "Definitions"):
                for p in sorted((work / folder).rglob("*")):
                    if not p.is_file() or (p.name == "04b_human_study.tex" and not human_study_used):
                        continue
                    z.write(p, p.relative_to(work).as_posix())
            for fig in figs:
                z.write(work / fig, fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
