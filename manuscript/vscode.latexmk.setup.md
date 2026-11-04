# Setting Up LaTeX + `latexmk` in VS Code

This guide walks you through installing a LaTeX distribution, `latexmk`,
and the VS Code extension needed to compile `main.tex` (and get features
like auto-build-on-save, SyncTeX PDF preview, and error highlighting).

## 1. Install a LaTeX distribution

`latexmk` is a build tool — it needs an underlying LaTeX distribution
(`pdflatex`, `bibtex`/`biber`, etc.) to actually compile the document.

- **Windows:** Install [MiKTeX](https://miktex.org/download) (recommended —
  installs packages on the fly) or [TeX Live](https://tug.org/texlive/).
- **macOS:** Install [MacTeX](https://tug.org/mactex/) (full TeX Live
  distribution for macOS).
- **Linux:** Install TeX Live via your package manager, e.g.:
  ```bash
  sudo apt install texlive-full   # Debian/Ubuntu
  sudo dnf install texlive-scheme-full   # Fedora
  ```
  (`texlive-full`/`texlive-scheme-full` are large but avoid missing-package
  errors later — a smaller scheme works too if you know which packages
  you'll need.)

`latexmk` ships with both MiKTeX and TeX Live, so a separate install step
is normally not required. Verify it's on your `PATH`:

```bash
latexmk --version
```

If the command isn't found, restart your terminal (and VS Code) after
installing the distribution, or add its `bin` directory to your system
`PATH` manually.

## 2. Install the LaTeX Workshop extension

1. Open VS Code.
2. Go to the Extensions view (`Ctrl+Shift+X` / `Cmd+Shift+X`).
3. Search for **LaTeX Workshop** (by James Yu) and click **Install**.

This extension integrates `latexmk`, PDF preview, SyncTeX (click-to-jump
between source and PDF), and error/warning highlighting directly into
VS Code.

## 3. Configure `latexmk` as the build tool

LaTeX Workshop uses `latexmk` by default, but this project also uses
`natbib`/BibTeX for citations, so it's worth confirming the recipe runs
`bibtex` (or `biber`) between LaTeX passes.

Open your VS Code **settings.json** (`Ctrl+Shift+P` → "Preferences: Open
User Settings (JSON)", or use a workspace-local `.vscode/settings.json` to
scope it to this repo) and add:

```json
{
  "latex-workshop.latex.tools": [
    {
      "name": "latexmk",
      "command": "latexmk",
      "args": [
        "-synctex=1",
        "-interaction=nonstopmode",
        "-file-line-error",
        "-pdf",
        "-outdir=%OUTDIR%",
        "%DOC%"
      ]
    }
  ],
  "latex-workshop.latex.recipes": [
    {
      "name": "latexmk",
      "tools": ["latexmk"]
    }
  ],
  "latex-workshop.latex.recipe.default": "latexmk",
  "latex-workshop.latex.autoBuild.run": "onSave",
  "latex-workshop.view.pdf.viewer": "tab"
}
```

`latexmk` automatically detects that `references.bib` is cited (via the
`natbib` package) and reruns `bibtex` and `pdflatex` as many times as
needed to resolve citations and cross-references — you don't need a
separate recipe step for it.

## 4. Build the project

1. Open this repository's folder in VS Code (`File → Open Folder...`).
2. Open `main.tex`.
3. Build it either:
   - Automatically on save (with `autoBuild.run: "onSave"` set above), or
   - Manually via the green "▶" **Build LaTeX project** button in the top
     right of the editor, or `Ctrl+Alt+B`.
4. View the compiled PDF with the "View LaTeX PDF" icon next to the build
   button, or `Ctrl+Alt+V` — it opens in a side tab with SyncTeX enabled
   (click text in the PDF to jump to the corresponding source line, and
   vice versa).

## 5. Troubleshooting

- **"latexmk: command not found"** — the LaTeX distribution's `bin`
  directory isn't on your `PATH`. Reinstall/repair the distribution, or add
  it to `PATH` manually, then restart VS Code.
- **Missing package errors** — with MiKTeX, allow it to install packages
  on the fly when prompted. With TeX Live, install the missing package via
  `tlmgr install <package-name>`, or reinstall using the `full`/`scheme-full`
  option to avoid this entirely.
- **Citations show as "`?`" in the PDF** — this usually means `bibtex`
  hasn't run yet, or `references.bib` has a syntax error. Delete the
  auxiliary files (`main.aux`, `main.bbl`, `main.blg`) and rebuild; check
  the `.blg` log for BibTeX-specific errors if it still fails.
- **Build succeeds but the PDF doesn't update** — check the "LaTeX
  Compiler" output panel (`View → Output`, then select "LaTeX Compiler"
  from the dropdown) for the actual `latexmk`/`pdflatex` log.
- **Stale auxiliary files causing weird errors** — run the "Clean up auxiliary
  files" command from the LaTeX Workshop sidebar (trash-can icon), or
  delete `.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc` files manually, then
  rebuild from scratch.