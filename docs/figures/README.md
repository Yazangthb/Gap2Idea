# Thesis figures

Mermaid sources for the diagrams used in the thesis. Each `.mmd` is a
plain-text spec — small enough to track in git, and renders inline on GitHub
when you open the file.

| File | Purpose | Where in the thesis |
|---|---|---|
| `pipeline.mmd` | End-to-end data flow with the four idea-generation modes | Methodology — overview |
| `bridge_score.mmd` | Composition of the `bridge_score` formula | Methodology — pair selection |
| `orchestrated_mode.mmd` | Multi-agent loop: Synthesiser → Critic → Reviser → JudgePanel | Methodology — orchestrated mode |
| `bench_methodology.mmd` | Pipeline-vs-gold benchmark against unarXive | Evaluation — extraction quality |

## Rendering

Three options, in increasing fidelity:

1. **GitHub** — open the `.mmd` file in the web UI; Mermaid is rendered inline.
2. **mermaid.live** — paste the contents; export PNG/SVG/PDF from the toolbar.
3. **Local mermaid-cli** (needs Node):
   ```
   npm install -g @mermaid-js/mermaid-cli
   mmdc -i pipeline.mmd -o pipeline.pdf -t neutral -b transparent
   ```
   Repeat per file. For LaTeX inclusion, prefer `-o pipeline.pdf` (vector).

For final thesis-quality figures, the bridge-score one (`bridge_score.mmd`)
benefits most from a TikZ rewrite — the triangular peak function reads
better as a tiny inset plot than as a flowchart node label. Until that's
authored, the Mermaid version is the placeholder.
