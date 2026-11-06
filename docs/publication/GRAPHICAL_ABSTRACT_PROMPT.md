# Gap2Idea graphical abstract

Generated using the built-in image-generation tool. This is a separate visual
draft; the manuscript's existing Figure 1 is preserved. The illustration uses
no experiment counts, so its conceptual flow is independent of pending runs.

Scientific review note: the three Stage C modes are alternatives. Stage A is
dataset-specific and bypassed for FWS. Human labels cover SAL limitations and
FWS future work; the three-type LLM-labelled corpus remains provisional.
The source links in the output panel describe provenance, not web retrieval.

## Generation prompt

Create a publication-quality graphical abstract for an academic NLP research paper, Gap2Idea. Wide landscape composition, approximately 3:2 or 16:10, crisp flat vector-like scientific infographic on pure white. Restrained navy typography, muted teal, blue, violet, warm amber accents. Large readable text, generous whitespace, sharp thin connectors and understated rounded panels. No decorative robot brains, no gradients, no 3D, no fake charts or numerical results. This is a conceptual overview, not an empirical-results chart. Exact scientific content and unambiguous arrows are essential.

Title: "Gap2Idea"
Subtitle: "Extracting author-stated research gaps from scientific papers"

Organize into a prominent central left-to-right workflow and a separate lower evaluation band. Five clearly separated columns in the main workflow, with a continuous arrow between each adjacent pair. Each panel may use a small simple meaningful line icon.

MAIN WORKFLOW:
1. "Paper texts"
   A small stack of scholarly documents. Three concise source tags: "SAL: RCT reports", "FWS: conclusion sections", "arXiv papers".
   A small internal step below the sources: "Prepare sentence streams".
2. "A · Localise"
   Simple narrowing funnel icon. Text: "Structural rules" and "Remove irrelevant text".
   Small footer: "Dataset-specific; bypassed for FWS".
3. "B · Classify"
   Text: "Zero-shot NLI" and "Score gap-type hypotheses".
   Show a few sentence strips, some selected.
   Small footer: "Threshold set on development data".
4. "C · Filter"
   Text: "LLM + neighbouring sentences".
   Clearly indicate three ALTERNATIVE settings, not sequential operations, as three small pills stacked or aligned under ONE selector heading "Choose a mode": "High precision", "Balanced", "High recall".
   Decision text: "Keep / drop".
5. "Extracted gaps"
   Three colored small cards: "Limitations", "Future work", "Research gaps".
   Footer: "Original sentences + source links".
   Make no implication that all three types have human validation in every benchmark.

Below the workflow, place a restrained single-line principle spanning it: "Select existing sentences · Preserve wording and provenance".
No generated research ideas in this image: this study ends at extraction.

LOWER EVALUATION BAND: visually separate from inference flow, with the heading "Evaluate the pipeline". A thin connector from extracted output down to this band, no arrows back into test predictions. Three aligned compact panels:
- "Reference labels" with two lines: "Human labels: SAL / FWS" and "LLM labels: three-type corpus (provisional)".
- "Research questions" with three short lines: "RQ1  Stage contribution", "RQ2  Precision–recall control", "RQ3  Classifier choice and cost".
- "Measures" with two short lines: "Precision · Recall · F1" and "Paper-level uncertainty · LLM input cost".

Keep all text legible at paper-figure size. Correct English spelling. Avoid tiny microtext. Do not add datasets, performance numbers, novelty claims, generation stages, or promises of validated research ideas. Keep the diagram editorially refined, clear enough to understand without the manuscript.


## Header-free manuscript version

The image was edited with the built-in image-generation tool to remove only
the top Gap2Idea title and descriptive subtitle and trim their unused space.
Diagram headings, labels and evaluation content are retained. The edited
asset is `manuscript_mdpi/figures/gap2idea_graphical_abstract_v2.png`.
It is included near the start of Materials and Methods as an overview,
with LaTeX label `fig:pipeline-overview`. The original quantitative funnel
diagram and its `fig:funnel` label are retained; numbering is automatic.
