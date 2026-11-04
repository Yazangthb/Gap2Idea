# Paper Structure

> **Owner: Hamza** · Status: first draft (needs your review) · Feeds into: `main.tex` layout + Introduction roadmap paragraph
> Paper: *Scalable Extraction of Author-Stated Research Gaps in Software Engineering: Toward Automated Idea Generation*

Mapped to the course template (`firas-jolha/em.dcs`). Keep the overall structure recognizable — the
README says that makes grading and peer review easier.

| Template file | Section | Contents |
|---|---|---|
| `sections/01_introduction.tex` | **1. Introduction** | Motivation, research problem, RQs, contributions, roadmap. **← draft written** ([01_introduction.tex](sections/01_introduction.tex)) |
| `sections/02_related_work.tex` | **2. Related Work** | Future-work-sentence classification; limitation detection & generation; topic modeling of gaps; rhetorical/discourse zoning; how we differ (SE domain, scale, statistical rigor). |
| `sections/03_data_and_methods.tex` | **3. Data & Methods** | Corpus construction (venues, years, retrieval, full-text parsing); annotation protocol & gold set; extraction methods (baselines + LLM); evaluation metrics & statistical tests. |
| `sections/04_experimental_design.tex` | **4. Experimental Design** | Hypotheses (H0/H1 per RQ — see [research-questions.md](research-questions.md)); data splits; comparison protocol; significance-testing plan; human-evaluation design; threats to design validity. |
| `sections/05_results.tex` | **5. Results** | RQ1 extraction quality; RQ2 method comparison + significance; RQ3 distributions & trends; (RQ4) clustering. Tables + **TikZ/pgfplots** figures (required by template). |
| `sections/06_discussion.tex` | **6. Discussion** | Interpretation, implications for automated idea generation, limitations, threats to validity. |
| `sections/07_conclusion.tex` | **7. Conclusion** | Summary + future work (link to the forecasting / idea-generation thesis thrust). |
| `references.bib` | **References** | natbib / `plainnat`, cited with `\citep{}` / `\citet{}`. **← seeded** ([references.bib](references.bib)) |

---

## Your job on this file
- Sanity-check the section order against the template's `main.tex` once we clone it (coding phase) and
  confirm each `\input{}` is wired up.
- Write the **roadmap paragraph** at the end of the Introduction so it names these sections with
  `\Cref{}` (a draft roadmap paragraph already exists at the bottom of
  [01_introduction.tex](sections/01_introduction.tex) — align it with any section renames).
- Decide whether **Related Work** is a standalone section (as here) or folded into the Introduction —
  standalone is safer for this template.
