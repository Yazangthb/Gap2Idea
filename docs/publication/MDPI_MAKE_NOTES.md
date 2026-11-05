# MDPI MAKE: submission notes

Target journal: **Machine Learning and Knowledge Extraction (MAKE), MDPI**.
Source: https://www.mdpi.com/journal/make/instructions (read 2026-11-04). LaTeX template: https://www.mdpi.com/authors/latex (also on Overleaf as "MDPI Article Template").
Draft source being converted: branch `origin/paper-writing`, folder `manuscript/` (generic `article` class, natbib).

Use this file as the checklist while porting the paper section by section.

## 0. Decisions log

| Date | Item | Decision |
|---|---|---|
| 2026-11-04 | Working folder | `manuscript_mdpi/` (official MDPI ACS template v20260911, option `make`); builds with `pdflatex main.tex` |
| 2026-11-04 | Title | **Scalable Extraction of Author-Stated Research Gaps: Towards Automated Idea Generation** ("Gap2Idea" is reserved for the follow-up idea-generation paper; use "Towards" consistently) |
| 2026-11-04 | Authors | Yazan Alnakri, Osama Orabi, Hamza Aldaghstany (single Innopolis affiliation). Supervisor **Rustam** to be added later (last author, likely corresponding). Firas Jolha and Giancarlo Succi are **not** authors. |
| 2026-11-04 | Corresponding author | Yazan for now (placeholder until supervisor added) |
| 2026-11-04 | Terminology | Stage B = **sentence classifier** (covers frozen/fine-tuned encoders and zero-shot NLI cross-encoders); Stage C = **LLM (precision) filter**, not "verifier". The draft says "verifier/verification" throughout, so rename it in every section as we port. Sci-Sentence task = "research-gap sentence classification" (not "auxiliary literature-gap task"). |
| 2026-11-04 | Scope | **Include** the gold-dataset experiments (230-paper LLM-ensemble gold, funnel v2, Stage-B encoder sweep, NLI cross-encoders, Stage A section-recovery negative result) alongside SAL-40. Report gold numbers as provisional until human validation is done. |
| open | ModernBERT-large-nli and fine-tuned bge-small (Oct 27, V100): aggregate JSONs recovered to `docs/experiments/stage_b_v100_oct27_results.json`. Originals and any predictions are still on the box (`~/stageb/results_v2/`). Full picture: `docs/publication/PROJECT_INVENTORY.md`. | TODO |
| open | ORCIDs, exact affiliation address/department, Hamza's surname capitalisation | TODO |

Template gotchas found: this version has **no `\AuthorCitation`** command (remove it if copying from older templates).

---

## 1. Article type and section order

Type: **Article** (original research).

| # | MDPI section | Our draft file | Notes |
|---|---|---|---|
| - | Title, Authors, Affiliations | `main.tex` | see §2 |
| - | Abstract (≤ ~200 words), Keywords (3–10) | `main.tex` | draft abstract is about 260 words, so **cut it** |
| 1 | Introduction | `01_introduction.tex` | must state purpose, significance, hypotheses/RQs, main conclusions |
| 2 | Related Work (optional own section) | `02_related_works.tex` | allowed; MDPI expects the field to be reviewed in the Intro anyway |
| 3 | **Materials and Methods** | `03_methodology.tex` | rename; software names + versions; code availability; **GenAI disclosure** |
| 4 | Results | `04_results.tex` | |
| 5 | Discussion | `05_discussion.tex` | relate to prior work + hypotheses; **limitations must be stated** |
| 6 | Conclusions (optional) | `06_conclusion.tex` | keep (our Discussion is long) |
| - | Appendix A, B, … | `07_appendix.tex` | `\appendixtitles{yes}` + `\appendixstart` + `\appendix`; or move to Supplementary Materials |
| - | Back matter | **missing** | see §5 |
| - | References | `references_handnotes.bib` | numbered style, see §6 |

## 2. Front matter rules

- **Title:** concise, specific, relevant. No running title or short title (the editorial office deletes them). Say if it's a replication/systematic review (n/a for us).
- **Authors:** full first and last names; middle initials optional.
- **Affiliations:** full PubMed/MEDLINE-style address: dept, institution, city, **zip code**, country.
  - Innopolis University, 1 Universitetskaya St, Innopolis 420500, Russia *(verify)*.
- **Corresponding author:** at least one is **required**. Draft has Firas commented out; decide.
- **Every author's email is published.** The corresponding author must get everyone's consent.
- Equal contribution: mark with † and add "These authors contributed equally to this work".
- LLMs/GenAI **cannot** be authors.
- ORCID icons are added on acceptance; collect each author's ORCID.

## 3. Abstract and keywords

- ≤ **~200 words**, **one paragraph**, no headings, no citations.
- Follow the structured order without labels: **Background → Methods → Results → Conclusion**.
- Must not contain results that are not in the main text; **must not exaggerate**.
- Keywords: **3–10**, specific to the paper yet common in the field.
  - Candidates: limitation extraction; future work; research gap mining; scientific text mining; large language models; selective verification; sentence classification; PubMedBERT; cost-efficient NLP.

## 4. Body content rules

- **Methods:** enough detail to replicate. Name + **version** of every software/model (PubMedBERT checkpoint, LLM model id + date, Python libs). State whether code is available.
- **GenAI disclosure (important for us):** LLMs are part of the *method* and some references are *model-labelled*. Describe in Methods exactly how GenAI was used (verification stage, reference construction, prompt refinement on benchmark feedback). Grammar/spelling help does not need declaring.
- **Results:** concise and precise, with interpretation and experimental conclusions.
- **Discussion:** interpret against prior studies and the hypotheses; highlight limitations; future directions OK.
- **Reproducibility (MAKE-specific):** "Full experimental details must be provided… publish all experimental controls and make full datasets available where possible." Release the code in a public repo (GitHub + Zenodo DOI recommended).

## 5. Back matter (required, in this order)

```latex
\supplementary{...}            % Figure S1: title, Table S1: title, ...
\authorcontributions{...}      % CRediT taxonomy, initials
\funding{This research received no external funding.}   % or funder + grant no.
\institutionalreview{Not applicable.}
\informedconsent{Not applicable.}
\dataavailability{...}         % MANDATORY
\acknowledgments{...}          % incl. GenAI sentence if used
\conflictsofinterest{The authors declare no conflicts of interest.}
\abbreviations{...}            % optional list (LLM, RCT, SAL, ...)
```

- **Author Contributions:** CRediT roles, e.g. "Conceptualization, Y.A. and F.J.; Methodology, Y.A.; Software, Y.A., O.O. and H.A.; Validation, …; Formal Analysis, …; Investigation, …; Data Curation, …; Writing—Original Draft Preparation, …; Writing—Review and Editing, …; Visualization, …; Supervision, F.J.; Project Administration, …". End with "All authors have read and agreed to the published version of the manuscript."
- **Funding:** exact funder names (check https://search.crossref.org/funding).
- **Data Availability Statement:** mandatory. Our case uses public-domain derived data plus our code/annotations, e.g.:
  > The code and derived annotations are openly available at [GitHub/Zenodo DOI]. The data were derived from the following public-domain resources: [SAL / RCT corpus, FWS, Sci-Sentence, arXiv, …].
- **Acknowledgments + GenAI:** if GenAI was used for text/data/analysis, use the template sentence:
  > "During the preparation of this manuscript/study, the author(s) used [tool name, version] for the purposes of [description]. The authors have reviewed and edited the output and take full responsibility for the content of this publication."
- **Conflicts of Interest:** "The authors declare no conflicts of interest."

## 6. References

- Numbered **[1]** in **order of first appearance** (including in table/figure captions); placed before punctuation: `[1]`, `[1–3]`, `[1,3]`.
- ACS-like style with **full titles**; DOIs strongly encouraged.
- LaTeX: `mdpi.cls` loads natbib in **numeric** mode and sets `\bibliographystyle{Definitions/mdpi}` itself. **Do not** add `\bibliographystyle` or load natbib again.
  `\citep{}` already prints `[n]`, so no mass conversion is needed. Check every `\citet{}`: it prints "Author et al. [n]", which is fine, but reread each sentence.
- Cite data, code and models too (datasets, HF checkpoints, our repo).

## 7. Figures, tables, equations

- Figures: cited in the text and placed after the paragraph that first cites them; high resolution; vector preferred. TikZ is fine in LaTeX.
- Tables: every column has an explanatory heading; font ≥ 8 pt; wide tables via `\begin{adjustwidth}{-\extralength}{0cm}` (template provides).
- Equations editable (LaTeX native, fine).
- **Graphical abstract** (optional, recommended): PNG/JPEG/TIFF, ≥ **560 × 1100 px** (h × w), must **not** duplicate a body figure, no "Graphical Abstract" heading inside.

## 8. Formatting / template conversion checklist

- [ ] `\documentclass[make,article,submit,pdftex,moreauthors]{Definitions/mdpi}`
- [ ] Remove `\tableofcontents`, `\newpage`, `geometry`, `setspace`, `authblk`, `natbib`, `lmodern`, custom `\corrauthor`
- [ ] Map authors to `\Author`, `\AuthorNames`, `\address`, `\corres`, `\firstnote`
- [ ] `\Title{}`, `\abstract{}`, `\keyword{}` (semicolon-separated)
- [ ] Section rename: Methodology → Materials and Methods
- [ ] Check custom packages (tabularx, longtable, pdflscape, subcaption, tikz, cleveref) still compile with mdpi.cls
- [ ] `\cite` conversion + `mdpi.bst`; check every bib entry has title/venue/year/volume/pages/DOI
- [ ] Back matter block (§5)
- [ ] Appendix via `\appendixtitles{yes}\appendixstart\appendix`
- [ ] Total upload ≤ 120 MB

## 9. Policy notes

- The MDPI template is for **journal submission only**. Do not post the MDPI-formatted PDF to arXiv; post a generic-format preprint if needed (preprints are allowed in general).
- The paper must not be under review elsewhere.
- Peer review is **single-blind (anonymous reviewers)**: author names stay in.
- Licence CC BY; there's an APC (check the current amount on the journal page / any institutional discount).
- After acceptance, author names/affiliations may not be changed, so get them right now.

## 10. Reviewer-proofing notes for our paper

- Keep claims scoped: the draft already says LLM results are *exploratory* (prompts tuned on benchmark feedback) and human validation is *pending*. Keep this in the abstract and Discussion; MDPI forbids exaggeration.
- Don't promise idea generation in the title/abstract. That's the next paper (Gap2Idea); here frame it as "towards".
- Report cost honestly (input-token estimate excludes detector compute + output tokens; already stated).
- Make the **data/code link real before submission** (Zenodo DOI of the `minimal` branch commit `28a6c2d` or later).
- Name exact LLM model ids + access dates, PubMedBERT checkpoint, and seeds.
