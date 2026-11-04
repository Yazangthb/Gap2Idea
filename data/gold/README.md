# Gap2Idea gap-gold dataset — datasheet

A per-sentence **research-gap** benchmark (limitation / future-work / research-gap) over biomedical and
CS/ML papers, built with a Claude judge-ensemble and (pending) human validation. This datasheet
documents how it was collected, generated and validated, for the paper's methods + data-availability
sections. Follows the "Datasheets for Datasets" structure.

---

## 1. Motivation
Research-gap / limitation detection has **no reused, shared, human-validated benchmark**: existing sets
(SAL/Lan 2024, Kilicoglu 2018, FWS, LimGen, BAGELS, Sci-Sentence) are each used almost only by their own
authors, cover one domain/one gap type, and SAL/Kilicoglu are known to **under-label** (see
`docs/experiments/dataset_citation_audit.md`). This dataset provides a **multi-domain, multi-type,
ensemble-labeled + human-validated** gap gold to evaluate extraction systems comparably across domains.

## 2. Composition
- **230 papers, 68,719 sentences.**
  - **arXiv CS/ML**: 190 full-text papers (`data/paper_texts.jsonl`, extracted from `data/pdfs/*.pdf`).
  - **Biomedical**: the 40 SAL (Lan et al. 2024) test papers (reused with their own sentence boundaries
    and gold limitation labels for a built-in validity check).
- **Label per sentence**: `limitation`, `future_work`, `research_gap`, or none. Binary **gap** = any type.
- **Final gold: 4,462 gap sentences** — research_gap 2,234 · limitation 1,543 · future_work 685
  (by domain: arXiv 3,751, biomedical 711).
- **Confidence tiers**: `high` = both judges agree same type (2,960); `adjudicated` = one-sided candidate
  confirmed by the Opus adjudicator (1,502).

## 3. Collection & generation process
1. **Source papers.** arXiv CS/ML PDFs → text via container-free `pdf_to_blocks` (PyMuPDF); SAL papers
   from `MengfeiLan/SAL_Type_Classification` (test split).
2. **Preprocessing.** Sentence segmentation with `split_sentences` (arXiv); SAL keeps its own boundaries.
   References/Bibliography/Acknowledgements trimmed for arXiv. Each paper is a numbered transcript
   (`[i] sentence`, with inline `## section` headers where available).
3. **Labeling — 2-judge ensemble.** The whole paper (one call, fits 1M context) is sent to each judge:
   - Judge 1 **`claude-opus-5-5`**, Judge 2 **`claude-sonnet-5-5`** (default effort), independent.
   - Each returns JSON `{type:[{i,q}]}` (index + short quote); labels are mapped by index and
     **verified by quote** (re-aligned if the quote doesn't match the index).
   - Shared annotation guideline (the three gap-type definitions; limitation follows SAL's
     cause-of-a-self-acknowledged-limitation criterion). Script: `scripts/dataset/gold_label_full.py`.
4. **Agreement = confidence.** Sentences where both judges pick the same type → `high` gold.
5. **Recall boost via adjudication.** Candidate set = **union** of both judges (SAL recall 0.956 vs 0.868
   for the intersection). The 1,737 one-sided / type-mismatch candidates were adjudicated by
   **`claude-opus-5-5` (effort high, ±3 context)**; 1,502 confirmed (joined the gold), 235 rejected.
   Scripts: `scripts/dataset/gold_adjudicate.py`, `gold_finalize.py`.
6. **Independent triple-check (validity probe).** The 227 "extra" limitations on SAL papers (ours, not in
   SAL gold) were strictly re-verified by a 3rd frontier judge **`claude-fable-5-1`** (fallback Opus-5.5
   max): **116/227 = 51% confirmed** → about half are real SAL under-labels, half are over-labels of our
   inclusive adjudicator (hence the human pass). Script: `scripts/dataset/verify_extras.py` (scratch).
7. **Cost.** Labeling $31.90 + adjudication $4.10 + triple-check $1.53 ≈ **$37.5** (Claude API).

## 4. Validation
- **Inter-model agreement (Cohen's κ, 68,719 sentences):** any gap **0.778**, limitation 0.748,
  future_work 0.869, research_gap 0.725. κ=0.778 ≈ the human inter-annotator ceiling on this task
  (SAL/Kilicoglu α ≈ 0.78).
- **Built-in SAL check (40 papers, 205 gold limitations):** binary-gap recall **0.956**, limitation
  recall 0.927; precision-vs-SAL 0.456 (SAL under-labeling artifact — on the same papers we label ~2×
  the limitations, 190/205 overlap + 227 extras).
- **Triple-check:** 51% of extras confirmed by an independent strict frontier model.
- **Human validation (PLANNED — the proof):** blind 362-row worksheet (227 extras + 15 SAL-missed + 60
  control-positives + 60 negatives, shuffled, no system labels), ≥2 annotators, report inter-annotator κ
  and human-vs-gold / human-vs-SAL agreement. Files: `human_validation_worksheet.xlsx`,
  `worksheet_key_PRIVATE.csv` (kept out of the shared sheet).

### Human validation — how to run
1. **Build the labeling UI** (optional; `annotate.html` is already generated):
   `python scripts/dataset/build_annotate_html.py` → writes `data/gold/annotate.html` (self-contained;
   reproduces the exact seed-13 pool, so rows match `worksheet_key_PRIVATE.csv` and the Google Sheet).
2. **Annotate (blind).** Each annotator opens `annotate.html` in Chrome/Edge, picks **A1** or **A2**,
   and judges all 362 items (target sentence highlighted in its section; "Open in paper · highlighted ↗"
   deep-links to the live PMC article). Answers autosave in the browser; **Export .xlsx** when done
   (`gap_annotations_A1.xlsx`, `gap_annotations_A2.xlsx`). The Google Sheet is an online alternative.
3. **Score.** `python scripts/dataset/gold_human_merge.py data/gold/gap_annotations_A1.xlsx data/gold/gap_annotations_A2.xlsx`
   → control-sanity, inter-annotator κ (is_gap + type), empirical precision of the 227 "extra"
   limitations (vs Fable's 51%), SAL-missed confirmation, and human-adjudicated precision of ours vs
   SAL. Writes `docs/experiments/human_validation_results.md`. Also accepts one combined Sheet export
   with `A1_*`/`A2_*` columns. **Keep `worksheet_key_PRIVATE.csv` out of the shared sheet / any export.**

## 5. Files
| file | contents |
|---|---|
| `gap_gold_final.jsonl` | **the gold** — per sentence: paper_id, domain, idx, section, sentence, label, confidence (high/adjudicated), sal_gold |
| `gap_gold.jsonl` | per-paper raw: each judge's votes (indices per type), agreement, per-paper cost |
| `gap_gold_sentences.jsonl` | per-sentence flattened intermediate (both judges' labels, confidence) |
| `adjudication.jsonl` | per-paper adjudicator verdicts on one-sided candidates |
| `extras_triplecheck.json` | Fable-5.1 strict verdicts on the 227 SAL extras |
| `disagreement_pools.json` | the extras / SAL-missed index pools |
| `human_validation_worksheet.xlsx` | blind 2-annotator worksheet (Guidelines + Annotate tabs, dropdowns) |
| `annotate.html` | self-contained blind labeling UI (highlighted sentence + PMC deep-link, autosave, .xlsx export) |
| `worksheet_key_PRIVATE.csv` | row→source answer key (PRIVATE; not for the shared sheet) |
| `run.log` | labeling run log |

Scripts (reproduction): `scripts/dataset/gold_label_full.py` (label) → `gold_adjudicate.py` (adjudicate)
→ `gold_finalize.py` (final gold + stats) → `gold_postprocess.py` (agreement report).
Human validation: `build_annotate_html.py` (build the UI) → `gold_human_merge.py` (merge + score).
Report: `docs/experiments/gold_dataset_results.md`; plan: `docs/experiments/gold_dataset_plan.md`.

## 6. Label schema
- **limitation** — a weakness/caveat/bias/constraint the authors acknowledge about **their own** work.
- **future_work** — an explicit direction the authors propose for future research.
- **research_gap** — an unsolved problem / missing capability / open challenge in the field or prior work.
- **none** — everything else.

## 7. Known limitations / caveats (state in the paper)
- **LLM-labeled**; validity rests on inter-model agreement + the pending human pass, not on any single model.
- The **adjudicator is inclusive** → the `adjudicated` tier over-labels limitations (only 51% of SAL extras
  pass a strict 3rd-model check). Prefer the `high` tier for high-precision use; use `adjudicated` for recall.
- A recall-biased exhaustive pass for gaps **both** judges missed (~4% of SAL gold) was **not** run.
- arXiv sentence segmentation is noisy on a few math/table-heavy papers (fragmented "sentences").
- "Precision vs SAL" is not true precision — SAL under-labels; true precision comes from the human pass.

## 8. Provenance, licensing, ethics
- arXiv papers: public; per-paper arXiv license (mostly CC-BY / arXiv non-exclusive). SAL: reused under its
  dataset license (Lan et al. 2024). Only sentence text + derived labels are stored; no PII, no human subjects.
- Labels generated via the Anthropic Claude API (Opus 5.5, Sonnet 5.5, Fable 5.1), Oct 2026.

## 9. Citation
TODO (paper in prep). Cite alongside SAL (Lan et al. 2024, J. Biomed. Informatics) and the gap-mining
funnel (Gap2Idea, RQ2).

_Last updated: 2026-10-18._
