# Zenodo submission, preprint v1

Everything needed for the Zenodo upload form (zenodo.org → New upload), in the order the form asks for it.
Copy each value into its field. Items marked **TODO** need a decision or a value before submitting.

## Before you open the form

- [ ] All co-authors have read the final PDF and agreed to the upload.
- [ ] Decide whether Rustam (supervisor) is on v1. `main.tex` says "to be added later as last author".
- [ ] Collect ORCIDs for every author (they are placeholders in `main.tex`).
- [ ] Confirm "no external funding" (`main.tex` still has `% TODO confirm`).
- [ ] Merge the 9B data from `chore/repo-reorg` (commit `657bd66`, without the revert `7db8835`) into
      `paper-writing-v2`. The paper says every LLM verdict for every model size is in the repository.
- [ ] Push `paper-writing-v2`, then tag the exact commit, e.g. `git tag preprint-v1 && git push origin preprint-v1`,
      so the Zenodo record points to a fixed state of the code and data.
- [ ] Rebuild `manuscript_mdpi/preprint.pdf` from that commit.
- [ ] Optional: reserve the DOI first (step 1 below) and print it on the preprint's title page, then rebuild.

## The form

### 1. Files and DOI

| Field | Value |
|---|---|
| Files | `manuscript_mdpi/preprint.pdf`, renamed to `Alnakri_et_al_2026_Gap_Extraction_Preprint_v1.pdf` |
| Do not upload | `main.pdf` (the MDPI submission template; keep it for the journal) |
| "Do you already have a DOI?" | No |
| Reserve DOI | Optional. Click "Get a DOI now!" to reserve it before publishing, if you want to print it in the PDF. |

### 2. Basic information

| Field | Value |
|---|---|
| Resource type | Publication → Preprint |
| Title | Scalable Extraction of Author-Stated Research Gaps: Towards Automated Idea Generation |
| Publication date | The day you publish (YYYY-MM-DD) |
| Version | v1 |
| Language | English |
| Publisher | Zenodo (default) |

### 3. Creators (in this order)

| # | Name (Family, Given) | Affiliation | ORCID | Role |
|---|---|---|---|---|
| 1 | Alnakri, Yazan | Innopolis University (Laboratory for Robotics and Artificial Intelligence) | **TODO** | Contact person |
| 2 | Orabi, Osama | Innopolis University (Laboratory for Robotics and Artificial Intelligence); Moscow Independent Research Institute of Artificial Intelligence (MIRAI) | **TODO** | |
| 3 | Aldaghstany, Hamza Shafee | Innopolis University (Laboratory for Robotics and Artificial Intelligence) | **TODO** | |
| 4 | **TODO** Rustam, if on v1 | Innopolis University | **TODO** | Supervisor |

Pick the affiliations from Zenodo's ROR list where they appear (Innopolis University has a ROR entry), so they link correctly.

### 4. Description

Paste as plain text (it matches the paper's abstract):

> Papers state their own research gaps: limitations, future work and open problems. These statements could seed automated idea generation, but they are rare, scattered and easily confused with criticism of prior work. Asking a large language model (LLM) to judge every sentence is costly, and giving it the whole paper buries a few relevant sentences. We instead narrow each paper with a funnel: structural rules discard unlikely text, a natural-language-inference classifier proposes candidates, and the LLM verifies them without rewriting. Every output is a sentence the authors wrote, and one instruction sets the verifier to favour precision or recall. We evaluate on human-labelled limitation (SAL) and future-work (FWS) benchmarks and on 230 papers labelled by two LLM annotators and an LLM judge. The instruction controlled the trade-off on every dataset, and verification more than doubled SAL precision at 85% recall. Because the LLM only checks a short candidate list, it reads about 7–8 times fewer input tokens than whole-paper prompting on full papers, while matching or improving F1 on the human-labelled benchmarks at every model size. The gain is largest for small models: with a 4B LLM, F1 rose from 0.30 to 0.48 on arXiv papers and from 0.66 to 0.71 on SAL.
>
> Code, prompts, labels and all LLM verdicts: https://github.com/Yazangthb/Gap2Idea

**If the abstract changes before submission, update this text from `manuscript_mdpi/main.tex`.**

### 5. Licence and rights

| Field | Value |
|---|---|
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Why | Standard for preprints, and compatible with MDPI's CC BY open-access licence for the later journal version |
| Copyright | © 2026 the authors (optional field) |

### 6. Keywords

research gap extraction · limitations · future work · scientific text mining · large language models ·
natural language inference · zero-shot classification · LLM filtering

(Enter one per box. These are the paper's own keywords.)

### 7. Related works

| Relation | Identifier | Scheme | Resource type |
|---|---|---|---|
| Is supplemented by | https://github.com/Yazangthb/Gap2Idea/tree/preprint-v1 (**TODO** after tagging) | URL | Software |
| References | 10.1016/j.jbi.2024.104628 (Lan et al. 2024, SAL) | DOI | Publication / Journal article |
| References | 10.1145/3383583.3398526 (Hao et al. 2020, ACL FWS-RC) | DOI | Publication / Conference paper |
| References | 10.1016/j.joi.2022.101373 (Zhang et al. 2023, FWS recognition) | DOI | Publication / Journal article |
| References | 10.5281/zenodo.24018390 (LR-AI, used for the related-work search) | DOI | Publication / Preprint |

The first row matters most: it links the preprint to the exact code and data. The others are optional.

### 8. Funding

None (leave empty), unless the funding check above changes this.

### 9. Additional notes (Zenodo's "Additional descriptions" → Notes)

> Preprint v1. Results on the three-type corpus rest on labels produced by language models that have not yet been validated by people; a human study is planned for a later version. The SAL and FWS benchmarks have human labels.
>
> Use of generative AI: the authors used Claude Opus 5.5 (Anthropic), through Claude Code, to write and run analysis code and to draft and edit text, and OpenAI Codex to revise text. Claude Haiku 5.5 (Anthropic), within LR-AI, screened candidate papers for the related-work review. The authors reviewed and edited the output and take full responsibility for the content.

### 10. Communities (optional)

None needed. Add the record to a lab or university community later if one exists.

## After publishing

- [ ] Copy the version DOI and the concept DOI (the one that always points to the latest version).
- [ ] Cite the concept DOI when sharing.
- [ ] Add the DOI to the repository README and to `references.bib` if the paper should cite its own preprint.
- [ ] When submitting to MDPI, mention the preprint DOI in the cover letter.
- [ ] For v2 (e.g. after the human study), use "New version" on the same record, so both versions share the concept DOI.
