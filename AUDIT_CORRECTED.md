# Gap2Idea — Corrected Audit Summary

*Consolidated from the 5 Oct 2026 audit (README / EXPERIMENTS / METHODS_DATASETS / TEAM_TASKS / SCRIPT_CATALOG), corrected against the actual working tree on this machine (6 Oct 2026). Corrections are marked **[FIX]**.*

---

## What the project is

**Gap2Idea** = a pipeline that turns academic papers into **research-gap extractions**, then into **generated research ideas**. It is really **two bodies of work** that need separate paper claims:

1. **Gap extraction + evaluation** — well benchmarked (the safe publishable scope).
2. **Gap → idea generation** — implemented but not yet validated (needs controlled study + human ratings).

There is no single wired-up "final method" yet.

---

## Repository facts (corrected)

| Item | Status |
|---|---|
| Current checkout (HEAD) | **`e1844b5`** on branch `chore/repo-reorg`, 19 Aug 2026 |
| `master` tip | `e98f82f`, 30 May 2026 |
| **[FIX]** "current checkout is e98f82f" | Wrong — `e98f82f` is master, not the checkout |
| **[FIX]** `ec733de` / branch `experments/bench-n100-v2a` | **Does not exist in this clone.** All "historical N=100" evidence (E09, E10, `sources/n100-ec733de`) references a missing ref |
| Scripts in `scripts/` | **87** ✓ |
| Referenced source files (gap_funnel, grobid_sections, pdf_text_v2, gap_graph, …) | All present ✓ |

---

## The key correction: "missing artifacts" are mostly here

The audit read **git-tracked content only**. Many files it calls *absent / unavailable / must-be-recovered* **exist on this machine** as **gitignored** local data (0 git-tracked, so invisible to a git-only audit):

- `runs/{ai,ml,math,*_v1}/` — **296 files**
- `artifacts/` (incl. `bagels_cache/`)
- `data/bench_gold/` (`gaps_full.jsonl`, `_flat.tsv`, `papers_manifest.tsv`)
- `data/bench_gap/` (`gold_sentences.tsv`, `_v2`, `_unverified`, `train/`)
- `data/datasets/` (`dataset_v1..v9plus`, `r2`, `test25`, `excellent_200`)
- `data/arxiv-metadata-oai-snapshot.json` (**5 GB**), `data/pdfs/`, `data/limgen/`, `data/acl_limitations.tsv`, `fws_recognition.csv`

**Implication:** T01/T02 recovery is largely already satisfied locally. The real job is to **package/commit these gitignored artifacts**, not hunt for them.

---

## Verified numbers (these are correct)

| Metric | Value | Source |
|---|---|---|
| N=100 extraction | **161 gaps / 90 papers** | `data/bench_n100/gaps.tsv` |
| N=100 sections coverage | **98** paper IDs | `sections_extracted.jsonl` |
| N=100 gold records | **100** | `bench_papers.jsonl` |
| v2b N=10 recovery@0.6 (`llm_gap`) | **0.700** | `bench_v2b/REPORT.md` |
| Legacy label sheet | **481 rows, 0 filled `gold_label`** (all silver) | `data/bench/label_sheet.tsv` |
| Gold v2 | **49 gaps / 10 papers** | `bench_gap/gold_sentences_v2.tsv` |
| Gold v1 | **19 retained / 9 papers** (31 unverified) | `bench_gap/gold_sentences.tsv` |

---

## [FIX] The "malformed N=100 input" finding is false

The audit claims `data/bench_n100/paper_texts.jsonl` is corrupt ("21 invalid lines, only 99 records"). **It is not.** On both `e1844b5` and `e98f82f` the file is **100 clean CRLF records, all valid JSON, 100 unique IDs** — under strict *and* naive readers, 0 embedded stray CRs.

→ **Task T02 (P0) is based on a non-existent problem.** Drop or re-scope it.

---

## Evidence strength (unchanged, still valid)

Experiments E01–E40 are tagged:
- **A** saved artifact · **L** logged-only (no predictions/checkpoints) · **N** notebook · **I** implemented, unrun · **P** proposal

Most benchmark results (LimGen, RCT/SAL, FWS, BAGELS, clustering) are **L — logged but not reproduced here**. Treat all "SOTA/paper-ready" log language as **claims to verify**, not established results. LLM-built gold = **silver**, not human gold. Synthetic demo content (`expand_corpus_demo.py`, `fake_responses.csv`) must stay out of evidence.

---

## Corrected task priorities

| Was | Corrected action |
|---|---|
| T01 recover artifacts | Mostly done locally — **commit/package the gitignored `runs/`, `data/`, `artifacts/`** |
| T02 fix malformed N=100 file | **No corruption exists** — close or re-scope |
| T03 clean repo skeleton | Still valid and pickable now |
| Gold counts "unavailable" (D07/D08/D09) | **Verifiable now** — files present (49/19/31 confirmed) |
| "Logged" benchmarks | Still the real work: reproduce from inputs that are now on disk |

---

## Bottom line

**Logged metrics are trustworthy; the availability status and the corruption finding are not.** Re-run the audit against this working tree **including gitignored paths** before acting on the recovery cards — most of what it declared missing is already here.
