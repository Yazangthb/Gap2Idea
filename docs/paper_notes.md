# Paper notes — Gap2Idea

Scratch notes for the write-up. Numbers are from our own runs (2026-08); citations
need verifying before they go in a bibliography.

## Positioning (one line)
Prior work *generates* per-paper limitations; Gap2Idea *extracts* verbatim gap
sentences with a cheap funnel and recombines them **across papers** into research
ideas. A limitation is our intermediate, not our product — so we can't hallucinate one.

## Closest prior work — the NIU / Al Azher limitation-mining line
Verify all IDs/DOIs before citing (links came from a web search, not checked here).

- **LimGen** — Faizullah et al. 2024, *Probing LLMs for Generating Suggestive Limitations*.
  Supervised generation: full paper → limitations section, ACL-mandated sections as label,
  DPR retrieval for the ~5k-tok vs ~230-tok context gap. Defines the task + dataset.
  arXiv 2403.15529 · ECML PKDD 2024 · github.com/arbmf/LimGen
- **LimTopic** — Al Azher et al. 2024 (JCDL). Cross-corpus aggregation → taxonomy of
  limitation types (BERTopic + LLM cluster naming). arXiv 2503.10658 · doi 10.1145/3677389.3702605
- **Graph-limitation** — Al Azher 2024 (JCDL short). LimGen + more stages (draft → refine vs
  GT → per-paper graph → RAG → LLM eval loop). doi 10.1145/3677389.3702612
- **Visual limitations** — Al Azher & Alhoori 2024 (IEEE BigData, pp. 8614–8616). Multimodal
  LLM captions for charts/figures. doi 10.1109/BigData62323.2024.10826112
- **BAGELS** — same first author; the related-work paragraph above is from it. Three of the
  four are one group iterating → largely a self-citation lineage.

## How Gap2Idea differs (for related work)
- **Extraction, not generation** → verbatim, no hallucinated limitations, cheaper.
- **Funnel pre-filter**: Stage A (GROBID sections: keep Limitations/Future-Work/Discussion/
  Introduction, blacklist Related-Work/Background) → Stage B (cue rules + bge-small head) →
  Stage C (batched LLM filter). A+B cut ~98% of sentences before any LLM.
- **Cross-paper graph** (theme-mining: Leiden communities, bridge/frontier scoring) is for
  *idea recombination*, NOT a per-paper limitation graph. Pre-empt reviewers conflating this
  with Al Azher's per-paper 5-stage graph (which they themselves note doesn't scale).

## Measured cost / scale (our runs, 25-paper set)
- **Stage C batched judge**: 50 candidates → 22 kept; **2 LLM calls** (chunk=40),
  2,945 in / 547 out ≈ **3.5k tokens** (~140 tok/paper).
- vs **per-sentence**: 46 calls, ~29.7k tokens → batching is **−88% tokens, 46→2 calls**,
  same decisions (±1 borderline).
- Extrapolated **~1M papers**: ~80k calls, ~140M tokens ≈ **$25–30/1M** at gpt-4o-mini ref
  rates (YandexGPT billed in units). Below the cheapest per-paper *generation* row in the
  BAGELS-style cost table, because we never take a full-paper LLM pass.
- **Precision**: GROBID limitations ~40% (my manual adjudication) → **~90% after Stage C at
  ~86% recall**. Stage C removes prior-work critiques ("existing methods cannot…"),
  contribution/result claims, vague self-promotion.
- Stage A+B = embedding-only (self-hosted bge-small); GROBID ingestion = the "PDF parsing is
  a real line item" cost (~$100s/1M CPU; PyMuPDF fallback).

## Context experiment (negative result worth reporting)
Feeding the judge ±1 sentence (`paragraph_text`) and/or title, batched, same 50 candidates:

| variant | total tokens | kept |
|---|---|---|
| bare sentence | 3,488 | 22 |
| + paragraph (fields) | 8,244 | 24 |
| + paragraph (inline «marked») | 6,426 | 23 |
| + paragraph + title (inline) | 7,333 | 24 |

- Bare is **fully deterministic** (0 flips over 3 runs) → all changes are real signal.
- Context is **net-negative on precision** and 1.8–2.4× the tokens. Inline vs fields is a
  wash → it's **context itself, not the format**.
- **Why**: a gap sentence and its adjacent motivating/prior-work sentence have *opposite
  ownership*; a ±1 window blurs the own-vs-prior boundary the bare sentence draws cleanly.
- Decision: pipeline runs on the **bare sentence**; `use_context`/`use_title` kept off by default.

## Design decisions prior work independently validates
- **Self-validation bias**: judge ≠ generator model family. We saw it when Yandex-only
  collapsed the judge panel; keep a cross-family judge for `evaluate-ideas`. (BAGELS flags it too.)
- **Ground truth is only ~5–15% of papers** (ACL limitations mandate is post-2022). Our funnel
  doesn't need a mandated section per paper (structural + cue rules generalize); ACL sections
  are only a distillation signal for the head — a robustness edge over supervised generation.

## TODO before submission
- Defensible eval protocol (not per-paper manual adjudication, which is LimGen's non-scaling
  60-paper eval): held-out mandated-Limitations set for P/R + second-family LLM judge with a
  **measured human-agreement** number on a sample.
- Check theme-mining memory profile at scale (BERTopic UMAP+HDBSCAN hits a memory wall at 1M;
  confirm Leiden-on-graph doesn't).
- Reproducibility: freeze the exact commands + run artifacts behind the headline numbers.
