# Gold gap-dataset — LLM-ensemble labeling + human validation (plan)

Goal: build a **scientifically defensible** ground-truth benchmark of papers with per-sentence
limitation / future-work (gap) labels. LLM outputs alone are **not** a valid benchmark; validity comes
from (a) a strong multi-model ensemble with measured agreement, and (b) a **human-validated** subset with
reported human–LLM agreement. Pilot on a few papers before the full run.

## 1. Task + label schema
- Unit: **sentence** (full paper → numbered sentences via our `split_sentences`).
- Label per sentence: **`limitation`**, **`future_work`**, or **`none`** (binary "gap" = limitation ∪ future_work for the headline metric; the two-way type is kept for analysis).
- Definitions (annotation guideline, reused by both LLMs and humans):
  - **limitation** = a weakness/caveat/bias/constraint the authors acknowledge about *their own* work
    (SAL criteria: cause of a self-acknowledged limitation; consequences kept only when the cause is evident).
  - **future_work** = an explicit direction the authors propose for future research.
  - **none** = everything else (background, method, results, related-work description, etc.).

## 2. Labeling method — strong ensemble, used smartly
- **Judge 1 (primary):** `claude-opus-5-5`, effort `high`, adaptive thinking — reads the whole paper
  (numbered sentences), returns `{limitation:[...], future_work:[...]}` indices (structured output).
- **Judge 2 (secondary):** `claude-sonnet-5-5`, same prompt — independent second opinion.
- **Agreement → confidence:** sentences where both judges agree = high-confidence gold. Disagreements go to
  **adjudication by `claude-opus-5-5` at effort `max`** (or are flagged for the human pass).
- **Cost levers (smart use):**
  - **Batch API** (`messages.batches`, async) = **50% off** — this job is not latency-sensitive.
  - **Prompt caching** of the shared system prompt + guideline (stable prefix) across papers.
  - Structured outputs (`output_config.format`) so parsing is exact (indices, not prose).
- Report **inter-model agreement** (Opus vs Sonnet, Cohen's κ) as a quality signal.

## 3. Human validation (what makes it a real benchmark)
- **Stratified sample** of ~200 sentences from the labeled set: positives (both-agree), negatives, and
  **all/most disagreements** (the informative cases).
- **≥2 human annotators** label them blind against the same guideline; compute **inter-annotator
  agreement** (Cohen's κ / Krippendorff α) and **human vs LLM-gold agreement**.
- Deliverable: a worksheet (sentence + ±1 context + guideline) + an agreement report. If human–LLM κ is
  high (target ≥0.75, ~ SAL's α≈0.78 ceiling), the LLM-gold is validated for the rest of the corpus;
  disagreement patterns are reported as the error analysis.

## 4. Corpus
- **Primary:** the 190 full-text arXiv CS/ML papers (`data/paper_texts.jsonl`) — a CS gap benchmark
  (none exists; see `dataset_citation_audit.md`). Optionally **+ 40 SAL biomedical** for cross-domain.
- Expandable via `arxiv_select.py` if a larger N is wanted and budget allows.

## 5. Budget (≈$100 Claude credits)
Est. ~17k input + ~1.5k output tokens/paper (arXiv papers are long). Per paper:
- Opus 5.5 ≈ $0.098, Sonnet 5.5 ≈ $0.049 → 2-judge ≈ **$0.15/paper** at standard rates.
- **With Batch API (−50%)** ≈ **$0.075/paper** → **190 papers ≈ $14**, + adjudication ≈ $3.
- So the full 190 (2-judge ensemble + adjudication) ≈ **$17**; ~$100 comfortably covers **500–900 papers**
  or a 3-judge ensemble / higher effort. (Exact numbers from the pilot.)

## 6. Execution plan
0. **Add `ANTHROPIC_API_KEY`** to `.env` (local only; never printed/committed).
1. **Pilot (≈$0.50):** run the ensemble on **5 papers**; inspect agreement + a few sentences by hand;
   tune the guideline/prompt; get the **measured** per-paper cost.
2. **Full run** (budget-bounded) via Batch API; write `data/gold/gap_gold.jsonl` (per-sentence labels +
   per-judge votes + confidence).
3. **Human-validation worksheet** + agreement report.
4. Fold into the benchmark + `comparison_plan.md` (now with a *human-validated* gold).

## Guardrails
- API key stays local (`.env`); never echoed or committed. Keep the H100 out of this (Claude runs from here).
- Pilot-validate before the full spend; stop if agreement is poor or cost/paper is off.
