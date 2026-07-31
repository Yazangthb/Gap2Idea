# IEEE Access paper — outline + early drafts

> Draft scaffolding for the full-system paper. Sections marked **[READY]** can be
> written now from existing material; **[BLOCKED]** need the P0/P1 evaluation work
> (see `00_full_system_review.md`). Drafts below are first-pass — honest, no
> invented numbers.

## Title (candidates)
- "Gap2Idea: A Cost-Efficient End-to-End System for Mining Research Gaps and Generating Evaluated Research Ideas"
- "From Papers to Vetted Ideas: A Deployable Pipeline for Scalable Research-Gap Mining and Grounded Idea Generation"

## Contributions (claim list — keep honest)
1. **An end-to-end, deployed system** (ingestion → gap extraction → gap-graph clustering → grounded multi-agent idea generation → multi-faceted evaluation → paper drafting), open and reproducible (Cloud Run, MCP, tests).
2. **A cost-efficient extraction funnel** (free structural slice → cue+embedding classifier → LLM precision filter on ~6 survivors/paper): ~$10 vs ~$4,000 per 1M papers, with an empirical **data-not-model** finding and a **mandated-Limitations-section harvesting** recipe.
3. **A multi-relational gap-graph** (semantic/paper/section/method edges; Leiden communities; edge-betweenness *bridge* and *frontier* scoring) for *novelty-by-recombination* idea seeding.
4. **A grounded multi-agent idea generator** with anti-hallucination gates (verbatim-evidence constraint, evidence-overlap check, named-baseline + falsifiable-prediction requirements).
5. **A reproducible idea-evaluation methodology**: cross-provider LLM judge panel (inter-judge agreement) + Semantic-Scholar novelty + **a human expert study (Krippendorff's α)**, with **LLM-vs-human correlation** reported.

## Section outline + experiment matrix
| § | Section | Status | Key content / experiment |
|---|---|---|---|
| I | Introduction | [READY] | problem, cost wall, contributions |
| II | Related work | [READY] | FWS/limitation extraction (funnel pattern), gap datasets, idea generation, LLM-as-judge — from `related_work_analysis.md` |
| III | System architecture | [READY] | the 6-stage pipeline + deployment (Cloud Run/MCP) |
| IV | Gap extraction | [READY] | funnel; data-not-model ablation; cost; LimGen comparison (`01_gaps_extraction.md`) |
| V | Gap-graph clustering | [PARTIAL] | graph method [ready]; **benchmark on real corpus incl. leiden_graph [P2]** |
| VI | Idea generation | [PARTIAL] | modes + multi-agent [ready]; **orchestrated runs + with/without-critic ablation [P1]** |
| VII | Evaluation | [BLOCKED] | **panel + S2 + human study + LLM-vs-human correlation [P0]; graph-vs-random idea-quality ablation [P1]** |
| VIII | Deployment & reproducibility | [READY] | Docker/Cloud Run/CI, MCP, templates, tests |
| IX | Threats to validity | [READY] | LLM non-determinism, small corpora, LLM-as-judge limits |
| X | Conclusion | [READY] | |

---

## DRAFT — Abstract (first pass, honest)
> The volume of scientific literature makes it infeasible for researchers to
> manually identify open research gaps and synthesize new directions. Large
> language models (LLMs) can help, but applying them per paper does not scale —
> extracting gaps from a million papers with a per-paper LLM costs on the order of
> thousands of dollars. We present **Gap2Idea**, an end-to-end, deployable system
> that mines research gaps from scientific papers and generates evaluated research
> ideas. Extraction uses a **cost-efficient funnel**: a free structural slice and a
> lightweight classifier reduce each paper to a handful of candidate gap sentences,
> over which an LLM is applied only as a precision filter — cutting cost by two
> orders of magnitude (≈$10 vs ≈$4,000 per million papers) while matching the
> quality range of a fine-tuned transformer on a shared benchmark. We show
> empirically that extraction quality is **bounded by training data, not model
> capacity**, and contribute a recipe that harvests mandated "Limitations" sections
> as clean supervision. Extracted gaps are organized into a **multi-relational
> gap-graph** whose community bridges and frontier nodes seed **novelty-by-
> recombination** idea generation, performed by a **grounded multi-agent** process
> with explicit anti-hallucination constraints. Generated ideas are assessed by a
> **multi-faceted protocol** combining a cross-provider LLM judge panel, automated
> novelty checks against Semantic Scholar, and **a human expert study**. The system
> is open-source, reproducible, and deployed on Google Cloud Run with a Model
> Context Protocol interface. [RESULTS SENTENCE — fill after P0/P1: human ratings,
> inter-rater α, LLM-human correlation, graph-vs-random ablation.]

## DRAFT — Introduction (skeleton + opening, honest)
**¶1 Problem.** Scientific output grows faster than any researcher can track;
identifying *what has not yet been done* — the research gaps — and turning them
into concrete, novel, feasible ideas is a core but unscalable scholarly task.

**¶2 Why naive LLM use fails.** LLMs can read a paper and list its gaps, but a
per-paper call is cost- and rate-limited (~$4,000 / 1M papers, days–weeks), and
end-to-end "generate me an idea" prompting yields ungrounded, hard-to-verify
output with no provenance or novelty guarantees.

**¶3 Our approach.** Gap2Idea separates *cheap, scalable retrieval* from *expensive,
selective reasoning*: a structural-lexical funnel does the bulk extraction for
free and routes only a few candidate sentences per paper to an LLM; gaps are then
structured into a graph that makes *recombination* (a known engine of novelty)
explicit; idea generation is grounded in retrieved evidence and gated against
hallucination; and idea quality is evaluated with both automated and **human**
signals rather than self-reported LLM confidence.

**¶4 Contributions.** [bullet the 5 contributions above.]

**¶5 Availability.** Open-source, tested, containerized, deployed (Cloud Run +
MCP); all benchmarks and prompts released.

---

## Writing-discipline reminders (for all sections)
- No invented numbers; mark every TBD explicitly.
- "matches the range of / competitive with" — never "beats SOTA."
- Present the human study as *primary* idea-quality evidence; LLM panel as
  *secondary*, validated by correlation with humans.
- Foreground cost/scalability + deployment/reproducibility (IEEE Access values these).
