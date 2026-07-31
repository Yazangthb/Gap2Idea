# Full-system review + realistic improvement plan (IEEE Access)

> Audit of all components (extraction done separately in `01_gaps_extraction.md`).
> Goal: a credible IEEE Access *full-system* paper. Verdict: **achievable, but the
> evaluation evidence — not the code — is what's missing.** The system is built,
> deployed, and reproducible; the claims are not yet backed by real data.

## Component scorecard
| Component | Built? | Evidence/eval state | Verdict |
|---|---|---|---|
| **1. Gap extraction** (funnel + LLM filter) | ✅ strong | small silver gold (19); LimGen comparison | paper-ready w/ honest claims |
| **2. Clustering / gap-graph** | ✅ novel (Leiden + edge-betweenness bridges + frontier) | **bench is N=11 and does NOT include the shipped graph method**; no ground truth | **needs real eval** |
| **3. Idea generation** (5 modes + multi-agent) | ✅ substantive scaffolding (grounded, cross-provider, hallucination gates) | **all committed runs are the SIMPLE path; ZERO orchestrated/multi-agent outputs saved** | **needs runs** |
| **4. Idea evaluation / novelty** | ✅ code ready (panel, S2, human form, Krippendorff α) | **panel never used (single judge, self-bias); human study = ZERO real responses (only `fake_responses.csv`); S2 novelty on ~7/27; evidence-overlap uniformly 1.0** | **MAKE-OR-BREAK; not credible yet** |
| **5. Output + deployment** | ✅ genuinely strong | drafter w/ anti-hallucination guards, LaTeX/PDF/IEEE templates, MCP (12 tools), Docker+Cloud Run+CI, 19 tests | **paper-ready strength** |

## The honest bottom line
The **engineering/systems story is strong and real** (complete, deployed, reproducible, anti-hallucination-engineered) — exactly IEEE Access's profile. The **scientific-evidence story is hollow**: idea quality is currently graded by *one LLM in the same family as the generator*, with **fabricated human data** as the only "human" artifact, and the flagship multi-agent + panel paths are unexercised in the repo. Reviewers will reject on this. Fixing it needs **runs and people, not new code.**

---

## Improvement plan — make it realistic & logical (priority order)

### P0 — Evaluation integrity (the make-or-break; mostly no GPU/credits-light)
1. **Delete all fake data** (`fake_responses.csv`, any analysis built from it) immediately — a reviewer finding fabricated human results is an instant reject + ethics flag.
2. **Run a REAL human study.** The form is built — deploy it. Get **5–10 domain readers** (supervisors, lab peers, grad students) to rate ~15 ideas on novelty/feasibility/impact. Report **Krippendorff's α**. This single step is the highest-value action for publishability.
3. **Actually run the multi-provider judge panel** (code exists) on all ideas, report **inter-judge agreement**, and **report LLM-judge vs human correlation** — that's what justifies using the LLM judge at all.
4. **Fix S2 novelty coverage** (run for all ideas, handle failures) and **drop or upgrade evidence-overlap** (uniformly 1.0 = uninformative; either make it a real source-contribution-overlap check or stop calling it a novelty safeguard).

### P1 — Prove the pipeline LOGIC (the experiment that justifies the whole system)
5. **The key ablation: does the graph/bridge structure actually produce better ideas?** Generate ideas from (a) **bridge pairs** (graph), (b) **random gap pairs**, (c) **single gaps** — and have the panel + humans rate them blind. If bridge/frontier ideas score higher, *every upstream component earns its place*. This is the experiment that turns "a pipeline of LLM calls" into "a justified method." **Do this.**
6. **Exercise the multi-agent path at scale:** produce orchestrated runs (critic loop + sanity + panel), report **gate pass-rates** and a **with/without-critic ablation** (does the critic loop raise scores / cut hallucinations?). Turns the multi-agent claim into evidence.

### P2 — Real-sized component evals
7. **Re-run the clustering benchmark on hundreds of gaps from `runs/*`, INCLUDING the shipped `leiden_graph` method** (currently excluded from the grid). Add the extrinsic signal from #5 (graph pairs → better ideas) so clustering isn't judged only by silhouette.
8. **Bigger / human-spot-checked extraction gold** (19 → a few hundred) — lower priority now that extraction is one component.

### P3 — Narrative, scope, consistency (presentation = IEEE Access acceptance factor)
9. **Frame as a *system + evaluation-methodology* paper**, not "we beat SOTA." Lead with: end-to-end gap→idea pipeline, **cost-efficiency** (the $10-vs-$4000 funnel), **anti-hallucination engineering** (verbatim extraction, evidence-overlap, citation filtering, named-baseline/falsifiability gates), **deployability** (Cloud Run/MCP), and a **real human-validated idea-quality study**.
10. **Honest threats-to-validity section:** LLM non-determinism, small corpora, LLM-as-judge limits, single-snapshot models.
11. **Fix consistency/drift:** complete `run-all` (it omits `extract-methods` → method-gap mode has no methods, and omits export/drafting + never passes `--mode`); align version numbers (pyproject 0.2.0 vs README 0.4); update README template list (3 templates, not 1).

## What's blocked vs doable now (given CPU + credits-out)
- **Doable now (no GPU, credits-light):** the human study (#2) needs *people*, not compute. Deleting fake data (#1). Drafting the paper. The clustering re-run on `runs/*` (#7, CPU-fine, no LLM). The graph-vs-random ablation *design* (#5).
- **Needs credits/GPU:** running the panel + orchestrated idea gen + S2 at scale (LLM calls), the full-scale extraction comparison.
- **Sequencing:** delete fake data → re-run clustering on real corpus (CPU) → draft system/extraction/related-work sections now → when credits return, run panel + orchestrated + the graph ablation → run human study in parallel (people) → write results + finalize.
