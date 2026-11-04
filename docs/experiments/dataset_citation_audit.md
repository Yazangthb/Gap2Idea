# Dataset reuse audit — who actually evaluates on the gap/limitation datasets (Oct 2026)

For each dataset we use, checked its citing papers (Semantic Scholar + OpenAlex) and flagged which
**reuse the dataset for evaluation on a detection/extraction/generation task** (vs citing it
conceptually or building their own). **Caveat:** judged from title/abstract — silent dataset use is
undercounted, so "reuse = 0" means "none detected," not a proof of zero. Even so, the signal is strong.

## Per-dataset

| dataset | paper | citations | reuse the data for a task? | notable citers |
|---|---|---|---|---|
| **Sci-Sentence** | Bolaños et al. 2025 (arXiv 2508.04337 / PeerJ cs-3921) | 5 | **0** (the one "match" is the authors' own PeerJ version) | UniCite; OntoGPT; OUTLINEFORGE (none use it) |
| **SAL** | Lan et al. 2024, *J. Biomed. Informatics* (10.1016/j.jbi.2024.104628) | ~8–10 | **0** reuse for limitation detection | *Multi-label Seq. Sentence Classification via LLM* (EMNLP-F 2024) uses its own BIORC800; SPIRIT-CONSORT-TM (own corpus) |
| **Kilicoglu 2018** | *Automatic recognition of self-acknowledged limitations in clinical research literature* (OpenAlex W2802420380) | 15 | **0** reuse the dataset (its successor **SAL builds a new one** instead) | SAL (2024); FWS (2023); CONSORT-completeness (2024); transparency assessment (2021) |
| **FWS** | Zhang et al. 2023, *J. Informetrics* (arXiv 2212.13860) | 18 | **~0–1** (#17 *Using FWS to Explore Research Trends*, 2021, builds its **own** FWS corpus) | FutureGen (future-work *generation*, RAG); many scientometric topic-trend works cite conceptually |
| **LimGen** | Faizullah et al. 2024, ECML-PKDD (arXiv 2403.15529) | 21 | **0** confirmed reuse | BAGELS; LimTopic (2024); *Multi-Agent LLMs for Generating Research Limitations* (2025); mostly peer-review/generation/survey |
| **BAGELS** | Al Azhar et al. 2025, EMNLP-Findings (arXiv 2505.18207) | 4 | **0** (Tree-of-Concerns builds its **own** ToC-Bench) | FutureGen; LLLMs survey; Multi-Agent limitation generation |

## The finding
Across **~70+ citing papers** over six gap/limitation datasets, **essentially none reuse the datasets
as a shared evaluation benchmark** for limitation/future-work **detection or extraction**. Citers
overwhelmingly (a) cite conceptually, (b) **build their own new dataset** (SAL vs Kilicoglu; the FWS
trend papers; BAGELS→ToC-Bench; Yale's LimitGen), or (c) do an adjacent task (CONSORT completeness,
peer-review generation, topic-trend scientometrics).

**Implication for our paper (two-edged):**
- *Opportunity / novelty:* the limitation/gap-mining subfield has **no standard, reused benchmark** —
  everyone rolls their own and reports numbers that aren't comparable. A single funnel evaluated
  **across** several existing datasets (SAL + Kilicoglu + FWS + Sci-Sentence) is itself a contribution:
  cross-dataset comparability nobody else provides.
- *Risk:* it also means there is **no external baseline for our exact task** on most of these sets — our
  comparisons are to each dataset author's own (often different-task) numbers. State this honestly.

## Adjacent works worth tracking / citing
- **LimitGen** (Xu et al., Yale-NLP, ACL 2025) — "Can LLMs Identify Critical Limitations…" (separate from LimGen).
- **Multi-Agent LLMs for Generating Research Limitations** (2025).
- **LimTopic** (2024) — LLM topic modeling of limitations.
- **Tree-of-Concerns** (2026) — unstated-limitation extraction (own ToC-Bench, 414 papers / 1,905 limitations).
- **What Limits Us? Analyzing Self-Reported Limitations in NLP Research** (2026).

_Sources: Semantic Scholar graph API (citations), OpenAlex (Kilicoglu W2802420380). Counts as of Oct 2026._
