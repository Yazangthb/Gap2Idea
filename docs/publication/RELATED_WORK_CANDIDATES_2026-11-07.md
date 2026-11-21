# Related-work candidates from the LR-AI run (2026-11-07)

Quick literature search for references the manuscript does not cite yet. Run folder:
`lr-runs/gap-funnel-related-2026-11-07/` (config, papers, screening and summary results, `papers.xlsx`).
Screening and summaries were done by Claude Haiku subagents from titles and abstracts, so check each paper before
citing it. None of these is in `manuscript_mdpi/references.bib` yet.

Counts: 208 unique records (OpenAlex and arXiv; Semantic Scholar refused anonymous access), 4 removed by filters,
204 screened, 88 included (32 core, 56 skim), 7 of them already cited.

## Recommended

| Paper | Where it fits | Why |
|---|---|---|
| Liu et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2024 (arXiv 2307.03172) | Introduction; long inputs | LLM accuracy drops when the relevant text sits in the middle of a long input. Backs "long inputs are hard for smaller models". |
| Hsieh et al., *RULER: What's the Real Context Size of Your Long-Context Language Models?*, 2024 (arXiv 2404.06654) | Long inputs; model-size results | Many models degrade well before their claimed context size. Frames the 4B and 9B whole-paper breakdown on long arXiv papers. |
| Leng et al., *Long Context RAG Performance of Large Language Models*, 2024 (arXiv 2411.03538) | Long inputs | Across 20 LLMs, few stay accurate beyond 64k tokens; our longest whole-paper call is 72k tokens. |
| Yu et al., *In Defense of RAG in the Era of Long-Context Language Models*, 2024 (arXiv 2409.01666) | Funnel vs whole paper | Selecting and ordering relevant chunks beats full-context input with fewer tokens, the same argument as the funnel against whole-paper prompting. |
| Ma et al., *Large Language Model Is Not a Good Few-shot Information Extractor, but a Good Reranker for Hard Samples!*, Findings of EMNLP 2023 (doi 10.18653/v1/2023.findings-emnlp.710) | Cascades; Stage B to C | A small fine-tuned model filters, then an LLM decides on the hard cases: the closest analogue to Stage B followed by Stage C. |
| Chen et al., *FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*, 2023 (arXiv 2305.05176) | Cascades; cost | The standard reference for LLM cascades that cut cost; places the funnel among cascade designs. |
| Wu et al., *Less is More for Long Document Summary Evaluation by LLMs*, Journal of Natural Language Processing 2024 (doi 10.5715/jnlp.31.1389) | Funnel vs whole paper | Extract key sentences first, then evaluate with the LLM: lower cost and better agreement with humans on long documents. |
| Hu and Wan, *Mining and Analyzing the Future Works in Scientific Articles*, 2015 (arXiv 1507.02140) | Future-work extraction | Early rule-based future-work sentence extraction and typing; missing from the future-work line of work. |
| Azher et al., *LimTopic: LLM-based Topic Modeling and Text Summarization for Analyzing Scientific Articles Limitations*, JCDL 2024 (doi 10.1145/3677389.3702605) | Limitation analysis | LLM topic modelling of limitation sections, same group as LimGen (already cited). |
| Si et al., *Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers*, 2024 (arXiv 2409.04109) | Introduction; idea generation | Motivates using mined gaps for idea generation: LLM ideas were judged more novel but less feasible than expert ideas. |
| *MOLE: Metadata Extraction and Validation in Scientific Papers Using LLMs*, Findings of EMNLP 2025 (doi 10.18653/v1/2025.findings-emnlp.655) | Long inputs; scholarly extraction | LLM extraction from full scientific papers, including an analysis of context length. |
| *Research Gap Discovery Dataset: LLM-Extracted Limitations and Research Gaps from 3,326 Scientific Papers*, Mendeley Data 2026 (doi 10.17632/px9xd7tw8n) | Datasets; LLM labels | LLM-extracted limitations and gaps from arXiv papers, a resource close to our gold corpus. A dataset record, not a paper: check what it contains and how it was labelled before citing. |

## Optional, if the cascade discussion grows

- *CascadeFilter: Cost-Efficient Semantic Filtering with Joint Threshold Selection*, 2026 (arXiv 2611.08367): small-LLM
  cascade with jointly chosen thresholds, minimising tokens.
- *Streaming Model Cascades for Semantic SQL*, 2026 (arXiv 2604.00660): proxy-then-LLM routing with precision and
  recall guarantees.
- *PrecLLM*, 2024 (arXiv 2412.02868): regex and retrieval pre-filtering lets small local LLMs extract from long
  clinical notes, matching our small-model finding.
- *Retrieval-Augmented LLMs for Evidence Localization in Clinical Trial Recruitment from Longitudinal EHR
  Narratives*, 2026 (arXiv 2604.05190): long-context vs retrieval strategies for screening long documents.

## Skip

Several "automated research-gap detection" systems came from outlets whose claims cannot be checked (IJNRD,
Dandao Xuebao, Zenodo-only records such as GapFinder and an AI research decision-support system). Not worth citing.
