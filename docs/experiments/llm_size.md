# Funnel vs whole-paper prompting with a smaller Stage C model (2026-11-07)

Question: does the funnel's advantage over whole-paper prompting grow when Stage C uses a smaller LLM?

Models: Qwen3.8-27B-FP8 (the paper's Stage C) and Qwen3.5-4B (bf16, rev 851bf6e8), each served with vLLM, temperature 0,
seed 0, thinking off, one call at a time. Whole paper = task 5a setup A (its script unchanged, same prompts, 4,096-token
answer limit). Funnel setups: **published** (zero-shot NLI, 20 candidates per call with context), **FT 20/call**
(fine-tuned NLI, same Stage C), **FT per paper** (fine-tuned NLI, all of a paper's candidates in one call, ids-only
answer). F1 differences are paired, resampling papers (2,000 draws). arXiv = the 28 arXiv gold test papers (LLM
labels); SAL-40 and FWS have human labels.

## F1, funnel minus the same model's whole-paper run

| Set | Mode | 27B whole | 27B best funnel (diff) | 4B whole | 4B published (diff) | 4B FT per paper (diff) |
|---|---|---|---|---|---|---|
| arXiv | high recall | 0.741 | 0.691 per paper (-0.050) | **0.275** | 0.480 (**+0.205**) | 0.530 (**+0.255**) |
| arXiv | balanced | 0.749 | 0.734 FT 20/call (-0.015) | **0.314** | 0.524 (**+0.210**) | 0.460 (+0.146, CI -0.018 to +0.234) |
| arXiv | high precision | 0.679 | 0.687 FT 20/call (+0.008) | **0.296** | 0.544 (**+0.248**) | 0.484 (**+0.187**) |
| SAL-40 | high recall | 0.663 | 0.677 published (+0.014) | 0.599 | 0.576 (-0.023) | 0.697 (**+0.098**) |
| SAL-40 | balanced | 0.673 | 0.716 published (+0.043) | 0.618 | 0.611 (-0.007) | 0.708 (**+0.090**) |
| SAL-40 | high precision | 0.702 | 0.744 published (+0.042) | 0.661 | 0.650 (-0.012) | 0.711 (+0.050, CI -0.008 to +0.116) |
| FWS | high recall | 0.694 | 0.755 published (**+0.060**) | 0.719 | 0.731 (+0.013) | 0.741 (+0.023) |
| FWS | balanced | 0.694 | 0.764 FT 20/call (**+0.069**) | 0.719 | 0.711 (-0.008) | 0.721 (+0.003) |
| FWS | high precision | 0.694 | 0.760 published (**+0.065**) | 0.719 | 0.727 (+0.009) | 0.725 (+0.006) |

Bold differences have a 95% CI excluding zero. The 4B FT 20/call results are in `docs/experiments/llm_size/results.json`
(arXiv +0.24 to +0.29, SAL-40 -0.03 to 0.00, FWS -0.03 to -0.01).

## What happens to the 4B on whole papers

- **arXiv (median 239 sentences, up to 72k tokens):** precision 18-23%, recall 51-54%. On long papers the 4B keeps
  almost everything (493 of 523 sentences, 490 of 692), and on the longest paper (1,526 sentences, 83 gaps) its answer
  ran past the 4,096-token limit in all three modes, so that call failed and its gaps count as missed. The 27B
  reached F1 0.68-0.75 on the same calls.
- **SAL-40 (about 200 sentences, at most 16k tokens):** the 4B drops only 0.04-0.06 F1 from the 27B.
- **FWS (conclusion excerpts of about 6 sentences):** no drop; the 4B is slightly better than the 27B (0.719 vs 0.694).

So the effect grows with input length: on short inputs the small model is fine on its own, on full arXiv papers it
breaks down, and the funnel, which never sends more than a paper's candidates, keeps it usable.

## Cost with the 4B (summed call time, one call at a time)

| Set | Whole paper: input / output / seconds | FT per paper: input / output / seconds |
|---|---|---|
| arXiv | 480k / 5-6.5k / 152-162 s | 73-75k / 2.1-2.5k / 24-27 s |
| SAL-40 | 346k / 1.5-1.9k / 30-38 s | 40-42k / 0.9k / 20 s |
| FWS | 183k / 2.1k / 72 s | 60-67k / 1.2k / 47 s |

The 4B's 20-per-call answers sometimes miss an item (19 of 20 verdicts, 1-2 batches per mode on arXiv); the runner
then asks for the missing items alone, as LLMGapFilter does. No verdict was left missing.

Files: verdicts in `docs/experiments/llm_size/qwen3.5-4b/{whole_paper,funnel_zs_batch20,funnel_ft_batch20,per_paper_ft}/`;
scripts `scripts/bench/whole_paper_other_llm.py`, `funnel_nli_ft.py` and `stage_c_per_paper_ft.py` (both with
`QWEN_MODEL`, `STAGE_B`, `FUNNEL_OUTDIR`, `PERPAPER_OUTDIR`), `llm_size_compare.py` (this table).
