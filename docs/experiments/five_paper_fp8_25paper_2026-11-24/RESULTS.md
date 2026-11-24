# Matched five-paper requests on 25 papers: FP8 comparison

## Setup

Twenty-five arXiv papers were selected without using labels by shuffling the 28 existing arXiv test IDs with `random.Random(20261124)` and taking the first 25. The frozen plan records all groups and the three unused papers. Five groups of five were formed; each method was evaluated once per group. The 25 papers contain 9,843 sentences and 477 LLM-labeled gaps. These are previously studied test papers, not a new blind test, and the labels have not been human validated.

All 15 calls used Qwen3.8-27B-FP8 (`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`) on the existing H100 vLLM service, the same neutral prompt, paper boundaries, original group-wide sentence IDs, schema, seed 0, temperature 0, no thinking and 8,192 output-token cap. The full-paper method sent every sentence in each group. Both funnel variants used Stage A cleanup, then either the zero-shot NLI threshold 0.3541100135 or fine-tuned NLI threshold 0.0021033690, and sent all candidates in a single request with the previous original sentence (last 800 characters) and next original sentence (first 800 characters). No labels were sent. Full-paper context and shortlist-with-neighbors necessarily differ in amount and arrangement; the prompt was selected for the funnel on development data, not separately optimized for the full-paper baseline.

All inputs passed the model-context preflight. All 15 calls returned valid output on the first attempt, with no duplicate IDs. The paper-level data, NLI test-score arrays, code, frozen plan, per-call results and token counts are recorded in the repository. The precomputed NLI scores and labels were taken from the project test split; each method uses the same five-paper groups and evaluation denominator.

## Aggregate results

| Method | Qwen calls | Target sentences passed | TP | FP | Precision | Recall | F1 | Qwen input tokens | Qwen output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full five-paper requests | 5 | 9,843 | 236 | 27 | 89.7% | 49.5% | 0.638 | 522,384 | 1,520 |
| Zero-shot NLI + neighbors | 5 | 1,994 | 377 | 308 | 55.0% | 79.0% | 0.649 | 266,134 | 4,656 |
| Fine-tuned NLI + neighbors | 5 | 1,553 | 367 | 263 | 58.3% | 76.9% | 0.663 | 210,996 | 4,486 |

Recall is measured against all 477 labeled gaps, including those removed by Stage A or NLI. Zero-shot NLI sent 427 labeled gaps to Qwen (89.5% candidate recall, 21.4% candidate precision); fine-tuned NLI sent 434 (91.0% candidate recall, 27.9% candidate precision). These are NLI-stage metrics before Qwen. Despite higher candidate recall, fine-tuned NLI + Qwen achieved 10 fewer final true positives than the zero-shot pipeline, highlighting that changing a shortlist can change the one-call model's decisions.

Compared with full papers, zero-shot and fine-tuned funnels used 49.1% and 59.6% fewer Qwen input tokens respectively. The fine-tuned funnel used 20.7% fewer Qwen input tokens than the zero-shot funnel. These are Qwen input-token counts, not end-to-end cost: NLI/cleanup, extra output tokens, full data extraction, overhead and hardware costs are not included. The summed call durations (88.182, 70.191 and 60.380 seconds) reflect this specific serving session, including possible prefix caching; they do not include preprocessing and are not a controlled throughput benchmark.

## Results by five-paper group

| Group | Labeled gaps | Full P / R / F1 | Zero-shot P / R / F1 | Fine-tuned P / R / F1 |
| --- | ---: | --- | --- | --- |
| 1 | 85 | 93.3% / 49.4% / 0.646 | 78.6% / 77.6% / 0.781 | 69.8% / 87.1% / 0.775 |
| 2 | 77 | 85.0% / 44.2% / 0.581 | 57.1% / 72.7% / 0.640 | 60.6% / 81.8% / 0.696 |
| 3 | 84 | 100% / 47.6% / 0.645 | 75.3% / 69.0% / 0.721 | 71.6% / 63.1% / 0.671 |
| 4 | 148 | 93.6% / 49.3% / 0.646 | 51.8% / 86.5% / 0.648 | 42.3% / 74.3% / 0.539 |
| 5 | 83 | 78.3% / 56.6% / 0.657 | 38.6% / 83.1% / 0.527 | 77.9% / 80.7% / 0.793 |

The methods show a precision–recall trade-off: full-paper prompting has higher measured precision and lower recall; the fine-tuned funnel has slightly higher aggregate precision and F1 than zero-shot, but slightly lower recall. The fine-tuned precision varies from 42.3% to 77.9% by group. This is descriptive, with only five shared-context groups; the differences should not be interpreted as a general win or statistical significance. For the LLM-generated labels, many labeled non-gaps may require adjudication. A new independent paper set and blind human review of selected and nonselected sentences are necessary before publication-grade claims.
