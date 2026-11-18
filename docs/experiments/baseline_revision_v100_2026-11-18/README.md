# V100 baseline evidence

Campaign status: see [the status note](../baseline_revision_v100_STATUS.md).
Protocol, launch commands and deviations: [revision protocol](../baseline_revision_v100_2026-11-18.md).
Runner source was frozen at `84ad950`; later commits add plans, documentation and evidence.

| Cohort directory | Models | Test configurations | Planned calls | Frozen plan |
|---|---|---:|---:|---|
| `revision-v100-small-20261118-01` | Qwen3.5 4B, 9B, BF16 GGUF | 36 | 7,416 | [small](../baseline_revision_v100_small_plan.json) |
| `revision-v100-27b-20261118-01` | Qwen3.5 27B, BF16 GGUF | 12 | 951 | [27B](../baseline_revision_v100_27b_plan.json) |
| `revision-v100-legacy38-20261118-01` | Original Qwen3.8-27B-FP8 checkpoint, dequantized to F16 GGUF | 12 | 951 | [historical](../baseline_revision_v100_legacy38_plan.json) |

All new runs use the private Ollama 0.40.16/llama.cpp backend on V100. They are reported separately
from the historical A100/vLLM results; the historical-generation F16 conversion is not a bitwise
reproduction of the original FP8 runtime. The three Qwen3.5 sizes form the same-generation comparison.
Model identities, conversion/checkpoint revisions, checksums, prompts, settings and hardware are in the plans.

Within each cohort, `<model>/dev/` contains development smoke evidence and `<model>/test/` contains
test evidence. Only completed configurations are copied from the server. Each configuration has:

- `.jsonl`: per-call predictions and original sentence keys, all attempts, raw responses, errors,
  token usage, UTC timestamps, client elapsed time and server timing/cache counters.
- `.manifest.json`: immutable configuration and prompt/input/data/code hashes plus model identity.
- `.prompt.txt`: exact system prompt.
- `.summary.json`: requested/answered/failed/unattempted/excluded coverage and attempt-inclusive costs.

`smoke_report.json` records complete development coverage and id checks. `test_report.json` records
the complete attempted test matrix; failed calls remain failed and are counted as not kept for scoring.
A run with failed calls can have all calls recorded while having incomplete answer coverage.
Missing token costs are null, with known subtotals and the number of missing attempts alongside them.

When a cohort finishes, `handover/scores.json` and `handover/RESULTS.md` contain sentence P/R/F1 and
95% paper-bootstrap intervals (2,000 draws, seed 0). SAL AS scores include all 7,968 sentences and
205 limitations, including the excluded sentences as not kept. Identity-matched P2 funnel exports
can be supplied to `baseline_report.py --comparisons`; differences are funnel minus baseline.

`_snapshots/` lists SHA-256 hashes of every completed file at each transfer. Git attributes disable
line-ending conversion for this evidence tree so those byte hashes survive Windows/Linux checkouts.
Old verdict files outside this directory are preserved.
