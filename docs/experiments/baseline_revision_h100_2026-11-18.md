# Prospective H100 extension, 18 November 2026

The user authorized releasing the two idle Qwen H100 reservations and using the H100s for this work.
The pending 27B comparisons move from the V100 queue to H100. The existing V100 4B/9B runs continue.
The user explicitly retained additional 4B/9B H100 controls so the three Qwen3.5 sizes can also be
compared under one native vLLM backend. No V100 prediction is overwritten or relabelled as H100 output.

The frozen [H100 plan](baseline_revision_h100_plan.json) registers 60 test configurations and 9,318 calls:
Qwen3.5 4B/9B/27B and the original Qwen3.8-27B-FP8 control. It uses the same data, prompts, A/AS/C
definitions, temperature 0, seed 0, thinking off, output budgets and two retries as the V100 plan.
All 120 development/test configuration input, prompt and data hashes match their corresponding
V100 configurations. Runner Python source remains `84ad950`; 29 tests pass on the H100 host too.

All native checkpoint files are pinned to immutable Hugging Face revisions and checksum-verified.
The three Qwen3.5 sizes use unquantized BF16, vLLM 0.32.0, context cap 81,920 and no speculation.
The original Qwen3.8 checkpoint is revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, native FP8,
with its original 262,144 context cap and MTP2 settings. Exact serving arguments and the Docker image
digest are in the plan. GPU 1 retains an unrelated approximately 1 GiB Python process; it is not stopped.

Execution order, one request at a time on each GPU:

- H100 GPU 0: Qwen3.5-27B, then the Qwen3.5-4B control.
- H100 GPU 1: Qwen3.8-27B-FP8, then the Qwen3.5-9B control.

Each model passes the first-two-development-papers smoke matrix before test inference. The plan is
committed before either phase. All failed/retried calls remain in the evidence; unknown costs are null.
All sentences are scored, including SAL AS exclusions, with 2,000 paper bootstrap draws, seed 0.
Paired differences use funnel minus baseline. H100/vLLM and V100/Ollama rows remain separate.

The unchanged runner resumes by skipping durable saved unit ids. `campaign_control.py pause` stops
only this campaign's clients and model servers, preserves all saved calls, and writes a versioned
checkpoint under the server's `checkpoints/` directory. A cancelled in-flight call may repeat on resume;
its unreported cost is explicitly missing, not zero. If a trailing partial JSONL line needs repair,
the original bytes are preserved before creating the valid resumable prefix. Non-tail corruption aborts.
`resume` refuses duplicate runners or GPUs occupied by another workload, then uses the same frozen plan.

The orchestration and pause/resume source are archived with checksums under
`baseline_revision_h100_2026-11-18/_runtime/`. No credentials or benchmark sentence text are included.
