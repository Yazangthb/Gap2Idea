# Context-first boolean flag: two-paper GPU 0 probe

After a keep-first `needs_more_context` boolean produced no flags on ten development papers, we changed the prompt to request a context-needed boolean before the provisional keep/drop decision, with a more liberal instruction to flag cases where additional original text might clarify or change the classification. This tests both a wording change and a field-order change and cannot isolate their individual effects. The two papers and candidate set match the earlier probe; the previously used LLM-labeled development papers are not independent or human-validated.

| First-pass layout | Papers | Candidate decisions | Marked as needing context |
|---|---:|---:|---:|
| Target only | 2 | 91 | 0 |
| Previous / target / next | 2 | 91 | 0 |

All four Qwen3.8-27B FP8 responses passed target ID validation. No second pass was triggered, and no adaptive-context performance result can be inferred from this probe. The exact prompt, output schema, inputs, GPU 0 service model, preflight and raw responses are preserved in the adjacent plan and response files. The ten-paper keep-first flag test is separately documented in `../adaptive_context_flag_gpu0_development_2026-11-27/RESULTS.md`.
