# Triple-check provenance audit

The recovered `data/gold/extras_triplecheck.json` contains 227 keyed Boolean verdicts (116 true) and one file-level `model` field naming `claude-fable-5-1`. It does not contain model identifiers or raw responses for individual items.

## Recovered execution evidence

The local Claude session `8966bf32-f3c8-4831-b391-51a114f93c7e` contains the producing script and its output from 18 October 2026. The script starts with Fable, permits a switch to Opus **only when the first batch fails**, then keeps that model for all following batches and saves the final model name to the file. The captured run log shows all 16 batches and a final `claude-fable-5-1` result of 116/227, with no fallback message; the recovered file reports the same model and totals. Taken together, this supports an inference that all batches in this run used Fable, assuming the recovered file is the file produced by the captured run. It is not a per-request identity log or proof that every response was complete.

The script also converts a missing response key to `false`, and individual raw model responses were not preserved in the recovered file. Therefore a `false` entry may reflect a missing answer rather than a clear rejection. Keep the original file unchanged, label the inferred model as inferred rather than a recorded per-item fact, and do not describe the 116 as human confirmation. The implication for any 116/227 claim needs author review.

`extras_triplecheck_requested_model_inferred.json` supplies the same qualified inference for each of the 227 input IDs. Regenerate it with `python scripts/dataset/derive_triplecheck_provenance.py`; the script refuses a different canonical source file. It records the model to which the producing script sent each item, conditional on the recovered-file linkage above. It does not assert that each item received a valid response.
