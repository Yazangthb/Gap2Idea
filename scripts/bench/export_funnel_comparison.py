"""Export a fully enumerated funnel run for baseline_report.py without inference.

The input manifest must be created from the actual funnel run, not copied from
the baseline: it carries the original model identity and a hash of its full
candidate sequence. No model or verdict provenance is inferred by this tool.
"""
import argparse
import json
from pathlib import Path

import baseline_qwen as baseline
from baseline_evidence import digest


def export(manifest, candidates, verdicts, rows, input_types):
    if manifest.get("split") != "test" or manifest.get("dataset") not in ("gold", "sal", "fws"):
        raise ValueError("Expected a test-set funnel manifest for gold, sal, or fws")
    if manifest.get("mode") not in ("high_recall", "balanced", "high_precision"):
        raise ValueError("Unknown funnel mode")
    if not isinstance(manifest.get("model_identity"), dict) or not manifest["model_identity"]:
        raise ValueError("The funnel's actual model identity is required")
    if manifest.get("evaluation_data_sha256") != digest(rows):
        raise ValueError("Evaluation data differs from the funnel manifest")
    if manifest.get("candidate_keys_sha256") != digest(candidates):
        raise ValueError("Candidate sequence differs from the funnel manifest")
    keys = [tuple(k) for k in candidates]
    universe = {(r["pid"], r["idx"]) for r in rows}
    key_set = set(keys)
    if len(keys) != len(key_set) or not key_set <= universe:
        raise ValueError("Duplicate or out-of-universe funnel candidate")
    recorded = {}
    for verdict in verdicts:
        key = tuple(verdict["key"])
        if key in recorded or key not in key_set or not isinstance(verdict["keep"], bool):
            raise ValueError("Duplicate, foreign, or invalid funnel verdict")
        recorded[key] = verdict["keep"]
    if set(recorded) != key_set:
        raise ValueError("Funnel verdicts do not cover all candidates")
    kept = [list(key) for key in keys if recorded[key]]
    return [dict(name=manifest["name"], dataset=manifest["dataset"], mode=manifest["mode"],
                 input_type=kind, model_identity=manifest["model_identity"], kept=kept,
                 funnel_manifest_sha256=digest(manifest), candidate_keys_sha256=digest(candidates),
                 funnel_verdicts_sha256=digest(verdicts)) for kind in input_types]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path, help="JSON array of [paper, idx] keys")
    parser.add_argument("--verdicts", required=True, type=Path, help="JSONL records with key and keep")
    parser.add_argument("--input-types", nargs="+", choices=("A", "AS", "B", "C"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    verdicts = [json.loads(line) for line in args.verdicts.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = baseline.load_rows(manifest["dataset"], "test")
    records = export(manifest, candidates, verdicts, rows, args.input_types)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Exported {len(records)} comparisons; {len(candidates)} candidate verdicts checked.")


if __name__ == "__main__":
    main()
