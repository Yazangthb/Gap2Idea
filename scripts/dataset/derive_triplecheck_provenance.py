"""Derive requested-model-by-item provenance from a specific recovered run.

This is an inference from the captured producing script and run summary, not
per-request response metadata. See TRIPLECHECK_PROVENANCE_2026-11-18.md.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/gold/extras_triplecheck.json"
OUT = ROOT / "data/gold/extras_triplecheck_requested_model_inferred.json"
EXPECTED = "d04788241e32ace99cd8d79f0b85cb081330e1d245053a585628a7dd354ea20f"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != EXPECTED:
        raise SystemExit("Source differs from the audited 227-item file; refusing to infer provenance")
    if data["model"] != "claude-fable-5-1" or len(data["verdicts"]) != 227:
        raise SystemExit("Unexpected source metadata")
    result = {"status": "inferred, not recorded per request", "requested_model": data["model"],
              "basis": "captured script permits fallback only on first batch; run log and saved file both name Fable",
              "limitation": "raw responses unavailable; the script defaulted missing response keys to false",
              "source_canonical_sha256": EXPECTED,
              "requested_model_by_item": {key: data["model"] for key in sorted(data["verdicts"])}}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote inferred requested-model provenance for {len(data['verdicts'])} items")


if __name__ == "__main__":
    main()
