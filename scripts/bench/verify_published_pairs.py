"""Recompute the nine original Qwen3.8-27B funnel-vs-whole-paper results.

Uses reconstructed SAL/FWS splits and saved historical verdicts only; no LLM
calls. The old FWS whole-paper prompt differs from the funnel's three prompts.
"""
import json
from pathlib import Path

import baseline_qwen as baseline
import funnel_nli_ft as funnel


ROOT = Path(__file__).resolve().parents[2]


def reconstructed_rows():
    out = {}
    for name in ("sal_dev", "sal_test", "fws_dev", "fws_test"):
        path = ROOT / "data/splits" / f"{name}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        out[name] = [dict(pmcid=r["paper_id"], sent=r["sentence"], label=r["label"],
                          **({"sec": r["section"]} if "section" in r else {})) for r in rows]
    return out


def main():
    funnel.STAGE_B = "zero_shot"
    funnel.BASELINE_DIR = ROOT / "docs/experiments/baseline_qwen"
    funnel.load_all = reconstructed_rows
    results = json.loads((ROOT / "docs/experiments/llm_size/results.json").read_text(encoding="utf-8"))
    for name in ("gold", "sal", "fws"):
        source = funnel.SETS[name]()
        rows = baseline.load_rows(name, "test")
        universe = {(row["pid"], row["idx"]) for row in rows}
        for mode in ("high_recall", "balanced", "high_precision"):
            kept = funnel.published(name, source["rows"], mode) & universe
            whole, _ = funnel.baseline(name, mode)
            difference = baseline.boot(rows, kept, whole & universe)
            expected = results[f"{name}/{mode}/27b"]["published"]["minus_whole"]
            if difference != expected:
                raise SystemExit(f"Mismatch for {name}/{mode}: {difference} != {expected}")
            print(f"OK {name}/{mode}: {difference}", flush=True)


if __name__ == "__main__":
    main()
