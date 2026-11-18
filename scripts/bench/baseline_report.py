"""Score a revision run's saved predictions on ALL sentences and export a handover.

Paired comparison inputs are optional JSON records with dataset, model_identity, mode,
input_type, and kept [[paper_id, idx], ...]. Model identity must equal the run manifest's
identity, so historical vLLM/FP8 verdicts cannot silently become V100/BF16 comparisons.
"""
import argparse
import json
from pathlib import Path

import baseline_qwen as b
from baseline_evidence import digest


MODES = dict(keep_biased="high_recall", neutral="balanced", strict="high_precision")


def report(root, comparisons=None):
    funnels = json.loads(comparisons.read_text(encoding="utf-8")) if comparisons else []
    result, rows_by_dataset = [], {}
    for path in sorted(root.glob("*/test/*.jsonl")):
        manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        dataset = manifest["dataset"]
        if dataset not in rows_by_dataset:
            rows_by_dataset[dataset] = b.load_rows(dataset, "test")
        rows = rows_by_dataset[dataset]
        if digest(rows) != manifest["data_sha256"]:
            raise ValueError(f"Data changed since inference: {path}")
        kept, stats = b.score_run(rows, path)
        mode, setup = MODES[manifest["prompt"]], manifest["setup"]
        row = dict(dataset=dataset, model=manifest["model"], model_identity=manifest["model_identity"],
                   mode=mode, input_type=setup, predictions=str(path.relative_to(root)), **stats,
                   funnel_minus_baseline=[])
        baseline_complete = stats["coverage"]["all_calls_recorded"]
        for funnel in funnels:
            if (funnel["dataset"], funnel["mode"], funnel["input_type"], funnel["model_identity"]) != (
                    dataset, mode, setup, manifest["model_identity"]):
                continue
            if not baseline_complete:
                continue  # Never treat unattempted calls as negative predictions.
            fk = {tuple(k) for k in funnel["kept"]}
            if not fk <= {(r["pid"], r["idx"]) for r in rows}:
                raise ValueError("Funnel predictions contain keys outside the evaluation universe")
            diff = b.boot(rows, fk, kept)  # Always funnel minus baseline, on identical sentences.
            row["funnel_minus_baseline"].append(dict(name=funnel["name"], **diff,
                interpretation="difference not resolved" if diff["ci95"][0] <= 0 <= diff["ci95"][1]
                else "interval excludes zero"))
        result.append(row)
    out = root / "handover"
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Baseline evidence", "", "All evaluation sentences are scored; SAL AS exclusions and failed calls count as not kept.",
             "INCOMPLETE rows contain unattempted calls and are provisional; paired differences are suppressed.",
             "Token totals include every attempt. A missing total has a known subtotal in scores.json.",
             "Model/backend differences from historical experiments are retained in the manifests.", "",
             "| Model | Dataset | Input | Mode | Status | P | R | F1 (95% CI) | Answered/requested | Failed calls | Input/output tokens | Seconds |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in result:
        p, r, f = row["PRF"]
        lo, hi = row["ci95"]["f1"]
        c = row["coverage"]
        tokens = lambda v: str(v) if v is not None else "missing"
        status = "complete" if c["all_calls_recorded"] else "INCOMPLETE"
        lines.append(f"| {row['model']} | {row['dataset']} | {row['input_type']} | {row['mode']} | {status} | {p:.3f} | {r:.3f} | "
                     f"{f:.3f} ({lo:.3f}, {hi:.3f}) | {c['answered_sentences']}/{c['requested_sentences']} | "
                     f"{c['failed_calls']} | {tokens(row['input_tokens'])}/{tokens(row['output_tokens'])} | {row['seconds']:.1f} |")
    paired = [(row, diff) for row in result for diff in row["funnel_minus_baseline"]]
    if paired:
        lines += ["", "## Paired funnel minus baseline", "",
                  "| Model | Dataset | Baseline input | Mode | Funnel | F1 difference (95% CI) | Interpretation |",
                  "|---|---|---|---|---|---|---|"]
        for row, diff in paired:
            low, high = diff["ci95"]
            lines.append(f"| {row['model']} | {row['dataset']} | {row['input_type']} | {row['mode']} | "
                         f"{diff['name']} | {diff['f1_diff']:+.3f} [{low:+.3f}, {high:+.3f}] | "
                         f"{diff['interpretation']} |")
    lines += ["", "Paired funnel differences require an identity-matched funnel export from the scoring owner.",
              "No comparison is substituted from the older PubMedBERT SAL funnel or a different Qwen generation."]
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--comparisons", type=Path)
    args = ap.parse_args()
    scored = report(args.root, args.comparisons)
    print(f"Scored {len(scored)} configurations; saved handover in {args.root / 'handover'}")
