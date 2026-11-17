"""Prepare, freeze and execute the MAKE revision's matched baseline matrix.

prepare only reads data/models and writes a plan: it never makes inference calls.
Commit the plan after development smoke tests and before run --split test.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import baseline_qwen as b
from baseline_evidence import OllamaClient, code_hashes, digest, fingerprint, git


def grid(model):
    # Full-paper controls for all three datasets, plus the SAL Stage A control.
    configs = [(d, "A", p) for d in ("fws", "sal", "gold") for p in b.RUBRIC_NAMES]
    configs += [("sal", "AS", p) for p in b.RUBRIC_NAMES]
    if any(size in model.lower() for size in ("4b", "9b")):
        configs += [(d, "C", p) for d in ("gold", "sal") for p in b.RUBRIC_NAMES]
    return configs


def prepare(args):
    if args.plan.exists():
        raise SystemExit("Plan already exists; choose a fresh plan and output root")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise SystemExit("Output root must be fresh")
    client = OllamaClient(args.base_url, args.num_ctx)
    models = {m: client.identity(m) for m in args.models}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
                         check=True, capture_output=True, text=True).stdout.strip()
    if "V100" not in gpu:
        raise SystemExit("This experiment was requested on V100 hardware")
    runs, outputs, data_cache, prompt_cache = [], {}, {}, {}
    for model in args.models:
        for split, limit in (("dev", 2), ("test", 0)):
            for dataset, setup, prompt in grid(model):
                opts = NS(model=model, backend="ollama", num_ctx=args.num_ctx, dataset=dataset,
                          setup=setup, prompt=prompt, split=split, limit_papers=limit)
                if (dataset, split) not in data_cache:
                    data_cache[dataset, split] = b.load_rows(dataset, split)
                rows = data_cache[dataset, split]
                if limit:
                    keep = set(list(b.papers_of(rows))[:limit])
                    rows = [r for r in rows if r["pid"] in keep]
                units = list(b.units(rows, setup))
                if (dataset, prompt) not in prompt_cache:
                    prompt_cache[dataset, prompt] = b.build_prompt(dataset, prompt)
                runs.append(fingerprint(opts, rows, units, prompt_cache[dataset, prompt]))
                outputs[digest(runs[-1])] = dict(calls=len(units), sentences=len(rows),
                                                 requested_sentences=sum(len(u["keys"]) for u in units))
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), code_commit=git("rev-parse", "HEAD"),
                code_hashes=code_hashes(), models=models, runs=runs, expected=outputs,
                hardware=gpu, server_command=args.server_command,
                weight_provenance=json.loads(args.weight_provenance.read_text(encoding="utf-8")) if args.weight_provenance else None,
                prepare_command=[sys.executable, *sys.argv],
                output_root=args.output_root.as_posix(),
                protocol="X1 matched FWS A/SAL A+AS; full-paper gold; X2 gold/SAL C for 9B and 4B. "
                         "All sentences scored, including AS exclusions. No test tuning. "
                         "P/R/F1, 2000 paper bootstrap draws seed 0; funnel minus baseline, paired by model/mode/input. "
                         "Sequential calls per dedicated GPU, two retries; failed sentences not kept, unknown costs null. "
                         "New V100/Ollama backend is reported separately from historical A100/vLLM results.")
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    with args.plan.open("x", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=2)
    print(json.dumps(dict(plan=str(args.plan), configurations=len(runs) // 2,
                          test_calls=sum(outputs[digest(r)]["calls"] for r in runs if r["split"] == "test")), indent=2))


def run(args):
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    root = Path(plan["output_root"])
    selected = args.models or list(plan["models"])
    if not set(selected) <= set(plan["models"]):
        raise SystemExit("Selected model is outside the frozen plan")
    if args.split == "test":
        for model in selected:
            smoke_path = root / model.replace(":", "-").replace("/", "-") / "smoke_report.json"
            if not smoke_path.exists():
                raise SystemExit(f"Run the development smoke matrix first: {model}")
            smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
            if (smoke["plan_sha256"] != digest(plan) or not smoke["passed"] or
                    smoke["configurations"] != len(grid(model))):
                raise SystemExit("Development smoke matrix failed, is incomplete, or used a different plan")
    summaries = []
    for spec in plan["runs"]:
        if spec["split"] != args.split or spec["model"] not in selected:
            continue
        model_dir = spec["model"].replace(":", "-").replace("/", "-")
        opts = {k: spec[k] for k in ("dataset", "setup", "prompt", "split", "limit_papers", "model", "backend", "num_ctx")}
        opts.update(output_dir=root / model_dir / args.split, freeze_plan=args.plan, base_url=args.base_url)
        b.run(NS(**opts))
        path = opts["output_dir"] / f"{spec['dataset']}_{args.split}_{spec['setup']}_{spec['prompt']}.summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        records = b._jsonl(path.with_name(path.name.replace(".summary.json", ".jsonl")))
        summary["invalid_ids"] = sum(len(r.get("invalid_ids", [])) for r in records)
        summaries.append(summary)
        if summary["coverage"]["failed_calls"] == summary["coverage"]["expected_calls"]:
            raise SystemExit("Every call in this configuration failed; inspect evidence before continuing")
    for model in selected:
        subset = [s for s in summaries if s["manifest"]["model"] == model]
        report = dict(plan_sha256=digest(plan), configurations=len(subset),
                      passed=bool(subset) and all(s["coverage"]["every_requested_sentence_answered"] and not s["invalid_ids"] for s in subset),
                      summaries=subset)
        report_path = root / model.replace(":", "-").replace("/", "-") / (
            "smoke_report.json" if args.split == "dev" else "test_report.json")
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Saved {report_path}; complete coverage: {report['passed']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--num-ctx", type=int, default=131072)
    p.add_argument("--base-url")
    p.add_argument("--server-command", required=True, help="exact launch command, with secrets removed")
    p.add_argument("--weight-provenance", type=Path, help="conversion revisions, file checksums and runtime build evidence")
    r = sub.add_parser("run")
    r.add_argument("--plan", type=Path, required=True)
    r.add_argument("--split", choices=("dev", "test"), required=True)
    r.add_argument("--base-url")
    r.add_argument("--models", nargs="+", help="run a subset of the frozen model matrix, e.g. on a dedicated GPU")
    args = parser.parse_args()
    {"prepare": prepare, "run": run}[args.command](args)


if __name__ == "__main__":
    main()
