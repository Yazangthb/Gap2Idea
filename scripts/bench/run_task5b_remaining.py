"""Prepare/check/run the remaining data-ready Task 5b work, without pushing Git.

No flags: check only. --run executes independent registered jobs sequentially.
Completed, verified jobs are skipped. Incomplete jobs are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.experiments.task5b.data import (
    digest, load_prepared, read_jsonl, validate, write_json, write_jsonl)
from gap2idea.experiments.task5b.runner import main as task5b, fit, evaluate, provenance
from gap2idea.experiments.task5b.secondary import score_external

DATA = ROOT / "data/task5b/prepared-current-gold-v1"
SAL_DATA = ROOT / "data/task5b/prepared"
REVISIONS = {
    "scibert_ft": "24f92d32b1bfb0bcaf9ab193ff3ad01e87732fc1",
    "biomedbert_ft": "e1354b7a3a09615f6aba48dfad4b7a613eef7062"}
JOBS = {
    "arxiv-baselines": ("arxiv", "tfidf_lr,binary_nb,modernbert_nli,modernbert_nli_typed"),
    "arxiv-scibert": ("arxiv", "scibert_ft"),
    "arxiv-biomedbert": ("arxiv", "biomedbert_ft"),
    "fws-scibert": ("fws", "scibert_ft"),
    "sal-cached-encoders": ("sal", "legacy_scibert_ft,legacy_pubmedbert_ft")}
CACHED = {"legacy_scibert_ft": "SciBERT-FT", "legacy_pubmedbert_ft": "PubMedBERT-FT"}


def absolute(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def data_root(dataset):
    return SAL_DATA if dataset == "sal" else DATA


def verify_completed(run, dataset, methods):
    """Check existing evidence before skipping it; never score a test again."""
    run = Path(run)
    if not run.exists():
        return False
    if not (run / "test_complete.json").exists():
        raise ValueError(f"Incomplete run exists: {run}. Preserve it; retry this stage with a NEW --run-root.")
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if meta["dataset"] != dataset or set(meta["requested_methods"]) != set(methods.split(",")):
        raise ValueError(f"Existing run configuration differs: {run}")
    _, current_manifest = load_prepared(data_root(dataset) / dataset, dataset)
    if json.loads((run / "input_manifest.json").read_text(encoding="utf-8")) != current_manifest:
        raise ValueError(f"Existing run used a different prepared dataset: {run}")
    settings = json.loads((run / "fit_complete.json").read_text(encoding="utf-8"))["settings_sha256"]
    results = json.loads((run / "results.json").read_text(encoding="utf-8"))
    if set(settings) != set(methods.split(",")) or set(results) != set(settings):
        raise ValueError(f"Existing run did not finish every requested method: {run}")
    for method, sha in settings.items():
        directory = run / method
        if digest(directory / "frozen_settings.json") != sha:
            raise ValueError(f"Frozen settings changed: {directory}")
        spec = json.loads((directory / "frozen_settings.json").read_text(encoding="utf-8"))
        for file, expected in spec["artifact_sha256"].items():
            if digest(directory / file) != expected:
                raise ValueError(f"Saved artifact changed: {directory / file}")
        metric = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metric != results[method] or digest(directory / "test_predictions.jsonl") != metric["predictions_sha256"]:
            raise ValueError(f"Saved result/prediction changed: {directory}")
    return True


def cached_records(test, fixture, source_model):
    if len(test) != len(fixture):
        raise ValueError("Cached SAL vectors and frozen SAL test lengths differ")
    indices = {}
    records = []
    for r, f in zip(test, fixture):
        pid = str(f["pmcid"])
        idx = indices.get(pid, 0)
        indices[pid] = idx + 1
        if (r["paper_id"], r["idx"], r["sentence"], r["y"]) != (pid, idx, f["sent"], int(f["label"])):
            raise ValueError(f"Cached SAL identity mismatch: {pid}/{idx}")
        pred = f["preds"][source_model]
        if type(pred) is not int or pred not in (0, 1):
            raise ValueError(f"Cached SAL prediction is not binary: {pid}/{idx}")
        records.append(dict(paper_id=pid, idx=idx, text_sha256=r["text_sha256"], y_pred=pred))
    return records


def import_cached_sal(output, bootstrap):
    fixture_path = ROOT / "stage_c_package/sal_criteria/sal_test_allmodels.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    split, manifest = load_prepared(SAL_DATA / "sal", "sal")
    if output.exists():
        marker = output / "stage_complete.json"
        if not marker.exists():
            raise ValueError(f"Incomplete cached import: {output}; use a new --run-root")
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state["fixture_sha256"] != digest(fixture_path):
            raise ValueError("Original cached fixture changed")
        if json.loads((output / "input_manifest.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Cached import was evaluated against another prepared SAL manifest")
        for relative, expected in state["evidence_sha256"].items():
            if digest(output / relative) != expected:
                raise ValueError(f"Cached import evidence changed: {relative}")
        print("SKIP verified completed sal-cached-encoders", flush=True)
        return
    output.mkdir(parents=True)
    for method, original_id in CACHED.items():
        raw, protocol = output / f"{method}_input.jsonl", output / f"{method}_protocol.json"
        write_jsonl(raw, cached_records(split["test"], fixture, original_id))
        write_json(protocol, dict(method=method,
            training_data="Legacy author-supplied SAL fine-tune predictions; original exact recipe not independently verified",
            threshold_selection="Original cached 0/1 decisions preserved; no threshold selected in this import; original selection history not verified",
            model_or_code_revision=dict(fixture=str(fixture_path.relative_to(ROOT)),
                fixture_sha256=digest(fixture_path), original_prediction_key=original_id),
            limitations="Legacy full-stream encoder (NO Stage A). Not a new validation-controlled fine-tune, not A+B funnel, not Lan EDA/PBA; report provenance separately."))
        score_external(argparse.Namespace(data_dir=str(SAL_DATA), dataset="sal", method=method,
            target="limitation", predictions=str(raw), protocol=str(protocol),
            output_dir=str(output / method), bootstrap=bootstrap, seed=0))
    write_json(output / "input_manifest.json", manifest)
    evidence = {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}
    write_json(output / "stage_complete.json", dict(fixture_sha256=digest(fixture_path), evidence_sha256=evidence))


def export_reference_requests(output):
    if output.exists():
        raise FileExistsError(f"Reference request directory exists: {output}; choose a new output path")
    # Check every data source before creating any output directory.
    loaded = {d: load_prepared(data_root(d) / d, d) for d in ("sal", "fws", "arxiv")}
    output.mkdir(parents=True)
    for dataset, (split, manifest) in loaded.items():
        directory = output / dataset
        directory.mkdir()
        rows = split["test"]
        label_free = []
        for i, r in enumerate(rows):
            context = " ".join(rows[j]["sentence"] for j in range(max(0, i - 1), min(len(rows), i + 2))
                               if rows[j]["paper_id"] == r["paper_id"])
            label_free.append(dict(paper_id=r["paper_id"], idx=r["idx"], sentence=r["sentence"],
                                   text_sha256=r["text_sha256"], section=r.get("section", ""),
                                   coarse_section=r.get("coarse_section", ""), context=context))
        write_jsonl(directory / "sentences.jsonl", label_free)
        write_json(directory / "request.json", dict(dataset=dataset, papers=len({r["paper_id"] for r in rows}),
            sentences=len(rows), source_test_sha256=manifest["prepared_sha256"]["test"],
            text_transform=manifest.get("notes", {}).get("text_transform", "unchanged canonical sentence")))
        write_json(directory / "protocol_TEMPLATE.json", dict(method=f"REQUIRED_frozen_funnel_{dataset}",
            training_data="REQUIRED: checkpoint training corpus/split, no target test training",
            threshold_selection="REQUIRED: validation-only thresholds and prompt/model selection history",
            model_or_code_revision="REQUIRED: Stage A/B checkpoint, NLI hypothesis if used, Stage C model/prompt revisions",
            limitations="REQUIRED: disclose previous test exposure, unavailable provenance and any deviations"))
    (output / "README.md").write_text(
        "# Funnel prediction request\n\n"
        "These are label-free exact test inputs, not instructions to tune on test. "
        "Use a previously frozen funnel; record its actual provenance. Preserve all paper_id/idx/text hashes. "
        "Return one JSONL row for EVERY input with paper_id, idx, text_sha256 and integer y_pred (0/1), "
        "including Stage-A/B rejected rows as negatives. Fill protocol_TEMPLATE.json with real metadata "
        "and save as protocol.json. Do not submit the template as provenance. "
        "FWS uses the current-gold exclusion protocol and lowercase text. "
        "Do not regenerate different sentences or infer decisions from aggregate F1.\n",
        encoding="utf-8")
    print(f"Wrote label-free requests for all 3 datasets: {output}", flush=True)


def smoke_encoders(args):
    output = absolute(args.smoke_output)
    if output.exists():
        raise FileExistsError(f"Choose a new --smoke-output: {output}")
    directory = DATA / "arxiv"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    # No official test file is opened. Training uses train; temporary evaluation
    # uses disjoint validation papers. Long inputs exercise max-length memory.
    pools = {p: read_jsonl(directory / f"{p}.jsonl") for p in ("train", "val")}
    for p in pools:
        if digest(directory / f"{p}.jsonl") != manifest["prepared_sha256"][p]:
            raise ValueError(f"Prepared {p} hash differs before smoke")
    val_papers = sorted({r["paper_id"] for r in pools["val"]})
    def sample(rows):
        return [r for y in (0, 1) for r in sorted(
            [r for r in rows if r["y"] == y], key=lambda r: len(r["sentence"]), reverse=True)[:4]]
    split = dict(train=sample(pools["train"]),
        val=sample([r for r in pools["val"] if r["paper_id"] in val_papers[:2]]),
        test=sample([r for r in pools["val"] if r["paper_id"] in val_papers[2:4]]))
    stats = validate(split, "arxiv", strict=False)
    prepared = output / "prepared/arxiv"
    prepared.mkdir(parents=True)
    for p, rows in split.items():
        write_jsonl(prepared / f"{p}.jsonl", rows)
    write_json(prepared / "manifest.json", dict(dataset="arxiv", mode="smoke", stats=stats,
        notes=dict(smoke="REAL TRAIN/VAL ONLY; up to 8 long sentences per split; not research results"),
        input_sha256=manifest["input_sha256"],
        prepared_sha256={p: digest(prepared / f"{p}.jsonl") for p in split}))
    for method in ("scibert_ft", "biomedbert_ft"):
        options = argparse.Namespace(dataset="arxiv", data_dir=str(output / "prepared"),
            run_dir=str(output / method), methods=method, seed=0, max_features=200000,
            batch_size=args.encoder_batch_size, epochs=1, device=args.device, encoder_revision=REVISIONS[method],
            bootstrap=200, smoke=True)
        fit(options)
        evaluate(options)
    write_json(output / "smoke_complete.json", dict(note="Actual pinned encoders, tiny train/val smoke only"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true", help="Run all selected data-ready jobs sequentially")
    actions.add_argument("--check", action="store_true", help="Readiness only (default); no model downloads/training")
    actions.add_argument("--smoke-encoders", action="store_true", help="Actual pinned encoders on tiny train/val-only subsets")
    actions.add_argument("--export-reference-requests", metavar="DIR", help="Export label-free funnel inputs/provenance templates")
    parser.add_argument("--stage", action="append", choices=tuple(JOBS), help="Select a job; repeat for several. Default: all five")
    parser.add_argument("--run-root", default="runs/task5b/remaining-v1")
    parser.add_argument("--smoke-output", default="runs/task5b/encoder-smoke-20261106")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--encoder-batch-size", type=int, default=1, help="Fine-tune batch size; 1 for the tested 6 GB laptop GPU")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1 or args.encoder_batch_size < 1:
            raise ValueError("Batch size must be positive")
        if args.export_reference_requests:
            export_reference_requests(absolute(args.export_reference_requests))
            return 0
        if args.smoke_encoders:
            smoke_encoders(args)
            return 0
        selected = args.stage or list(JOBS)
        checked = set()
        for name in selected:
            dataset, methods = JOBS[name]
            if dataset not in checked:
                load_prepared(data_root(dataset) / dataset, dataset)
                checked.add(dataset)
            print(f"READY {name}: {dataset} / {methods}", flush=True)
        print(f"Result-drive free space: {shutil.disk_usage(ROOT).free / 2**30:.1f} GiB. "
              "Encoder stages save three ~0.45 GB checkpoints; base downloads use the HF cache.", flush=True)
        print("BLOCKED extras: original Lan EDA/PBA recipe; aligned original FWS lemmas for published-protocol CV; "
              "independent section/paragraph metadata for rule extensions; complete matching funnel decisions.", flush=True)
        if not args.run:
            print("Check only: no training/test scoring was started. Add --run to execute.")
            return 0
        run_root = absolute(args.run_root)
        # One immutable registration for the whole suite, including jobs that
        # will be invoked on a later day. Existing completed jobs are not rerun.
        plan = dict(jobs={k: dict(dataset=v[0], methods=v[1]) for k, v in JOBS.items()},
                    encoder_revisions=REVISIONS, batch_size=args.batch_size, device=args.device,
                    encoder_batch_size=args.encoder_batch_size,
                    epochs=3, bootstrap=2000, seed=0,
                    source_sha256=provenance()["source_sha256"], launcher_sha256=digest(Path(__file__)))
        plan_path = run_root / "suite_plan.json"
        if plan_path.exists():
            previous = json.loads(plan_path.read_text(encoding="utf-8"))
            if previous != plan:
                raise ValueError("Suite settings/source changed since registration. Keep the old suite; use a new --run-root for a new protocol.")
        else:
            write_json(plan_path, plan)
        for name in selected:
            dataset, methods = JOBS[name]
            output = run_root / name
            if name == "sal-cached-encoders":
                import_cached_sal(output, bootstrap=2000)
                continue
            if verify_completed(output / dataset, dataset, methods):
                print(f"SKIP verified completed {name}", flush=True)
                continue
            if output.exists():
                raise FileExistsError(f"Stage directory exists but is incomplete: {output}; use a new --run-root")
            command = ["run", "--datasets", dataset, "--data-dir", str(data_root(dataset)),
                       "--methods", methods, "--batch-size", str(args.encoder_batch_size if methods in REVISIONS else args.batch_size), "--device", args.device,
                       "--epochs", "3", "--bootstrap", "2000", "--run-dir", str(output)]
            if methods in REVISIONS:
                command.extend(["--encoder-revision", REVISIONS[methods]])
            code = task5b(command)
            if code:
                return code
        print("All selected data-ready jobs are complete. Blocked extras are still required or need explicit scope decisions.")
        return 0
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
