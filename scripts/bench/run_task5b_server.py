"""Restore and verify the handoff, then run only unfinished Task 5b stages.

Default / --check restores missing packaged files but never trains. --run starts
only FWS SciBERT and the cached SAL import, skipping verified server completions.
Original Windows manifests remain immutable; paths in their provenance are not
rewritten to pretend the previous experiments ran on Linux.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gap2idea.experiments.task5b.data import digest, load_prepared
import run_task5b_remaining as remaining


def relative_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError(f"Unsafe relative artifact path: {value}")
    return Path(*path.parts)


def restore_bundle(folder, bundle):
    expected = {r["path"]: r for r in bundle["files"]}
    missing = []
    for name, record in expected.items():
        target = ROOT / relative_path(name)
        if target.exists():
            if not target.is_file() or digest(target) != record["sha256"]:
                raise ValueError(f"Existing file differs; refusing to overwrite: {target}")
        else:
            missing.append(name)
    if not missing:
        print(f"VERIFIED already restored {bundle['name']}", flush=True)
        return
    temp_dir = ROOT / "data/task5b/handoff-temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=temp_dir) as assembled:
        h = hashlib.sha256()
        for part in bundle["parts"]:
            source = folder / relative_path(part["file"])
            if source.stat().st_size != part["size"] or digest(source) != part["sha256"]:
                raise ValueError(f"Corrupt/missing handoff part: {source}")
            with source.open("rb") as f:
                for block in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(block)
                    assembled.write(block)
        if h.hexdigest() != bundle["sha256"]:
            raise ValueError("Assembled archive hash differs")
        assembled.seek(0)
        seen = set()
        with tarfile.open(fileobj=assembled, mode="r:gz") as archive:
            for member in archive:
                name = relative_path(member.name).as_posix()
                if not member.isfile() or name not in expected or name in seen:
                    raise ValueError(f"Unexpected/duplicate/non-file archive member: {name}")
                seen.add(name)
                record = expected[name]
                if member.size != record["size"]:
                    raise ValueError(f"Archive size mismatch: {name}")
                target = ROOT / relative_path(name)
                if name not in missing:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                # Never traverse an existing symlink outside the checkout.
                if not target.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError(f"Extraction target escapes the checkout: {name}")
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False, suffix=".handoff-part") as out:
                    tmp = Path(out.name)
                    try:
                        with archive.extractfile(member) as source:
                            shutil.copyfileobj(source, out)
                        out.flush()
                    except BaseException:
                        tmp.unlink(missing_ok=True)
                        raise
                if digest(tmp) != record["sha256"]:
                    tmp.unlink()
                    raise ValueError(f"Extracted file hash mismatch: {name}")
                tmp.replace(target)
        if seen != set(expected):
            raise ValueError("Archive is missing expected entries")
    print(f"RESTORED {bundle['name']}: {len(missing)} exact files", flush=True)


def verify_finished(run):
    if not (run / "test_complete.json").exists():
        raise ValueError(f"Missing completion marker: {run}")
    result = json.loads((run / "results.json").read_text(encoding="utf-8"))
    complete = json.loads((run / "fit_complete.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if set(result) != set(meta["requested_methods"]) or set(result) != set(complete["settings_sha256"]):
        raise ValueError(f"Incomplete method coverage: {run}")
    for method, settings_sha in complete["settings_sha256"].items():
        directory = run / method
        if digest(directory / "frozen_settings.json") != settings_sha:
            raise ValueError(f"Frozen settings changed: {directory}")
        settings = json.loads((directory / "frozen_settings.json").read_text(encoding="utf-8"))
        for relative, expected in settings["artifact_sha256"].items():
            if digest(directory / relative_path(relative)) != expected:
                raise ValueError(f"Artifact hash mismatch: {directory}/{relative}")
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        if metrics != result[method] or digest(directory / "test_predictions.jsonl") != metrics["predictions_sha256"]:
            raise ValueError(f"Result/prediction hash mismatch: {directory}")
    print(f"KEEP completed {run.relative_to(ROOT)} ({len(result)} methods)", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--handoff", default="artifacts/task5b/server-handoff-20261106")
    parser.add_argument("--run-root", default="runs/task5b/server-v1")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--encoder-batch-size", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.encoder_batch_size < 1:
            raise ValueError("Encoder batch size must be positive")
        folder = ROOT / relative_path(args.handoff)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        for bundle in manifest["bundles"]:
            restore_bundle(folder, bundle)
        for relative in manifest["prepared_directories"]:
            path = ROOT / relative_path(relative)
            load_prepared(path, path.name)
        for relative in manifest["completed_runs"]:
            verify_finished(ROOT / relative_path(relative))
        print("PENDING: FWS SciBERT (restart training), then cached SAL encoder import.", flush=True)
        print("The three completed arXiv jobs and completed SAL/FWS baselines will NOT be rerun.", flush=True)
        if not args.run:
            print("Handoff ready. Add --run to start the two unfinished stages.")
            return 0
        if args.device != "cpu":
            import torch
            if not torch.cuda.is_available():
                raise ValueError("No CUDA GPU is available. Fix the server PyTorch/GPU setup, or explicitly choose --device cpu.")
        return remaining.main(["--run", "--stage", "fws-scibert", "--stage", "sal-cached-encoders",
            "--run-root", args.run_root, "--device", args.device,
            "--encoder-batch-size", str(args.encoder_batch_size)])
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
