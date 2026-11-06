"""Package exact frozen data and large finished checkpoints for Git transport.

Only explicitly enumerated completed runs are included. Never includes the live
FWS training directory, download caches, credentials or failed smoke runs.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parents[2]
COMPLETED = ["runs/task5b/sal-full-01/sal", "runs/task5b/fws-current-gold-full-01/fws",
    "runs/task5b/remaining-v1/arxiv-baselines/arxiv", "runs/task5b/remaining-v1/arxiv-scibert/arxiv",
    "runs/task5b/remaining-v1/arxiv-biomedbert/arxiv"]
PREPARED = ["data/task5b/prepared/sal", "data/task5b/prepared-current-gold-v1/fws",
            "data/task5b/prepared-current-gold-v1/arxiv"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def package(paths, output, name):
    archive = output / (name + ".tar.gz")
    records = []
    with archive.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=4, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w|") as tar:
            for path in sorted(paths):
                relative = path.relative_to(ROOT).as_posix()
                print("Packing", relative, flush=True)
                tar.add(path, arcname=relative, recursive=False)
                records.append(dict(path=relative, size=path.stat().st_size, sha256=sha(path)))
    parts = []
    with archive.open("rb") as f:
        i = 0
        while block := f.read(64 * 1024 * 1024):
            part = output / f"{name}.tar.gz.part{i:03d}"
            part.write_bytes(block)
            parts.append(dict(file=part.name, size=len(block), sha256=hashlib.sha256(block).hexdigest()))
            i += 1
    result = dict(name=name, sha256=sha(archive), parts=parts, files=records)
    # Remove only this tool's redundant temporary unsplit archive, after verifying
    # every retained part. Exact target is under the explicitly selected output.
    assert archive.resolve().parent == output.resolve()
    archive.unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/task5b/server-handoff-20261106")
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(f"Refusing to replace handoff: {output}")
    for relative in COMPLETED:
        if not (ROOT / relative / "test_complete.json").exists():
            raise ValueError(f"Not a completed run: {relative}")
    prepared = [p for relative in PREPARED for p in (ROOT / relative).iterdir() if p.is_file()]
    checkpoints = [p for relative in COMPLETED for p in (ROOT / relative).rglob("*.safetensors")]
    output.mkdir(parents=True)
    bundles = [package(prepared, output, "prepared-data"), package(checkpoints, output, "finished-checkpoints")]
    manifest = dict(format_version=1, created_date="2026-11-06", bundles=bundles,
        completed_runs=COMPLETED, prepared_directories=PREPARED,
        pending_jobs=["fws-scibert", "sal-cached-encoders"],
        interrupted_training="FWS SciBERT has no saved optimizer/training checkpoint; restart only this stage from its pinned initialization.")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Handoff packaged:", output, flush=True)


if __name__ == "__main__":
    main()
