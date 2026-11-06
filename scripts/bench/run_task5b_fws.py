"""One-command FWS comparison using the current uploaded gold training corpus."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.experiments.task5b.data import load_prepared
from gap2idea.experiments.task5b.runner import main as task5b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="runs/task5b/fws-current-gold-full-01")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    data_dir = ROOT / "data/task5b/prepared-current-gold-v1"
    status = task5b(["preflight", "--datasets", "fws", "--fws-protocol", "current-gold"])
    if status:
        return status
    if not (data_dir / "fws/manifest.json").exists():
        status = task5b(["prepare", "--datasets", "fws", "--fws-protocol", "current-gold",
                        "--output-dir", str(data_dir)])
        if status:
            return status
    _, manifest = load_prepared(data_dir / "fws", "fws")
    if manifest.get("notes", {}).get("protocol_id") != "fws_current_gold_v1":
        raise ValueError("Launcher requires fws_current_gold_v1, not a legacy prepared split")
    print("Prepared FWS data validated: fws_current_gold_v1", flush=True)
    if args.prepare_only:
        return 0
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    return task5b(["run", "--datasets", "fws", "--data-dir", str(data_dir),
                   "--methods", "tfidf_lr,binary_nb,modernbert_nli,modernbert_nli_typed,zhang_nb",
                   "--batch-size", str(args.batch_size), "--bootstrap", str(args.bootstrap),
                   "--run-dir", str(run_dir)])


if __name__ == "__main__":
    raise SystemExit(main())
