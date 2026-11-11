# -*- coding: utf-8 -*-
"""Import the Task 5b baseline predictions (branch Task_5b) into this branch in a compact form.

The Task 5b runs live on another branch together with large checkpoints. For the paper we only need, for
every baseline, which test sentences it marked positive. This script reads the complete saved test
predictions of each finished Task 5b run and writes one small file per dataset:

  docs/experiments/task5b_baselines/kept_{sal,fws,arxiv}.json

Each file lists the test rows (paper id, sentence index, SHA-256 of the sentence text, gold target) once, and for
every method the indices of the rows it predicted positive, its frozen threshold and the hash of the source
prediction file. paper_baselines_table.py checks the rows against this branch's own test sets before scoring.

  python scripts/bench/import_task5b_baselines.py <path to a checkout of branch Task_5b>
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/experiments/task5b_baselines"
RUNS = {
    "sal": [("sal-full-01/sal", ["tfidf_lr", "binary_nb", "kilicoglu_section_proxy", "modernbert_nli"])],
    "fws": [("fws-current-gold-full-01/fws", ["tfidf_lr", "binary_nb", "zhang_nb", "modernbert_nli", "modernbert_nli_typed"])],
    "arxiv": [("remaining-v1/arxiv-baselines/arxiv", ["tfidf_lr", "binary_nb", "modernbert_nli", "modernbert_nli_typed"]),
              ("remaining-v1/arxiv-scibert/arxiv", ["scibert_ft"]),
              ("remaining-v1/arxiv-biomedbert/arxiv", ["biomedbert_ft"])],
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True, check=True).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    for dataset, runs in RUNS.items():
        rows, methods = None, {}
        for run, names in runs:
            folder = src / "runs/task5b" / run
            if not (folder / "test_complete.json").exists():
                sys.exit(f"{folder}: run is not complete")
            for name in names:
                file = folder / name / "test_predictions.jsonl"
                preds = [json.loads(x) for x in file.read_text(encoding="utf-8").splitlines() if x.strip()]
                these = [[p["paper_id"], p["idx"], p["text_sha256"], int(p["y"])] for p in preds]
                if rows is None:
                    rows = these
                elif rows != these:
                    sys.exit(f"{dataset}/{name}: test rows differ from the other methods of this dataset")
                methods[name] = {"positive_rows": [i for i, p in enumerate(preds) if int(p["y_pred"]) == 1],
                                 "threshold": preds[0].get("threshold"),
                                 "source": f"runs/task5b/{run}/{name}/test_predictions.jsonl",
                                 "source_sha256": sha(file)}
        out = {"dataset": dataset, "task5b_commit": commit, "rows": rows, "methods": methods}
        (OUT / f"kept_{dataset}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(dataset, len(rows), "rows;", {k: len(v["positive_rows"]) for k, v in methods.items()})


if __name__ == "__main__":
    main()
