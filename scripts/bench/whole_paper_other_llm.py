# -*- coding: utf-8 -*-
"""Task 5a's whole-paper baseline (setup A) with another served model, using its script unchanged.

baseline_qwen.py is read from origin/task-5a-qwen-baselines; only the model name, output folder and client are
redirected, and its "plan committed" check is skipped (the plan is frozen on that branch, not this one).

  QWEN_PORT=8011 python whole_paper_other_llm.py qwen3.5-4b OUT_DIR
"""
import argparse
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/bench"), str(ROOT / "src")]
from stage_c_prompt_vs_model import qwen_client  # noqa: E402

RUNS = [("sal", p) for p in ("keep_biased", "neutral", "strict")] + \
       [("gold", p) for p in ("keep_biased", "neutral", "strict")] + [("fws", "validate_fws")]


def main():
    model, out = sys.argv[1], Path(sys.argv[2])
    src = subprocess.run(["git", "show", "origin/task-5a-qwen-baselines:scripts/bench/baseline_qwen.py"], cwd=ROOT,
                         capture_output=True, text=True, encoding="utf-8", check=True).stdout
    mod_path = Path(tempfile.mkdtemp()) / "baseline_qwen.py"
    mod_path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("baseline_qwen", mod_path)
    bq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bq)
    bq.ROOT, bq.BENCH, bq.OUTDIR, bq.MODEL = ROOT, ROOT / "scripts/bench", out, model
    bq.qwen_client = qwen_client
    bq.plan_is_committed = lambda: True
    for dataset, prompt in RUNS:
        bq.run(argparse.Namespace(dataset=dataset, setup="A", prompt=prompt, split="test", limit_papers=0))


if __name__ == "__main__":
    main()
