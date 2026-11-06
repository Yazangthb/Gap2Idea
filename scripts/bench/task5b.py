"""Portable entry point; run from any directory without installing the repo."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.experiments.task5b.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
