"""Run an explicit baseline configuration using the local runner and its plan checks.

Example: whole_paper_other_llm.py --model qwen3.5:4b-bf16 --backend ollama
  --dataset sal --setup AS --prompt strict --split dev --limit-papers 2 --output-dir NEW_DIR
"""
import sys

from baseline_qwen import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
