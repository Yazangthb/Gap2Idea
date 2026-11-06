"""Remaining-run launcher integrity checks; no network or full-model training."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("task5b_remaining", ROOT / "scripts/bench/run_task5b_remaining.py")
remaining = importlib.util.module_from_spec(spec)
spec.loader.exec_module(remaining)


class RemainingTests(unittest.TestCase):
    def test_cached_encoder_import_requires_exact_original_identity(self):
        fixture = [dict(pmcid="p", sent="A test sentence.", label=1, preds={"SciBERT-FT": 0})]
        test = [dict(paper_id="p", idx=0, sentence="A test sentence.", y=1, text_sha256="original-hash")]
        records = remaining.cached_records(test, fixture, "SciBERT-FT")
        self.assertEqual(records[0]["y_pred"], 0)  # Preserve model decision, not gold.
        self.assertNotIn("y", records[0])
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            remaining.cached_records([{**test[0], "sentence": "changed"}], fixture, "SciBERT-FT")

    def test_incomplete_run_cannot_be_silently_restarted(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "arxiv"
            self.assertFalse(remaining.verify_completed(run, "arxiv", "scibert_ft"))
            run.mkdir()
            with self.assertRaisesRegex(ValueError, "Incomplete run"):
                remaining.verify_completed(run, "arxiv", "scibert_ft")

    def test_funnel_request_export_omits_gold_and_preserves_every_id(self):
        rows = [dict(paper_id="p", idx=i, sentence=f"Sentence {i}.", text_sha256=f"h{i}",
                     label="limitation", y=1, section="Discussion") for i in range(3)]
        fixture = dict(train=[], val=[], test=rows)
        manifest = dict(prepared_sha256={"test": "frozen"}, notes={})
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                remaining, "load_prepared", return_value=(fixture, manifest)), contextlib.redirect_stdout(io.StringIO()):
            target = Path(tmp) / "requests"
            remaining.export_reference_requests(target)
            for dataset in ("sal", "fws", "arxiv"):
                records = [json.loads(line) for line in (target / dataset / "sentences.jsonl").read_text().splitlines()]
                self.assertEqual([r["idx"] for r in records], [0, 1, 2])
                self.assertTrue(all("label" not in r and "y" not in r for r in records))
                self.assertEqual(records[0]["context"], "Sentence 0. Sentence 1.")
            with self.assertRaises(FileExistsError):
                remaining.export_reference_requests(target)

    def test_check_only_does_not_create_run_directories_or_train(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
                remaining, "load_prepared", return_value=({}, {})), patch.object(
                remaining, "task5b") as runner, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp) / "not-started"
            self.assertEqual(remaining.main(["--check", "--run-root", str(root)]), 0)
            self.assertFalse(root.exists())
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
