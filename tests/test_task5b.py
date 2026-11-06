"""Protocol safeguards: real alignment/leakage/math failures, no network needed."""
import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gap2idea.experiments.task5b.data import (
    apply_structure, attach_latex_sections, digest, load_prepared, parse_sal_label,
    row, validate, write_json, write_jsonl)
from gap2idea.experiments.task5b.metrics import bootstrap, point, select_threshold
from gap2idea.experiments.task5b.models import (
    classical_scores, fit_classical, nli_evc, rule_scores)
from gap2idea.experiments.task5b.runner import compare, evaluate, fit
from gap2idea.experiments.task5b.safeweights import copy_safetensors, load_windows_model
from gap2idea.experiments.task5b.secondary import audit_zhang, score_external


def fixture():
    return {part: [row("sal", f"{part}-paper-{i // 4}", i % 4,
                       "Our study has a small sample and limited generalizability." if i % 2 else
                       "We measured blood pressure using a calibrated instrument.",
                       "limitation" if i % 2 else "none", section="Study limitations")
                   for i in range(12)] for part in ("train", "val", "test")}


class MetricTests(unittest.TestCase):
    def test_metrics_and_zero_denominators(self):
        m = point([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual(m["f1"], .5)
        self.assertEqual(m["macro_f1"], .5)
        self.assertEqual(point([0, 0], [0, 0])["f1"], 0)
        with self.assertRaises(ValueError):
            point([1], [1, 0])

    def test_threshold_validation_only_with_deterministic_ties(self):
        threshold, m = select_threshold([0, 1], [.2, .8])
        self.assertEqual(threshold, .5)
        self.assertEqual(m["f1"], 1.)
        with self.assertRaises(ValueError):
            select_threshold([0, 0], [.2, .3])

    def test_paired_cluster_ci_identity_and_reordering(self):
        rows = fixture()["test"]
        pred = [r["y"] for r in rows]
        delta = bootstrap(rows, pred, pred, n=100)
        self.assertEqual(delta["f1"], [0., 0.])
        self.assertEqual(bootstrap(rows, pred, n=100)["f1"], [1., 1.])


class DataTests(unittest.TestCase):
    def test_paper_leakage_and_expected_test_size(self):
        split = fixture()
        validate(split, "sal", strict=False)
        with self.assertRaisesRegex(ValueError, "proposal"):
            validate(split, "sal")
        split["test"][0]["paper_id"] = split["train"][0]["paper_id"]
        with self.assertRaisesRegex(ValueError, "Paper leakage"):
            validate(split, "sal", strict=False)

    def test_labels_are_not_permissively_guessed(self):
        self.assertEqual(parse_sal_label("[False, True]"), "limitation")
        with self.assertRaises(ValueError):
            parse_sal_label("['False']")

    def test_structure_requires_identity_and_all_rows(self):
        split = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "structure.jsonl"
            write_jsonl(path, [dict(paper_id="train-paper-0", idx=0, text_sha256="wrong")])
            with self.assertRaisesRegex(ValueError, "structure"):
                apply_structure(split, path)

    def test_latex_section_adapter_retains_unmapped_and_indices(self):
        a = row("arxiv", "p", 3, "An opening sentence without a section.", "none")
        b = row("arxiv", "p", 4, "Our sample size is small.", "limitation")
        split = {"train": [a, b], "val": [], "test": []}
        with tempfile.TemporaryDirectory() as tmp:
            blocks, texts = Path(tmp) / "blocks.jsonl", Path(tmp) / "texts.jsonl"
            write_jsonl(blocks, [dict(id="p", blocks=[dict(role="heading", text="Limitations"),
                                                       dict(role="text", text=b["sentence"])])])
            write_jsonl(texts, [dict(id="p", text=a["sentence"] + "\nLimitations\n" + b["sentence"])])
            stats = attach_latex_sections(split, blocks, texts)
            self.assertEqual(stats["section_mapped_sentences"], 1)
            self.assertEqual(a["section"], "")
            self.assertEqual(b["section"], "Limitations")
            self.assertEqual([r["idx"] for r in split["train"]], [3, 4])


class ModelTests(unittest.TestCase):
    def test_streamed_safetensors_preserves_weights_and_logits(self):
        import torch
        from safetensors.torch import save_file
        from transformers import AutoModelForSequenceClassification, BertConfig
        config = BertConfig(vocab_size=20, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=24)
        original = AutoModelForSequenceClassification.from_config(config).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            save_file(original.state_dict(), str(path))
            restored = load_windows_model(path, config, "cpu", torch.float32)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
            tokens = dict(input_ids=torch.tensor([[2, 3, 4]]), attention_mask=torch.ones((1, 3), dtype=torch.long))
            with torch.inference_mode():
                torch.testing.assert_close(original(**tokens).logits, restored(**tokens).logits, rtol=0, atol=0)
            wrong_model = torch.nn.Linear(2, 3)
            with self.assertRaisesRegex(ValueError, "keys differ"):
                copy_safetensors(wrong_model, path)

    def test_finetune_safetensors_roundtrip_with_tiny_local_bert(self):
        from transformers import AutoModelForSequenceClassification, BertConfig, BertTokenizerFast
        from gap2idea.experiments.task5b.models import ENCODERS, Encoder, train_encoder
        with tempfile.TemporaryDirectory() as tmp:
            base, output = Path(tmp) / "tiny_base", Path(tmp) / "trained"
            base.mkdir()
            (base / "vocab.txt").write_text("\n".join([
                "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "our", "study", "sample", "small",
                "the", "we", "measured", "blood", "pressure", "limited", "."]), encoding="utf-8")
            tokenizer = BertTokenizerFast(vocab_file=str(base / "vocab.txt"), do_lower_case=True)
            tokenizer.save_pretrained(base)
            config = BertConfig(vocab_size=16, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=24)
            AutoModelForSequenceClassification.from_config(config).save_pretrained(base, safe_serialization=True)
            with patch.dict(ENCODERS, {"scibert_ft": str(base)}), contextlib.redirect_stdout(io.StringIO()):
                train_encoder("scibert_ft", fixture()["train"][:4], output,
                              epochs=1, batch_size=2, device="cpu")
            self.assertTrue((output / "model.safetensors").is_file())
            model = Encoder(output, batch_size=2, device="cpu")
            scores = model.scores(fixture()["val"][:2])
            self.assertTrue(np.isfinite(scores).all())
            self.assertTrue(((0 <= scores) & (scores <= 1)).all())
            model.close()

    def test_nli_semantic_label_order_and_neutral_exclusion(self):
        logits = np.array([[2., 10., 0.]])
        a = nli_evc(logits, {0: "entailment", 1: "neutral", 2: "contradiction"})
        b = nli_evc(logits[:, [2, 0, 1]], {0: "contradiction", 1: "entailment", 2: "neutral"})
        np.testing.assert_allclose(a, b)
        self.assertGreater(a[0], .8)  # Three-way entailment probability would be tiny.
        with self.assertRaises(ValueError):
            nli_evc(logits, {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})

    def test_rule_section_gate_intro_citation_contrast_and_paragraph_reset(self):
        sentences = ["This study has several limitations.",
                     "The sample was recruited from only one clinic.",
                     "However, the cohort remained representative.",
                     "The small sample was reported previously [12].",
                     "Measurements used a validated device."]
        rows = [row("sal", "p", i, s, "none", section="Discussion", top_section="Discussion",
                    section_id=0, paragraph_id=0 if i < 4 else 1) for i, s in enumerate(sentences)]
        np.testing.assert_array_equal(rule_scores(rows), [0, 1, 0, 0, 0])
        for r in rows:
            r["top_section"] = "Methods"
        self.assertFalse(rule_scores(rows).any())

    def test_classical_never_fits_vocabulary_on_validation(self):
        split = fixture()
        for method in ("tfidf_lr", "binary_nb", "zhang_nb"):
            with contextlib.redirect_stderr(io.StringIO()):
                model = fit_classical(method, split["train"], max_features=100)
            test = [{**split["val"][0], "sentence": "unseentesttoken pressure sample"}]
            classical_scores(model, test)
            self.assertNotIn("unseentesttoken", model.named_steps["vectorizer"].vocabulary_)


class ProtocolTests(unittest.TestCase):
    def test_pipeline_locks_artifacts_and_refuses_rescoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data_dir = base / "prepared/sal"
            data_dir.mkdir(parents=True)
            split = fixture()
            for part, rows in split.items():
                write_jsonl(data_dir / f"{part}.jsonl", rows)
            write_json(data_dir / "manifest.json", dict(dataset="sal", mode="smoke",
                prepared_sha256={p: digest(data_dir / f"{p}.jsonl") for p in split}))
            opts = argparse.Namespace(dataset="sal", data_dir=str(base / "prepared"),
                run_dir=str(base / "run"), methods="tfidf_lr,binary_nb,kilicoglu_rules",
                smoke=True, seed=0, max_features=100, batch_size=2, epochs=1,
                encoder_revision=None, device="cpu", bootstrap=20)
            with contextlib.redirect_stdout(io.StringIO()):
                fit(opts)
                self.assertFalse((base / "run/test_started.json").exists())
                frozen = json.loads((base / "run/tfidf_lr/frozen_settings.json").read_text())
                self.assertIn("threshold", frozen)
                artifact = base / "run/tfidf_lr/model.joblib"
                intact = artifact.read_bytes()
                artifact.write_bytes(intact + b"changed")
                with self.assertRaisesRegex(ValueError, "Model artifact changed"):
                    evaluate(opts)
                self.assertFalse((base / "run/test_started.json").exists())
                artifact.write_bytes(intact)
                evaluate(opts)
                self.assertTrue((base / "run/test_complete.json").exists())
                with self.assertRaises(FileExistsError):
                    evaluate(opts)
                saved = base / "run/tfidf_lr/test_predictions.jsonl"
                compare(argparse.Namespace(predictions=str(saved), reference=str(saved), bootstrap=20,
                    seed=0, output=str(base / "paired.json")))
            delta = json.loads((base / "paired.json").read_text())
            self.assertEqual(delta["ci95"]["f1"], [0., 0.])
            self.assertEqual(json.loads((base / "run/fit_status.json").read_text())["kilicoglu_rules"]["status"], "skipped")
            # Two equally truncated exports must not pass a paired comparison.
            records = [json.loads(line) for line in saved.read_text().splitlines()]
            write_jsonl(saved, records[:-1])
            with self.assertRaisesRegex(ValueError, "completed evaluation evidence"):
                compare(argparse.Namespace(predictions=str(saved), reference=str(saved), bootstrap=20,
                    seed=0, output=str(base / "bad_pair.json")))
            with (data_dir / "test.jsonl").open("a") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "changed since locking"):
                load_prepared(data_dir, "sal")

    def test_external_predictions_require_complete_identity_and_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            directory = base / "prepared/sal"
            directory.mkdir(parents=True)
            split = fixture()
            for part, rows in split.items():
                write_jsonl(directory / f"{part}.jsonl", rows)
            write_json(directory / "manifest.json", dict(dataset="sal", mode="smoke",
                prepared_sha256={p: digest(directory / f"{p}.jsonl") for p in split}))
            protocol, source = base / "protocol.json", base / "raw.jsonl"
            write_json(protocol, dict(method="synthetic_external", training_data="synthetic unit test",
                threshold_selection="fixed", model_or_code_revision="unit-test", limitations="no research evidence"))
            records = [{**r, "y_pred": r["y"]} for r in split["test"]]
            opts = argparse.Namespace(data_dir=str(base / "prepared"), dataset="sal", method="synthetic_external",
                target="limitation", predictions=str(source), protocol=str(protocol),
                output_dir=str(base / "external"), bootstrap=20, seed=0)
            write_jsonl(source, records[:-1])
            with self.assertRaisesRegex(ValueError, "every test row"):
                score_external(opts)
            write_jsonl(source, [{**r, "sentence": "mismatched"} for r in records])
            with self.assertRaisesRegex(ValueError, "sentence mismatch"):
                score_external(opts)
            write_jsonl(source, records)
            with contextlib.redirect_stdout(io.StringIO()):
                score_external(opts)
                saved = base / "external/test_predictions.jsonl"
                compare(argparse.Namespace(predictions=str(saved), reference=str(saved),
                    output=str(base / "paired.json"), seed=0, bootstrap=20))
            self.assertEqual(json.loads((base / "paired.json").read_text())["differences"]["f1"], 0.)

    def test_secondary_diagnostic_saves_both_oof_predictions(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            corpus, lemmas = base / "corpus.csv", base / "lemmas.jsonl"
            records = [dict(id=str(i // 8), text="Future studies should test larger samples." if i % 2 else
                            "We measured performance in controlled experiments.", label=i % 2) for i in range(80)]
            with corpus.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=("id", "text", "label"))
                writer.writeheader()
                writer.writerows(records)
            write_jsonl(lemmas, [dict(raw_idx=i, text_sha256=r["text_sha256"], lemmatized_text=r["sentence"])
                                  for i, r in enumerate([row("fws", "p", i, x["text"], "none") for i, x in enumerate(records)])])
            with contextlib.redirect_stdout(io.StringIO()):
                audit_zhang(argparse.Namespace(corpus=str(corpus), lemmatized_jsonl=str(lemmas),
                    output_dir=str(base / "diagnostic"), seed=0, folds=4))
            out = [json.loads(line) for line in (base / "diagnostic/oof_predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(out), 80)
            self.assertEqual({r["fold"] for r in out}, {0, 1, 2, 3})
            self.assertTrue(all("clean_prediction" in r and "deliberately_leaky_prediction" in r for r in out))


if __name__ == "__main__":
    unittest.main()
