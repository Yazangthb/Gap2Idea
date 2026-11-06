"""Cross-platform handoff restoration and immutable-evidence safeguards."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("task5b_server", ROOT / "scripts/bench/run_task5b_server.py")
server = importlib.util.module_from_spec(spec)
spec.loader.exec_module(server)


class ServerTests(unittest.TestCase):
    def test_windows_provenance_separators_are_portable_and_traversal_is_rejected(self):
        self.assertEqual(server.relative_path(r"checkpoint\model.safetensors"), Path("checkpoint/model.safetensors"))
        for invalid in ("../escape", "/absolute", r"C:\private\file", r"a\..\escape"):
            with self.assertRaises(ValueError):
                server.relative_path(invalid)

    def test_restore_preserves_exact_crlf_bytes_and_never_overwrites_changes(self):
        payload = b'{"value": 1}\r\n'
        name = "data/task5b/frozen/example.jsonl"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = root / "parts"
            parts.mkdir()
            archive = io.BytesIO()
            with tarfile.open(fileobj=archive, mode="w:gz") as tar:
                item = tarfile.TarInfo(name)
                item.size = len(payload)
                tar.addfile(item, io.BytesIO(payload))
            raw = archive.getvalue()
            part = parts / "archive.part000"
            part.write_bytes(raw)
            bundle = dict(name="fixture", sha256=hashlib.sha256(raw).hexdigest(),
                parts=[dict(file=part.name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest())],
                files=[dict(path=name, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())])
            with patch.object(server, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                server.restore_bundle(parts, bundle)
                self.assertEqual((root / name).read_bytes(), payload)
                server.restore_bundle(parts, bundle)
                (root / name).write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    server.restore_bundle(parts, bundle)

    def test_corrupt_archive_part_is_rejected_before_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.part").write_bytes(b"bad")
            bundle = dict(name="bad", sha256="wrong", parts=[dict(file="bad.part", size=3, sha256="wrong")],
                          files=[dict(path="data/target", size=2, sha256="wrong")])
            with patch.object(server, "ROOT", root), self.assertRaisesRegex(ValueError, "Corrupt/missing"):
                server.restore_bundle(root, bundle)
            self.assertFalse((root / "data/target").exists())


if __name__ == "__main__":
    unittest.main()
