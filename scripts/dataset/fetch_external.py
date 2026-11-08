# -*- coding: utf-8 -*-
"""Download the third-party benchmark files used in the paper, at the exact versions we used.

The SAL data (Lan et al.) and the FWS corpus (Zhang et al.) have no licence, so this repository does not
redistribute them. This script fetches them from their authors' GitHub repositories, pinned to a commit,
and checks each file's git blob hash, so the bytes are identical to what our experiments used.

    python scripts/dataset/fetch_external.py
    python scripts/dataset/build_eval_files.py   # then rebuild the evaluation splits and the SAL fixture
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SAL = "https://raw.githubusercontent.com/MengfeiLan/SAL_Type_Classification/1848913f4952f16169f1c693d88d6ca6ef6e412f/data"
FWS = "https://raw.githubusercontent.com/xiangyi-njust/FWS/79985c41c930ac07cd9c7c5a0f23c71c93c404e1/Dataset"

# (local path, URL, git blob SHA-1 of the file at that commit)
SOURCES = [
    ("data/external/sal_rct/train.csv", f"{SAL}/train.csv", "e03445f5ea75ebdb150fef0384b316dff315a932"),
    ("data/external/sal_rct/dev.csv", f"{SAL}/dev.csv", "ab987c593cf1b7eba08f0f276b3e20947661e46d"),
    ("data/external/sal_rct/test.csv", f"{SAL}/test.csv", "494aba82448050a9900e007f9dcf49e0668c0d04"),
    ("data/external/fws_corpus.csv", f"{FWS}/Corpus_For_FWS_Recognition.csv", "7790b5f026d925ea6af188e89ff76255cb8ea32e"),
]


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main():
    bad = []
    for rel, url, sha in SOURCES:
        path = ROOT / rel
        if path.exists() and blob_sha(path.read_bytes()) == sha:
            print(f"ok (already present)  {rel}")
            continue
        print(f"downloading           {rel}")
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        if blob_sha(data) != sha:
            bad.append(rel)
            print(f"HASH MISMATCH         {rel}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"ok                    {rel}")
    if bad:
        sys.exit(f"{len(bad)} file(s) did not match the pinned version: {bad}")


if __name__ == "__main__":
    main()
