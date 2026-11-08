# -*- coding: utf-8 -*-
"""Rebuild the SAL and FWS evaluation files from the third-party data and our own metadata.

Run scripts/dataset/fetch_external.py first. This writes, and checks against the SHA-256 values of the files
our experiments used:

    stage_c_package/sal_criteria/sal_test_allmodels.json   SAL-40 test rows with our Stage B predictions
    data/splits/{sal_test,sal_dev,fws_test,fws_dev}.jsonl   the exact evaluation splits (see data/splits/README.md)

Only our own annotations are stored in the repository (data/eval_meta/): the section of each SAL sentence,
taken from PubMed BioC, and our detectors' predictions. SAL's labels come from the fetched files. The sentence text comes from the
fetched files: SAL rows are the CSV rows with at least three words, in file order, and FWS rows follow the
selection described in data/splits/README.md.

    python scripts/dataset/build_eval_files.py [--out-root DIR]
"""
import argparse
import ast
import hashlib
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META = ROOT / "data/eval_meta"
MIN_WORDS = 3

# SHA-256 of the files used in the paper's experiments.
EXPECTED = {
    "stage_c_package/sal_criteria/sal_test_allmodels.json": None,
    "data/splits/sal_test.jsonl": None,
    "data/splits/sal_dev.jsonl": None,
    "data/splits/fws_test.jsonl": None,
    "data/splits/fws_dev.jsonl": None,
}
EXPECTED.update(json.loads((META / "sha256.json").read_text(encoding="utf-8")) if (META / "sha256.json").exists() else {})


def sal_sentences(split):
    """(paper, sentence, label) for every CSV row with at least three words, in file order."""
    df = pd.read_csv(ROOT / f"data/external/sal_rct/{split}.csv")
    rows = []
    for _, r in df.iterrows():
        sent = str(r.sentences).strip()
        if len(sent.split()) >= MIN_WORDS:
            rows.append((r.pmid, sent, int(any(ast.literal_eval(str(r.limitation_annotation))))))
    return rows


def sal_rows(split):
    meta = json.loads((META / f"sal_{split}_meta.json").read_text(encoding="utf-8"))
    text = sal_sentences(split)
    if len(text) != len(meta):
        sys.exit(f"SAL {split}: {len(text)} sentences in the CSV, {len(meta)} in the metadata")
    rows = []
    for (pid, sent, label), m in zip(text, meta):
        if pid != m["pmcid"]:
            sys.exit(f"SAL {split}: paper mismatch {pid} vs {m['pmcid']}")
        row = {"pmcid": pid, "sent": sent, "sec": m["sec"], "label": label}
        if "preds" in m:
            row["preds"] = m["preds"]
        rows.append(row)
    return rows


def fws_rows():
    """Test: the first 209 papers of a seeded shuffle; dev: the next papers up to 3000 sentences."""
    train = {json.loads(line)["sentence"].strip() for line in
             (ROOT / "data/scibert_prep/train.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()}
    df = pd.read_csv(ROOT / "data/external/fws_corpus.csv")
    df = df[~df.text.str.strip().isin(train)].reset_index(drop=True)
    papers = df.id.unique().tolist()
    random.seed(0)
    random.shuffle(papers)
    groups = {pid: sub for pid, sub in df.groupby("id", sort=False)}
    picked, used = {"test": [], "dev": []}, 0
    for pid in papers:
        part = "test" if used < 1500 else "dev"
        if part == "dev" and sum(len(groups[p]) for p in picked["dev"]) >= 3000:
            break
        picked[part].append(pid)
        used += len(groups[pid])
    # FWS text is lowercased: in the released corpus letter case reveals the label.
    return {part: [{"pmcid": pid, "sent": str(s).lower(), "label": int(y)}
                   for pid in ids for s, y in zip(groups[pid].text, groups[pid].label)]
            for part, ids in picked.items()}


def split_lines(rows, with_section):
    out, idx, last = [], 0, None
    for r in rows:
        idx = idx + 1 if r["pmcid"] == last else 0
        last = r["pmcid"]
        rec = {"paper_id": r["pmcid"], "idx": idx, "sentence": r["sent"], "label": int(r["label"])}
        if with_section:
            rec["section"] = r["sec"]
        out.append(json.dumps(rec, ensure_ascii=False) + "\n")
    return "".join(out).encode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default=str(ROOT))
    out_root = Path(ap.parse_args().out_root)
    sal_test = sal_rows("test")
    fws = fws_rows()
    files = {
        "stage_c_package/sal_criteria/sal_test_allmodels.json": json.dumps(sal_test).encode("utf-8"),
        "data/splits/sal_test.jsonl": split_lines(sal_test, True),
        "data/splits/sal_dev.jsonl": split_lines(sal_rows("dev"), True),
        "data/splits/fws_test.jsonl": split_lines(fws["test"], False),
        "data/splits/fws_dev.jsonl": split_lines(fws["dev"], False),
    }
    bad = []
    for rel, data in files.items():
        path = out_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        ok = EXPECTED.get(rel) in (None, digest)
        bad += [] if ok else [rel]
        print(f"{'ok' if ok else 'MISMATCH':9s} {rel}  sha256 {digest[:16]}")
    if bad:
        sys.exit(f"{len(bad)} file(s) differ from the versions used in the paper: {bad}")


if __name__ == "__main__":
    main()
