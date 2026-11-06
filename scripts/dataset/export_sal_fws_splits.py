# -*- coding: utf-8 -*-
"""Write the exact SAL and FWS evaluation splits used in the paper as plain JSONL files in data/splits/.

Teammates can then use these files directly instead of rebuilding the selection, which needs the
PubMed BioC section cache (SAL) and data/scibert_prep/train.jsonl (FWS).

    python scripts/dataset/export_sal_fws_splits.py

Writes data/splits/{sal_test,sal_dev,fws_test,fws_dev}.jsonl, one sentence per line, in reading order:
    {"paper_id", "idx", "sentence", "label"} plus "section" for SAL.
FWS text is lowercased, as in every FWS experiment (all FWS positives are lowercase in the released corpus).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/bench"))

from nli_sal_fws import SAL_FIXTURE, fws_rows, sal_rows  # noqa: E402

OUT = ROOT / "data/splits"


def write(name, rows, with_section):
    idx, last = 0, None
    with (OUT / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            idx = idx + 1 if r["pmcid"] == last else 0
            last = r["pmcid"]
            rec = {"paper_id": r["pmcid"], "idx": idx, "sentence": r["sent"], "label": int(r["label"])}
            if with_section:
                rec["section"] = r["sec"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    papers = len({r["pmcid"] for r in rows})
    print(f"{name}: {papers} papers, {len(rows)} sentences, {sum(int(r['label']) for r in rows)} positives")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write("sal_test", json.load(open(SAL_FIXTURE, encoding="utf-8")), True)
    write("sal_dev", sal_rows("dev"), True)
    fws = fws_rows()
    write("fws_test", fws["test"], False)
    write("fws_dev", fws["dev"], False)


if __name__ == "__main__":
    main()
