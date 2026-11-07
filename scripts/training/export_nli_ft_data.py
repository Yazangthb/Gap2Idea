# -*- coding: utf-8 -*-
"""Write the files for fine-tuning ModernBERT-large-nli on gold sentence labels (finetune_nli.py).

  python export_nli_ft_data.py OUT_DIR

  train.jsonl   gold TRAIN papers, arXiv only, Stage A survivors: {sentence, label}. The 28 SAL-40 papers in
                the gold train split are left out so SAL-40 stays an unseen test set.
  val.jsonl     gold VALIDATION, every sentence: {sentence, label, junk}. Used to pick the epoch.
  score_*.json  label-free sentence lists to score after training: gold val and test, SAL dev and test,
                FWS dev and test (lowercased, as nli_sal_fws.py does). No test labels leave this machine.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bench"))
from nli_variants import load_val  # noqa: E402
from nli_sal_fws import load_all  # noqa: E402


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    _, train = load_val("train")
    keep = [r for r in train if r["dom"] == "arxiv_cs" and not r["junk"]]
    with (out / "train.jsonl").open("w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps({"pid": r["pid"], "sentence": r["sentence"], "label": r["label"]}, ensure_ascii=False) + "\n")
    print("train papers", len({r["pid"] for r in keep}), "sentences", len(keep),
          "gaps", sum(r["label"] != "none" for r in keep), flush=True)
    _, val = load_val("val")
    with (out / "val.jsonl").open("w", encoding="utf-8") as fh:
        for r in val:
            fh.write(json.dumps({"sentence": r["sentence"], "label": r["label"], "junk": r["junk"]}, ensure_ascii=False) + "\n")
    print("val sentences", len(val), flush=True)
    for part in ("val", "test"):
        _, rows = load_val(part)
        (out / f"score_gold_{part}.json").write_text(json.dumps([{"sentence": r["sentence"]} for r in rows], ensure_ascii=False),
                                                   encoding="utf-8")
    for name, rows in load_all().items():
        (out / f"score_{name}.json").write_text(json.dumps([{"sentence": r["sent"]} for r in rows], ensure_ascii=False),
                                              encoding="utf-8")
        print(name, "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
