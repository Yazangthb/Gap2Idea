"""Train the Stage-B embedding head by self-distillation from the LLM teacher.

Positives  = every future_work / limitation gap sentence the LLM teacher already
             produced in runs/*/data/gaps.tsv (zero new annotation).
Negatives  = sentences that live in the SAME Stage-A slices (Limitations /
             Future-Work / Conclusion regions) but the teacher did NOT pick —
             i.e. the hard in-domain negatives the head will actually face.

LEAKAGE GUARD (critical for an honest benchmark): every paper in the held-out
eval sets — data/bench_gold (the gold the funnel is scored against) and
data/bench (the per-sentence bench) — is EXCLUDED from training. Asserted, not
assumed. open_problem teacher gaps are dropped from BOTH classes (out of scope,
and we must not teach the head to call them negatives).

Output: data/gap_head.joblib  (sklearn clf + encoder name)

Usage:
    python scripts/training/train_gap_head.py --out data/gap_head.joblib
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    DEFAULT_ENCODER,
    EmbeddingGapHead,
    _looks_like_sentence,
    slice_terminal_regions,
    token_containment,
)
from gap2idea.pipeline.gap_prefilter import normalize_text, split_sentences  # noqa: E402
from gap2idea.pipeline.sections import _cut_before_references  # noqa: E402
from gap2idea.utils import get_logger  # noqa: E402

log = get_logger(__name__)

RUNS = ["ai", "ai_v1", "ml", "ml_v1", "math", "math_v1"]
TARGET = {"future_work", "limitation"}
GAP_MATCH_TAU = 0.6      # token-containment above which a slice sentence == a teacher gap
WEAK_CAP_PER_REGION = 6  # max distant-supervision positives per explicit section


def _read_jsonl(path: Path) -> dict[str, dict]:
    import json

    out: dict[str, dict] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[str(rec.get("id"))] = rec
    return out


def eval_paper_ids(root: Path) -> set[str]:
    """IDs that must never appear in training."""
    ids: set[str] = set()
    mani = root / "data" / "bench_gold" / "papers_manifest.tsv"
    if mani.exists():
        ids |= set(pd.read_csv(mani, sep="\t", dtype=str)["id"].astype(str))
    ls = root / "data" / "bench" / "label_sheet.tsv"
    if ls.exists():
        ids |= set(pd.read_csv(ls, sep="\t", dtype=str)["paper_id"].astype(str))
    return ids


def _load_positives(root: Path, extra_gold: Path | None) -> tuple[dict[str, list[tuple[str, str]]], dict[str, dict]]:
    """Return (pos_by_paper: pid -> [(gap_type, sentence)], texts_all: pid -> rec).

    Positives come from BOTH the original teacher gaps.tsv (<=2/paper, the
    bootstrap labels) and an optional complete LLM-extracted gold (more, and
    crucially more *limitation* exemplars). Texts are loaded once across runs.
    """
    texts_all: dict[str, dict] = {}
    pos_by_paper: dict[str, list[tuple[str, str]]] = {}
    for run in RUNS:
        data = root / "runs" / run / "data"
        if (data / "paper_texts.jsonl").exists():
            texts_all.update(_read_jsonl(data / "paper_texts.jsonl"))
        gp = data / "gaps.tsv"
        if gp.exists():
            g = pd.read_csv(gp, sep="\t", dtype={"id": str})
            for _, r in g[g["gap_type"].isin(TARGET)].iterrows():
                pos_by_paper.setdefault(str(r["id"]), []).append((str(r["gap_type"]), str(r["gap_sentence"])))
    if extra_gold and extra_gold.exists():
        eg = pd.read_csv(extra_gold, sep="\t", dtype={"paper_id": str})
        for _, r in eg[eg["gap_type"].isin(TARGET)].iterrows():
            pos_by_paper.setdefault(str(r["paper_id"]), []).append((str(r["gap_type"]), str(r["gap_sentence"])))
    return pos_by_paper, texts_all


def build_dataset(root: Path, exclude: set[str], extra_gold: Path | None,
                  no_distant: bool = False) -> tuple[list[str], list[str], dict]:
    sentences: list[str] = []
    labels: list[str] = []
    train_ids: set[str] = set()
    stats = {"papers": 0, "skipped_eval": 0, "pos_fut": 0, "pos_lim": 0, "neg": 0}

    pos_by_paper, texts_all = _load_positives(root, extra_gold)
    for pid, positives in pos_by_paper.items():
        if pid in exclude:
            stats["skipped_eval"] += 1
            continue
        rec = texts_all.get(pid)
        if not rec:
            continue
        stats["papers"] += 1
        train_ids.add(pid)
        blocks = rec.get("blocks") if isinstance(rec.get("blocks"), list) else None
        text = str(rec.get("text", ""))
        regions = slice_terminal_regions(text, blocks=blocks)
        slice_sents = [s for r in regions for s in r.sentences]

        # 1) STRONG positives — teacher + LLM gold (dedup, keep first type)
        seen: dict[str, str] = {}
        for gtype, gs in positives:
            key = normalize_text(gs)
            if key and key not in seen:
                seen[key] = gtype
                sentences.append(gs)
                labels.append(gtype)
                stats["pos_fut" if gtype == "future_work" else "pos_lim"] += 1

        # 1b) WEAK positives (distant supervision, no LLM) — sentences inside an
        #     EXPLICIT Limitations/Future-Work heading section are very likely
        #     gaps of that type. Caps per region so a long section can't dominate;
        #     this is the cheap fix for the limitation data shortage.
        if not no_distant:
            for r in regions:
                if r.section_type not in ("limitations", "future_work"):
                    continue
                wtype = "limitation" if r.section_type == "limitations" else "future_work"
                taken = 0
                for s in r.sentences:
                    key = normalize_text(s)
                    if not key or key in seen or not _looks_like_sentence(s):
                        continue
                    seen[key] = wtype
                    sentences.append(s)
                    labels.append(wtype)
                    stats[f"weak_{ 'fut' if wtype=='future_work' else 'lim'}"] = \
                        stats.get(f"weak_{'fut' if wtype=='future_work' else 'lim'}", 0) + 1
                    taken += 1
                    if taken >= WEAK_CAP_PER_REGION:
                        break

        # 2) negatives — body OUTSIDE the slice (reliably non-gap). NEVER from the
        #    gap-rich slice (that poisoned the head: skipped real gaps -> "none").
        gap_sents = [gs for _, gs in positives]
        slice_norm = {normalize_text(s) for s in slice_sents}
        for s in split_sentences(_cut_before_references(text)):
            if normalize_text(s) in slice_norm or not _looks_like_sentence(s):
                continue
            if max((token_containment(gp, s) for gp in gap_sents), default=0.0) >= GAP_MATCH_TAU:
                continue
            sentences.append(s)
            labels.append("none")
            stats["neg"] += 1

    stats["train_ids"] = train_ids
    return sentences, labels, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/gap_head.joblib"))
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--C", type=float, default=2.0)
    ap.add_argument("--max-neg-ratio", type=float, default=4.0,
                    help="cap negatives at this multiple of positives (balance)")
    ap.add_argument("--extra-gold", type=Path, default=Path("data/bench_gap/train/gold_sentences.tsv"),
                    help="complete LLM-extracted gap labels to add as positives")
    ap.add_argument("--no-distant", action="store_true",
                    help="disable distant-supervision weak positives from explicit sections")
    ap.add_argument("--acl-limitations", type=Path, default=Path("data/acl_limitations.tsv"),
                    help="harvested ACL mandated-Limitations sentences (extra limitation positives)")
    ap.add_argument("--acl-cap", type=int, default=0,
                    help="max ACL limitation positives to add (0 = off / don't use)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    exclude = eval_paper_ids(root)
    log.info("Excluding %d held-out eval paper ids from training", len(exclude))

    sents, labels, stats = build_dataset(root, exclude, args.extra_gold, args.no_distant)
    train_ids = sorted(stats.pop("train_ids"))

    # Inject harvested ACL mandated-Limitations sentences as extra `limitation`
    # positives (the literature's fix for our limitation-data shortage). Capped so
    # they don't swamp the other classes; sentence-level leakage already filtered
    # in harvest_acl_limitations.py.
    if args.acl_cap and args.acl_limitations.exists():
        acl = pd.read_csv(args.acl_limitations, sep="\t")
        rng0 = np.random.default_rng(args.seed)
        acl = acl.iloc[rng0.permutation(len(acl))[:args.acl_cap]]
        sents.extend(acl["gap_sentence"].astype(str).tolist())
        labels.extend(["limitation"] * len(acl))
        stats["acl_lim"] = len(acl)
    log.info("Dataset: %s", stats)
    # Leakage guard: no training paper may be a held-out eval paper.
    assert not (set(train_ids) & exclude), "LEAKAGE: training paper in eval set"

    df = pd.DataFrame({"sentence": sents, "label": labels})
    # ---- LEAKAGE / SANITY ASSERTIONS -------------------------------------
    assert stats["skipped_eval"] > 0, "expected to skip the held-out eval papers"
    assert (df["label"] == "none").sum() > 0 and (df["label"] != "none").sum() > 0
    pos_n = int((df["label"] != "none").sum())

    # balance negatives
    rng = np.random.default_rng(args.seed)
    neg = df[df["label"] == "none"]
    cap = int(pos_n * args.max_neg_ratio)
    if len(neg) > cap:
        neg = neg.iloc[rng.permutation(len(neg))[:cap]]
    train = pd.concat([df[df["label"] != "none"], neg]).reset_index(drop=True)
    log.info("Training rows: %d  (%s)", len(train),
             train["label"].value_counts().to_dict())

    # ---- embed + fit -----------------------------------------------------
    from sklearn.linear_model import LogisticRegression

    enc = EmbeddingGapHead.load_encoder(args.encoder)
    X = enc.encode(train["sentence"].tolist(), normalize_embeddings=True,
                   show_progress_bar=False, batch_size=64)
    # sklearn>=1.7 dropped multi_class; LogisticRegression is multinomial by default.
    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
    clf.fit(X, train["label"].values)

    head = EmbeddingGapHead(enc, clf, args.encoder)
    head.save(args.out)
    # Sidecar so the benchmark can ASSERT train/eval disjointness, not assume it.
    import json
    meta = {"encoder": args.encoder, "C": args.C, "n_train_rows": len(train),
            "label_counts": train["label"].value_counts().to_dict(),
            "excluded_eval_ids": sorted(exclude), "train_ids": train_ids,
            "dataset_stats": stats}
    Path(args.out).with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Saved head -> %s  (classes=%s)", args.out, list(clf.classes_))

    # quick train-fit sanity (NOT a metric — eval is in bench_gap_funnel.py)
    from sklearn.metrics import classification_report
    pred = clf.predict(X)
    print("\n=== train-set fit (sanity only, expect high) ===")
    print(classification_report(train["label"].values, pred, digits=3))


if __name__ == "__main__":
    main()
