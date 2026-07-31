"""Mine a Tier-0 cue-phrase dictionary from a set of papers (data-driven).

Positives  = LLM-extracted gap sentences   (runs/*/data/gaps.tsv :: gap_sentence)
Negatives  = every other sentence in those papers' extracted sections
             (runs/*/data/sections_extracted.jsonl :: section_text, minus positives)

For each 1..max_n word n-gram that occurs in >= min_support positive sentences,
we compute:
    df_pos     # positive sentences containing it
    df_neg     # negative sentences containing it
    precision  df_pos / (df_pos + df_neg)
    lift       precision / base_rate          (base_rate = N_pos / (N_pos+N_neg))
Keep phrases with lift >= lift_min: these are enriched in gap sentences.

Output (data/tier0_dictionary.json):
    dictionary          full filtered pool          (recall-first — the default)
    dictionary_compact  greedy set-cover subset      (compact, higher filter-rate)
    max_n, thresholds, counts, training coverage

NOTE on label noise: the teacher capped gaps at ~2/paper, so some true gap
sentences sit in the NEGATIVE pool. This depresses absolute lift but preserves
relative ranking; thresholds are deliberately lenient. The bench eval
(scripts/archive/eval_tier0.py) on fully-labeled sentences is the real test.

LEAKAGE: any paper id present in the bench is excluded from mining (asserted).

Usage:
    python scripts/archive/mine_tier0_dictionary.py \
        --runs-glob "runs/*/data" \
        --bench-dir data/bench \
        --out data/tier0_dictionary.json \
        --max-n 3 --min-support 2 --lift-min 1.5 --target-recall 0.97
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from gap2idea.pipeline.gap_prefilter import (  # noqa: E402
    ngrams,
    normalize_text,
    split_sentences,
    tokenize,
)
from gap2idea.utils import get_logger  # noqa: E402

log = get_logger(__name__)


def _has_alpha(phrase: str) -> bool:
    return any(c.isalpha() for c in phrase)


def load_bench_ids(bench_dir: Path) -> set[str]:
    bp = bench_dir / "bench_papers.jsonl"
    ids: set[str] = set()
    if bp.exists():
        for line in bp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ids.add(str(json.loads(line)["id"]))
    return ids


def collect_corpus(
    run_dirs: list[Path], bench_ids: set[str]
) -> tuple[list[str], list[str], dict]:
    """Return (positive_sentences, negative_sentences, stats).

    Positives are deduped by normalized form. Negatives are section sentences
    whose normalized form is not a positive.
    """
    pos_norm: dict[str, str] = {}      # norm -> raw (deduped positives)
    excluded_papers: set[str] = set()
    pos_papers: set[str] = set()

    for d in run_dirs:
        gaps_path = d / "gaps.tsv"
        if not gaps_path.exists():
            continue
        df = pd.read_csv(gaps_path, sep="\t", dtype={"id": str})
        for _, r in df.iterrows():
            pid = str(r["id"])
            if pid in bench_ids:
                excluded_papers.add(pid)
                continue
            pos_papers.add(pid)
            sent = str(r.get("gap_sentence") or "")
            norm = normalize_text(sent)
            if len(norm) >= 8:
                pos_norm.setdefault(norm, sent)

    positives = list(pos_norm.values())
    pos_norm_set = set(pos_norm.keys())

    neg_norm: dict[str, str] = {}
    for d in run_dirs:
        sec_path = d / "sections_extracted.jsonl"
        if not sec_path.exists():
            continue
        for line in sec_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            pid = str(rec.get("id"))
            if pid in bench_ids:
                excluded_papers.add(pid)
                continue
            for sent in split_sentences(rec.get("section_text") or ""):
                norm = normalize_text(sent)
                if len(norm) < 8 or norm in pos_norm_set:
                    continue
                neg_norm.setdefault(norm, sent)

    negatives = list(neg_norm.values())
    stats = {
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "n_positive_papers": len(pos_papers),
        "n_excluded_bench_papers": len(excluded_papers),
        "excluded_bench_papers": sorted(excluded_papers),
    }
    return positives, negatives, stats


def document_frequency(sentences: list[str], max_n: int, restrict: set[str] | None = None
                       ) -> dict[str, int]:
    """df[phrase] = # sentences whose n-gram set contains phrase. If `restrict`
    is given, only count phrases in that set (faster for the negative pass)."""
    df: dict[str, int] = defaultdict(int)
    for s in sentences:
        grams = ngrams(tokenize(s), 1, max_n)
        if restrict is not None:
            grams &= restrict
        for g in grams:
            df[g] += 1
    return df


def mine(
    positives: list[str],
    negatives: list[str],
    max_n: int,
    min_support: int,
    lift_min: float,
    target_recall: float,
) -> tuple[list[dict], list[dict], dict]:
    n_pos, n_neg = len(positives), len(negatives)
    base_rate = n_pos / (n_pos + n_neg) if (n_pos + n_neg) else 0.0

    # 1. positive df, keep candidates with support and an alphabetic char
    df_pos_all = document_frequency(positives, max_n)
    candidates = {
        p for p, c in df_pos_all.items() if c >= min_support and _has_alpha(p)
    }
    log.info("candidates after support>=%d: %d (from %d positive n-grams)",
             min_support, len(candidates), len(df_pos_all))

    # 2. negative df for candidates only
    df_neg = document_frequency(negatives, max_n, restrict=candidates)

    # 3. score + lift filter
    pool: list[dict] = []
    for p in candidates:
        dp = df_pos_all[p]
        dn = df_neg.get(p, 0)
        prec = dp / (dp + dn) if (dp + dn) else 0.0
        lift = prec / base_rate if base_rate else 0.0
        if lift >= lift_min:
            pool.append({
                "phrase": p,
                "df_pos": dp,
                "df_neg": dn,
                "precision": round(prec, 4),
                "lift": round(lift, 3),
                "n": len(p.split()),
            })
    # rank: coverage first, then discrimination
    pool.sort(key=lambda x: (x["df_pos"], x["lift"]), reverse=True)
    log.info("pool after lift>=%.2f: %d phrases", lift_min, len(pool))

    # 4. precompute each pool phrase's covered positive-sentence indices
    phrase_set = {p["phrase"] for p in pool}
    covers: dict[str, set[int]] = defaultdict(set)
    for i, s in enumerate(positives):
        grams = ngrams(tokenize(s), 1, max_n) & phrase_set
        for g in grams:
            covers[g].add(i)

    # 5. greedy set-cover to target recall (compact dictionary)
    target_count = int(round(target_recall * n_pos))
    uncovered = set(range(n_pos))
    chosen: list[str] = []
    remaining = {p["phrase"]: set(covers[p["phrase"]]) for p in pool}
    while uncovered and (n_pos - len(uncovered)) < target_count:
        best_p, best_gain = None, 0
        for p, cov in remaining.items():
            gain = len(cov & uncovered)
            if gain > best_gain:
                best_gain, best_p = gain, p
        if not best_p or best_gain == 0:
            break
        chosen.append(best_p)
        uncovered -= remaining[best_p]
        del remaining[best_p]

    chosen_set = set(chosen)
    compact = [p for p in pool if p["phrase"] in chosen_set]
    compact.sort(key=lambda x: (x["df_pos"], x["lift"]), reverse=True)

    # training-set diagnostics
    covered_pool = set()
    for p in pool:
        covered_pool |= covers[p["phrase"]]
    train = {
        "base_rate": round(base_rate, 5),
        "pool_size": len(pool),
        "compact_size": len(compact),
        "pool_train_recall": round(len(covered_pool) / n_pos, 4) if n_pos else 0.0,
        "compact_train_recall": round((n_pos - len(uncovered)) / n_pos, 4) if n_pos else 0.0,
    }
    return pool, compact, train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="runs/*/data")
    ap.add_argument("--bench-dir", type=Path, default=Path("data/bench"))
    ap.add_argument("--out", type=Path, default=Path("data/tier0_dictionary.json"))
    ap.add_argument("--max-n", type=int, default=3)
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--lift-min", type=float, default=1.5)
    ap.add_argument("--target-recall", type=float, default=0.97)
    args = ap.parse_args()

    run_dirs = [Path(p) for p in sorted(glob.glob(args.runs_glob)) if Path(p).is_dir()]
    if not run_dirs:
        raise SystemExit(f"No run dirs matched {args.runs_glob!r}")
    log.info("Mining from %d run dirs: %s", len(run_dirs), [str(d) for d in run_dirs])

    bench_ids = load_bench_ids(args.bench_dir)
    log.info("Bench has %d paper ids (will be excluded from mining)", len(bench_ids))

    positives, negatives, stats = collect_corpus(run_dirs, bench_ids)
    log.info("Corpus: %d positives (%d papers), %d negatives. Excluded bench papers: %d",
             stats["n_positives"], stats["n_positive_papers"],
             stats["n_negatives"], stats["n_excluded_bench_papers"])

    # Hard leakage assertion
    assert stats["n_excluded_bench_papers"] == 0, (
        f"LEAKAGE: bench papers found in mining corpus: {stats['excluded_bench_papers']}"
    )
    if stats["n_positives"] < 20:
        log.warning("Only %d positives — dictionary may be thin", stats["n_positives"])

    pool, compact, train = mine(
        positives, negatives,
        max_n=args.max_n, min_support=args.min_support,
        lift_min=args.lift_min, target_recall=args.target_recall,
    )

    out = {
        "max_n": args.max_n,
        "thresholds": {
            "min_support": args.min_support,
            "lift_min": args.lift_min,
            "target_recall": args.target_recall,
        },
        "corpus": stats,
        "training": train,
        "dictionary": pool,            # recall-first default
        "dictionary_compact": compact, # greedy set-cover subset
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s", args.out)

    log.info("Training: base_rate=%.4f  pool=%d (recall=%.3f)  compact=%d (recall=%.3f)",
             train["base_rate"], train["pool_size"], train["pool_train_recall"],
             train["compact_size"], train["compact_train_recall"])
    log.info("Top 25 phrases by coverage:")
    for p in pool[:25]:
        log.info("  %-28s df_pos=%-3d df_neg=%-4d lift=%.2f", p["phrase"], p["df_pos"], p["df_neg"], p["lift"])


if __name__ == "__main__":
    main()
