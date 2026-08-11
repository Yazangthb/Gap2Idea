"""Benchmark the funnel (Stage A+B, then Stage C) on the BAGELS ACL subsets.

BAGELS (arXiv 2505.18207, EMNLP-F 2025) ground truth is VERBATIM limitation text
(strict no-paraphrasing extraction), so we score with the same scramble-robust
token-containment used elsewhere — this is BAGELS' "Ground-Truth Coverage".

We use ACL_23/24_with_limitation: each record has section text + a `Limitation`
gold field (the paper's mandated Limitations section). We reconstruct the paper
sections (incl. the Limitations section, as in the real PDF), run the funnel via
its GROBID-section path, and measure how many gold limitation sentences we
recover BEFORE and AFTER Stage C. NeurIPS/PeerJ gold mixes in OpenReview text
that isn't in the paper, so it's excluded from this extraction test.

    python scripts/bench/bench_bagels.py --n 60          # 60 papers / subset

Coverage here is a LOWER BOUND on BAGELS' BERTScore-coverage (verbatim match is
stricter than semantic match). Sections are cleanly labelled, so Stage-A
localization is optimistic vs a raw PDF — the honest signals are Stage-C recall
safety and the FP drop.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gap2idea.pipeline.gap_funnel import (  # noqa: E402
    EmbeddingGapHead, extract_gaps, token_containment,
)
from gap2idea.pipeline.gap_prefilter import split_sentences  # noqa: E402
from gap2idea.pipeline.gap_llm_filter import LLMGapFilter  # noqa: E402

REPO = "IbrahimAlAzhar/limitation-generation-dataset-bagels"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CACHE = ROOT / "artifacts" / "bagels_cache"
SUBSETS = ["ACL_23_with_limitation", "ACL_24_with_limitation"]
MATCH_TAU = 0.80
# The Limitations section is keyed by its real heading, inconsistently across
# records: "Limitation", "Limitations", "7 Limitations", "A.3 Corpus Limitations".
# ACL responsible-research checklist keys ("... Did you describe the limitations
# ...?", value just "Section 7") also contain the word — exclude those.
_LIM_KEY = re.compile(r"\blimitations?\b", re.I)
_CHECKLIST = re.compile(r"\bdid you\b|\?", re.I)


def gold_field(obj: dict) -> str:
    """The verbatim Limitations section text: the longest limitation-headed value
    that isn't a checklist question or a short 'Section N' pointer."""
    cands = [v for k, v in obj.items()
             if isinstance(v, str) and _LIM_KEY.search(k)
             and not _CHECKLIST.search(k) and len(v.strip()) >= 60]
    return max(cands, key=len) if cands else ""


def _get(url: str, retries: int = 3):
    req = urllib.request.Request(url, headers={"User-Agent": "bagels-bench"})
    for a in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8"), r.headers.get("Link", "")
        except Exception:
            if a == retries - 1:
                raise
            time.sleep(1.5)


def list_files(subdir: str) -> list[str]:
    files, url = [], f"{TREE}/{subdir}?recursive=false"
    while url:
        body, link = _get(url)
        files += [e["path"] for e in json.loads(body)
                  if e.get("type") == "file" and e["path"].endswith(".json")]
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                s, e = part.find("<"), part.find(">")
                if s != -1 and e != -1:
                    url = part[s + 1:e]
    return files


def fetch(path: str):
    cache = CACHE / path.replace("/", "__")
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        body, _ = _get(f"{RESOLVE}/{path}")
        d = json.loads(body)
    except Exception:
        return None
    cache.mkdir(parents=True, exist_ok=True) if False else cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(d), encoding="utf-8")
    return d


def to_sections(obj: dict) -> list[dict]:
    """Reconstruct the paper's sections (the limitations section is already present
    under its own heading); drop Title/File-Number and ACL checklist Q&A keys."""
    secs, i = [], 0
    if obj.get("abstractText", "").strip():
        secs.append({"n": str(i), "heading": "Abstract", "text": obj["abstractText"]}); i += 1
    for k, v in obj.items():
        if k in ("File Number", "Title", "abstractText"):
            continue
        if not isinstance(v, str) or not v.strip() or _CHECKLIST.search(k):
            continue
        secs.append({"n": str(i), "heading": k, "text": v}); i += 1
    return secs


_Q_STEM = re.compile(r"(?i)^(did|have|do|does|are|is|was|were|can|could|would|will|should)\s+you\b")
_PTR = re.compile(r"(?i)^(section|sections|appendix|figure|table|last paragraph)\b")


def gold_sentences(obj: dict) -> list[str]:
    """Verbatim limitation sentences, cleaned of ACL-checklist Q&A noise that some
    records mis-file under a 'Limitations' heading (questions, 'Section N' pointers)."""
    raw = gold_field(obj)
    if raw.count("?") >= 3:          # the field is an ACL checklist block, not a limitations section
        return []
    lim = re.sub(r"^\s*limitations?\s*\n?", "", raw, flags=re.I)
    out = []
    for s in split_sentences(lim):
        s = s.strip()
        if len(s.split()) < 5 or s.endswith("?") or _Q_STEM.match(s) or _PTR.match(s):
            continue
        out.append(s)
    return out


def covered(g: str, preds: list[dict]) -> bool:
    return any(max(token_containment(g, p["gap_sentence"]),
                   token_containment(p["gap_sentence"], g)) >= MATCH_TAU for p in preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="papers per subset")
    ap.add_argument("--backend", default="api")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    head = EmbeddingGapHead.load(ROOT / "data/gap_head.joblib")
    filt = LLMGapFilter(backend=args.backend, mode="validate")

    per_subset, agg = {}, {}
    for sub in SUBSETS:
        print(f"[{sub}] listing files ...", flush=True)
        files = list_files(sub)
        random.Random(args.seed).shuffle(files)
        picked = files[: args.n]
        print(f"[{sub}] {len(files)} available, downloading {len(picked)} ...", flush=True)
        with cf.ThreadPoolExecutor(max_workers=8) as ex:
            objs = list(ex.map(fetch, picked))

        tot_gold = cov_b = cov_a = np_b = np_a = n_papers = 0
        for obj in objs:
            if not obj:
                continue
            gold = gold_sentences(obj)   # handles the varied keys + checklist filtering
            if not gold:
                continue
            n_papers += 1
            secs = to_sections(obj)
            text = " ".join(s["text"] for s in secs)
            preds = extract_gaps(str(obj.get("File Number", n_papers)), text,
                                 head=head, mode="hybrid", grobid_sections=secs)
            kept = filt.filter_gaps([dict(p) for p in preds])
            tot_gold += len(gold)
            cov_b += sum(covered(g, preds) for g in gold)
            cov_a += sum(covered(g, kept) for g in gold)
            np_b += len(preds); np_a += len(kept)
        per_subset[sub] = dict(papers=n_papers, gold=tot_gold, cov_b=cov_b, cov_a=cov_a,
                               np_b=np_b, np_a=np_a)
        for k, v in per_subset[sub].items():
            agg[k] = agg.get(k, 0) + v
        print(f"[{sub}] papers={n_papers} gold_sents={tot_gold} "
              f"coverage {cov_b/max(1,tot_gold):.3f}->{cov_a/max(1,tot_gold):.3f} "
              f"preds {np_b}->{np_a}", flush=True)

    def line(name, d):
        return (f"| {name} | {d['papers']} | {d['gold']} | "
                f"{d['cov_b']/max(1,d['gold']):.3f} | {d['cov_a']/max(1,d['gold']):.3f} | "
                f"{d['np_b']} | {d['np_a']} |")

    R = ["# BAGELS extraction benchmark (ACL, verbatim gold coverage)", "",
         f"Backend {args.backend} ({filt.model}); {filt.n_judged} judged in {filt.n_calls} calls. "
         f"Match = token-containment >= {MATCH_TAU} (lower bound on BERTScore-coverage).", "",
         "| subset | papers | gold sents | coverage before | coverage after | preds before | preds after |",
         "|---|---|---|---|---|---|---|"]
    R += [line(s, per_subset[s]) for s in SUBSETS]
    R.append(line("**ALL**", agg))
    R += ["",
          f"**Extraction recall (Stage A+B): {agg['cov_b']/max(1,agg['gold']):.3f}** of verbatim gold "
          f"limitation sentences recovered — the BAGELS-comparable coverage number.", "",
          f"Stage C then drops {100*(agg['np_b']-agg['np_a'])/max(1,agg['np_b']):.0f}% of predictions "
          f"({agg['np_b']}->{agg['np_a']}) and coverage falls to {agg['cov_a']/max(1,agg['gold']):.3f}. "
          "This gold is limitations-only (no negatives), so Stage C — a precision filter — can only cost "
          "coverage here; its FP-removal value shows on full-paper benchmarks, not this one.", "",
          "Caveats: sections are cleanly labelled (Stage-A localization is easier than on raw PDFs); "
          "verbatim token-containment is a lower bound on BAGELS' BERTScore-coverage; BAGELS gold includes "
          "non-canonical limitations-section sentences (method observations) that a precision filter drops."]
    out = ROOT / "docs/experiments/bagels_output.md"
    out.write_text("\n".join(R), encoding="utf-8")
    print("\n" + "\n".join(R[4:]))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
