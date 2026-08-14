"""Reconstruct Lan et al. (2024)'s section-filtered candidate pool for the RCT/SAL
benchmark, so our detector is judged on THEIR evaluation condition (~20% positive),
not the full-article stream (~2.6%).

Their SAL classifier is run over "the abstract and discussion- and limitation-related
sections" (keywords: discussion, limitation, weakness, conclusion, caveat, shortcoming,
drawback). We recover the section of every test sentence from the PubMed Central BioC
API (which tags each passage with a section_type) and keep sentences whose section is
ABSTRACT / DISCUSS / CONCL -- the pool they actually evaluate on.

Writes a pool CSV (pmcid, section, label, sentence) and prints:
  - pool size + prevalence (should be ~20%, matching their 20.7%)
  - the section filter's recall over gold positives (what fraction of true
    limitations survive the filter -- the ceiling any detector can reach on the pool)

    python scripts/bench/build_rct_pool.py --split test --out data/rct_pool_test.csv
"""
from __future__ import annotations
import argparse, csv, io, json, sys, time, urllib.request
from pathlib import Path
import collections

RAW = "https://raw.githubusercontent.com/MengfeiLan/SAL_Type_Classification/main/data/"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/%s/unicode"
# BioC section_type tags that correspond to their keyword-filtered sections.
POOL_SECTIONS = {"ABSTRACT", "DISCUSS", "CONCL"}
MATCH_PREFIX = 50  # chars of a sentence used to locate its passage


def fetch_rows(split):
    raw = urllib.request.urlopen(RAW + split + ".csv", timeout=60).read().decode("utf-8", "replace")
    return list(csv.DictReader(io.StringIO(raw)))


def bioc_sections(pmcid, cache: Path):
    """Return list of (section_type, passage_text) for a PMC article, cached to disk."""
    pid = pmcid if str(pmcid).upper().startswith("PMC") else "PMC" + str(pmcid)
    cf = cache / f"{pid}.json"
    if cf.exists():
        doc = json.loads(cf.read_text(encoding="utf-8"))
    else:
        for attempt in range(3):
            try:
                doc = json.load(urllib.request.urlopen(BIOC % pid, timeout=90))
                cf.write_text(json.dumps(doc), encoding="utf-8")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"  !! BioC failed for {pid}: {type(e).__name__}", flush=True)
                    return []
                time.sleep(2)
    d = doc[0]["documents"][0]
    return [(p["infons"].get("section_type", "?"), p.get("text", "")) for p in d["passages"]]


def section_of(sentence, passages):
    key = " ".join(sentence.split())[:MATCH_PREFIX]
    for sect, text in passages:
        if key and key in " ".join(text.split()):
            return sect
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default="data/bioc_cache")
    args = ap.parse_args()
    ROOT = Path(__file__).resolve().parents[2]
    cache = (ROOT / args.cache); cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else ROOT / f"data/rct_pool_{args.split}.csv"

    rows = fetch_rows(args.split)
    pmcids = list(dict.fromkeys(r["pmcids"] for r in rows))
    print(f"{args.split}: {len(rows)} sentences, {len(pmcids)} articles", flush=True)

    sect_cache = {}
    for i, pid in enumerate(pmcids):
        sect_cache[pid] = bioc_sections(pid, cache)
        if (i + 1) % 10 == 0:
            print(f"  fetched {i+1}/{len(pmcids)}", flush=True)

    pool, sec_counts, unmatched = [], collections.Counter(), 0
    pos_total = pos_in_pool = 0
    for r in rows:
        sent = (r.get("sentences") or "").strip()
        if len(sent.split()) < 3:
            continue
        label = 0 if (r.get("category_spans", "[0]") or "[0]").strip() == "[0]" else 1
        sect = section_of(sent, sect_cache.get(r["pmcids"], []))
        if sect == "?":
            unmatched += 1
        sec_counts[sect] += 1
        pos_total += label
        if sect in POOL_SECTIONS:
            pos_in_pool += label
            pool.append((r["pmcids"], sect, label, sent))

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["pmcid", "section", "label", "sentence"])
        w.writerows(pool)

    n = len(pool); npos = sum(p[2] for p in pool)
    print(f"\nsection distribution (all sentences): {dict(sec_counts.most_common())}")
    print(f"unmatched sentences (section '?'): {unmatched}")
    print(f"\nPOOL (sections {sorted(POOL_SECTIONS)}): {n} sentences, "
          f"{npos} limitations = {npos/max(n,1):.1%} prevalence")
    print(f"section-filter recall over gold positives: {pos_in_pool}/{pos_total} "
          f"= {pos_in_pool/max(pos_total,1):.1%}  (ceiling for any detector on this pool)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
