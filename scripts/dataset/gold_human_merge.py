# -*- coding: utf-8 -*-
"""Merge the two human annotators' answers against the PRIVATE key and score the gap gold.

Inputs (any of):
  * two per-annotator exports from annotate.html / the Sheet:
      scripts/.../gold_human_merge.py A1.xlsx A2.xlsx
    (columns: row, is_gap, type, notes[, annotator]; .xlsx or .csv)
  * one combined Google-Sheet export with A1_*/A2_* columns:
      gold_human_merge.py combined.xlsx
  * no args -> defaults to data/gold/gap_annotations_A1.xlsx and _A2.xlsx

Needs data/gold/worksheet_key_PRIVATE.csv (row -> source, paper_id, idx) and, optionally,
data/gold/gap_gold_final.jsonl (for human-vs-our-type agreement).

Reports: coverage, control sanity (ctrl_pos should be Y, neg should be N), inter-annotator
Cohen's kappa (is_gap + type), the empirical precision of our "extra" limitations (the headline
validity number vs Fable's 51%), and human-adjudicated precision of OURS vs SAL on the pool.
Writes docs/experiments/human_validation_results.md.
"""
import sys, json, csv, re
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "gold"
KEY = GOLD / "worksheet_key_PRIVATE.csv"
OUTDOC = ROOT / "docs" / "experiments" / "human_validation_results.md"

YES = {"y", "yes", "1", "true", "gap"}
NO = {"n", "no", "0", "false"}
TYPES = ("limitation", "future_work", "research_gap")


def norm_yn(v):
    s = str(v or "").strip().lower()
    if s in YES: return "Y"
    if s in NO: return "N"
    return ""


def norm_type(v):
    s = str(v or "").strip().lower().replace(" ", "_")
    if s in TYPES: return s
    if s in ("none", "na", "n/a"): return "none"
    return ""


def read_rows(path):
    """Return list of dict rows from .xlsx or .csv (header-based)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb[wb.sheetnames[0]]
        it = ws.iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(it)]
        return [dict(zip(header, r)) for r in it]
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def col(d, *names):
    """First matching column (case/space/paren-insensitive)."""
    def k(s): return re.sub(r"[^a-z0-9]", "", str(s).lower())
    want = {k(n) for n in names}
    for key in d:
        if k(key) in want:
            return d[key]
    return None


def load_annotations(args):
    """-> {'A1': {row:{is_gap,type}}, 'A2': {...}}  (rows are ints)."""
    out = {"A1": {}, "A2": {}}
    # combined single file?
    if len(args) == 1:
        rows = read_rows(args[0])
        if rows and col(rows[0], "A1_is_gap", "A1isgapYN", "A1_is_gap(Y/N)") is not None:
            for r in rows:
                rn = col(r, "row")
                if rn in (None, ""): continue
                rn = int(rn)
                out["A1"][rn] = dict(is_gap=norm_yn(col(r, "A1_is_gap", "A1_is_gap(Y/N)")),
                                     type=norm_type(col(r, "A1_type")))
                out["A2"][rn] = dict(is_gap=norm_yn(col(r, "A2_is_gap", "A2_is_gap(Y/N)")),
                                     type=norm_type(col(r, "A2_type")))
            return out
    # per-annotator files (ordered A1, A2) or auto by 'annotator' col
    labels = ["A1", "A2"]
    for n, path in enumerate(args):
        rows = read_rows(path)
        for r in rows:
            rn = col(r, "row")
            if rn in (None, ""): continue
            who = str(col(r, "annotator") or labels[n] if n < len(labels) else f"A{n+1}").strip() or labels[min(n, 1)]
            if who not in out: out[who] = {}
            out[who][int(rn)] = dict(is_gap=norm_yn(col(r, "is_gap", "is_gap(Y/N)")),
                                     type=norm_type(col(r, "type")))
    return out


def kappa(pairs):
    """Cohen's kappa for a list of (a,b) label pairs."""
    pairs = [(a, b) for a, b in pairs if a and b]
    if not pairs: return float("nan"), 0, float("nan")
    labels = sorted({l for p in pairs for l in p})
    if len(labels) < 2:  # kappa undefined with a single label; report agreement only
        return float("nan"), len(pairs), 1.0
    try:
        from sklearn.metrics import cohen_kappa_score
        a, b = zip(*pairs)
        k = cohen_kappa_score(a, b, labels=labels)
    except Exception:
        n = len(pairs); po = sum(1 for a, b in pairs if a == b) / n
        ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
        pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
        k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    agree = sum(1 for a, b in pairs if a == b) / len(pairs)
    return k, len(pairs), agree


def pct(n, d):
    return f"{100*n/d:.1f}% ({n}/{d})" if d else "n/a (0)"


def main():
    args = sys.argv[1:]
    if not args:
        args = [str(GOLD / "gap_annotations_A1.xlsx"), str(GOLD / "gap_annotations_A2.xlsx")]

    if not KEY.exists():
        sys.exit(f"missing private key: {KEY}")
    key = {}
    with KEY.open(encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            key[int(r["row"])] = dict(source=r["source"], paper_id=r["paper_id"], idx=int(r["idx"]))
    n_total = len(key)
    by_src = defaultdict(list)
    for rn, v in key.items():
        by_src[v["source"]].append(rn)

    present = [a for a in args if Path(a).exists()]
    if not present:
        print("Annotation files not found yet. Expected (export from annotate.html):")
        for a in args: print("   ", a)
        print(f"\nPrivate key OK: {n_total} rows "
              f"(extras={len(by_src['extra'])}, missed={len(by_src['missed'])}, "
              f"ctrl_pos={len(by_src['ctrl_pos'])}, neg={len(by_src['neg'])}).")
        print("Re-run this script once both annotators have exported their .xlsx.")
        return

    ann = load_annotations(present)
    A1, A2 = ann.get("A1", {}), ann.get("A2", {})

    # optional: our per-sentence gold type (for human-vs-our-type)
    our_type = {}
    f = GOLD / "gap_gold_final.jsonl"
    if f.exists():
        for ln in f.read_text(encoding="utf-8").splitlines():
            if not ln.strip(): continue
            r = json.loads(ln)
            our_type[(r["paper_id"], r["idx"])] = r.get("label")

    L = []
    L.append("# Human validation — results\n")
    L.append(f"Pool: **{n_total} rows** — extras {len(by_src['extra'])}, SAL-missed {len(by_src['missed'])}, "
             f"control-positives {len(by_src['ctrl_pos'])}, negatives {len(by_src['neg'])}.\n")
    L.append(f"Annotators present: {', '.join(a for a in ('A1','A2') if ann.get(a))}. "
             f"Coverage — A1 {pct(sum(1 for r in A1.values() if r['is_gap']), n_total)}, "
             f"A2 {pct(sum(1 for r in A2.values() if r['is_gap']), n_total)}.\n")

    # ---- control sanity ----
    L.append("\n## 1. Control sanity (annotator reliability)\n")
    L.append("| set | expected | A1 %Y | A2 %Y | consensus %Y |\n|---|---|---|---|---|")
    def rate(rows, who):
        d = [who[r]["is_gap"] for r in rows if who.get(r, {}).get("is_gap")]
        y = sum(1 for x in d if x == "Y")
        return pct(y, len(d))
    def cons_rate(rows):
        y = tot = 0
        for r in rows:
            a, b = A1.get(r, {}).get("is_gap"), A2.get(r, {}).get("is_gap")
            if a and b and a == b:
                tot += 1; y += (a == "Y")
        return pct(y, tot)
    for src, exp in (("ctrl_pos", "mostly Y"), ("neg", "mostly N")):
        rows = by_src[src]
        L.append(f"| {src} | {exp} | {rate(rows,A1)} | {rate(rows,A2)} | {cons_rate(rows)} |")

    # ---- inter-annotator agreement ----
    both = [r for r in key if A1.get(r, {}).get("is_gap") and A2.get(r, {}).get("is_gap")]
    kg, ng, ag = kappa([(A1[r]["is_gap"], A2[r]["is_gap"]) for r in both])
    ty_pairs = [(A1[r]["type"], A2[r]["type"]) for r in both
                if A1[r]["is_gap"] == "Y" and A2[r]["is_gap"] == "Y" and A1[r]["type"] and A2[r]["type"]]
    kt, nt, at = kappa(ty_pairs)
    L.append("\n## 2. Inter-annotator agreement\n")
    L.append(f"- **is_gap:** Cohen's kappa = **{kg:.3f}**, raw agreement {ag*100:.1f}% (n={ng} both-answered).")
    L.append(f"- **type** (both said Y): kappa = **{kt:.3f}**, raw {at*100:.1f}% (n={nt}).")

    # ---- headline: human adjudication of our 'extra' limitations ----
    ex = by_src["extra"]
    ex_both = [r for r in ex if A1.get(r, {}).get("is_gap") and A2.get(r, {}).get("is_gap")]
    ex_cons_y = sum(1 for r in ex_both if A1[r]["is_gap"] == "Y" and A2[r]["is_gap"] == "Y")
    ex_any_y = sum(1 for r in ex_both if "Y" in (A1[r]["is_gap"], A2[r]["is_gap"]))
    L.append("\n## 3. Validity of our *extra* limitations (headline)\n")
    L.append("These are limitations we labeled that SAL did **not** (the contested set; Fable confirmed 51%).")
    L.append(f"- Human **both agree Y** (strict): **{pct(ex_cons_y, len(ex_both))}**.")
    L.append(f"- Human **at least one Y** (lenient): {pct(ex_any_y, len(ex_both))}.")
    L.append("- Interpretation: strict-Y = real SAL under-labels our gold rightly caught; "
             "the rest are over-labels to drop from the high-precision tier.")

    # ---- SAL-missed: did we miss real gaps? ----
    ms = by_src["missed"]
    ms_both = [r for r in ms if A1.get(r, {}).get("is_gap") and A2.get(r, {}).get("is_gap")]
    ms_cons_y = sum(1 for r in ms_both if A1[r]["is_gap"] == "Y" and A2[r]["is_gap"] == "Y")
    L.append("\n## 4. SAL-missed set (our recall holes)\n")
    L.append(f"- Human both-agree Y: {pct(ms_cons_y, len(ms_both))} "
             "(high = these are genuine gaps both our judges missed).")

    # ---- system comparison on human-adjudicated pool ----
    def human_truth(r):
        a, b = A1.get(r, {}).get("is_gap"), A2.get(r, {}).get("is_gap")
        if a and b and a == b: return a
        return None  # undecided / disagree
    ours_pos = set(by_src["ctrl_pos"]) | set(by_src["extra"])      # our gold positives in pool
    sal_pos = set(by_src["ctrl_pos"]) | set(by_src["missed"])      # SAL gold positives in pool
    def precision(posset):
        y = d = 0
        for r in posset:
            t = human_truth(r)
            if t: d += 1; y += (t == "Y")
        return y, d
    oy, od = precision(ours_pos); sy, sd = precision(sal_pos)
    L.append("\n## 5. Human-adjudicated precision on the pool (consensus rows)\n")
    L.append("| system | positives judged | human-confirmed precision |\n|---|---|---|")
    L.append(f"| **ours** (ctrl_pos+extra) | {od} | **{pct(oy, od)}** |")
    L.append(f"| **SAL** (ctrl_pos+missed) | {sd} | {pct(sy, sd)} |")
    neg = by_src["neg"]
    neg_y = sum(1 for r in neg if human_truth(r) == "Y")
    neg_d = sum(1 for r in neg if human_truth(r))
    L.append(f"\nNegatives flagged as gaps by humans: {pct(neg_y, neg_d)} "
             "(contamination / gaps every system missed).")

    # ---- human-vs-our-type (optional) ----
    if our_type:
        agree = tot = 0
        for r in (set(by_src["extra"]) | set(by_src["ctrl_pos"])):
            k = (key[r]["paper_id"], key[r]["idx"]); ot = our_type.get(k)
            a, b = A1.get(r, {}), A2.get(r, {})
            if ot and a.get("is_gap") == "Y" == b.get("is_gap") and a.get("type") and a["type"] == b["type"]:
                tot += 1; agree += (a["type"] == ot)
        if tot:
            L.append("\n## 6. Human-vs-our type (where humans agree on a type)\n")
            L.append(f"- Our gold type matches the human-agreed type: {pct(agree, tot)}.")

    doc = "\n".join(L) + "\n"
    OUTDOC.parent.mkdir(parents=True, exist_ok=True)
    OUTDOC.write_text(doc, encoding="utf-8")
    print(doc)
    print(f"\nwrote {OUTDOC}")


if __name__ == "__main__":
    main()
