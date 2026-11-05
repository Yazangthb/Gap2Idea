# -*- coding: utf-8 -*-
"""Score the human study. Design: docs/experiments/human_study_chosen_design.md

  tiebreak R1.json R2.json                 write the tie-breaker page for disputed items
  analyze R1.json R2.json [TB.json] --out FILE.md
  simulate DIR                             write fake answers for a dry run; real files are untouched

Rejects files from another sample, incomplete files and repeated raters. Claims need the decoy gate to pass.
"""
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "dataset"))
from human_study_sample import OUT, SEED, write_page

TYPES = {"limitation", "future_work", "research_gap"}
PRECISION_BAR = 0.70
MAX_DECOY_YES = 2
Z_ONE_SIDED = 1.645
TRACKS = {"arxiv": ("B1", "B2"), "sal": ("S1", "S2", "S3")}


def load_key():
    raw = (OUT / "key_PRIVATE.json").read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_answers(path, expected_ids, sample_hash):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("sample_hash") != sample_hash:
        sys.exit(f"{path}: comes from a different sample")
    answers = data.get("answers", {})
    missing, foreign = expected_ids - set(answers), set(answers) - expected_ids
    if missing or foreign:
        sys.exit(f"{path}: {len(missing)} items missing, {len(foreign)} items not in this sample")
    for item_id, a in answers.items():
        if not isinstance(a.get("is_gap"), bool) or (a["is_gap"] and a.get("type") not in TYPES):
            sys.exit(f"{path}: {item_id} has an invalid answer")
    return data["rater"], answers


def disputed(a, b):
    return a["is_gap"] != b["is_gap"] or (a["is_gap"] and a["type"] != b["type"])


def two_raters(p1, p2, key, sample_hash):
    ids = {x["id"] for x in key}
    r1, a1 = read_answers(p1, ids, sample_hash)
    r2, a2 = read_answers(p2, ids, sample_hash)
    if r1 == r2:
        sys.exit(f"Both files come from rater {r1}")
    return a1, a2, sorted(i for i in ids if disputed(a1[i], a2[i]))


def final_labels(key, a1, a2, tb):
    out = {}
    for item in key:
        i = item["id"]
        votes = [a1[i], a2[i]] + ([tb[i]] if i in tb else [])
        yes = [v for v in votes if v["is_gap"]]
        is_gap = len(yes) * 2 > len(votes)  # an unresolved item counts as not a gap
        kind = None
        if is_gap:
            ranked = Counter(v["type"] for v in yes).most_common()
            kind = ranked[0][0] if len(ranked) == 1 or ranked[0][1] > ranked[1][1] else "unresolved"
        out[i] = {"is_gap": is_gap, "type": kind}
    return out


def alpha(pairs):
    """Krippendorff's alpha, nominal data, two raters, no missing values."""
    n = 2 * len(pairs)
    totals = Counter(v for pair in pairs for v in pair)
    if n < 2 or len(totals) < 2:
        return None
    observed = sum(a != b for a, b in pairs) * 2 / n
    expected = sum(totals[x] * totals[y] for x in totals for y in totals if x != y) / (n * (n - 1))
    return round(1 - observed / expected, 3)


def weighted(shares, populations):
    """shares: stratum -> (correct, n). Stratified estimate with finite-population variance."""
    total = sum(populations[s] for s in shares)
    est = variance = 0.0
    for s, (correct, n) in shares.items():
        w, p = populations[s] / total, correct / n
        est += w * p
        if n > 1:
            variance += w * w * p * (1 - p) / (n - 1) * max(0.0, 1 - n / populations[s])
    return est, math.sqrt(variance)


def binomial_p(k, n, p0=0.5):
    return sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1))


def analyze(argv):
    if "--out" not in argv:
        sys.exit("Give --out FILE.md")
    at = argv.index("--out")
    out_path, files = Path(argv[at + 1]), argv[:at] + argv[at + 2:]
    key, sample_hash = load_key()
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sample_hash"] != sample_hash:
        sys.exit("The key does not match the manifest, so it changed after the sample was frozen")
    a1, a2, dispute = two_raters(files[0], files[1], key, sample_hash)
    tb = {}
    if dispute:
        if len(files) < 3:
            sys.exit(f"{len(dispute)} disputed items need the tie-breaker file")
        rt, tb = read_answers(files[2], set(dispute), sample_hash)
        if rt != "TB":
            sys.exit(f"{files[2]} is not from the tie-breaker")
    labels = final_labels(key, a1, a2, tb)
    by = {}
    for item in key:
        by.setdefault(item["stratum"], []).append(item)
    pop = manifest["populations"]

    decoy_yes = sum(labels[x["id"]]["is_gap"] for x in by["D"])
    gate = decoy_yes <= MAX_DECOY_YES
    pairs_gap = [(a1[x["id"]]["is_gap"], a2[x["id"]]["is_gap"]) for x in key]
    pairs_type = [(a1[x["id"]].get("type", "none") if a1[x["id"]]["is_gap"] else "none",
                   a2[x["id"]].get("type", "none") if a2[x["id"]]["is_gap"] else "none") for x in key]

    any_gap = {s: (sum(labels[x["id"]]["is_gap"] for x in by[s]), len(by[s])) for s in pop}
    lim = {s: (sum(labels[x["id"]]["type"] == "limitation" for x in by[s]), len(by[s])) for s in pop}
    arxiv, arxiv_se = weighted({s: any_gap[s] for s in TRACKS["arxiv"]}, pop)
    arxiv_lb = arxiv - Z_ONE_SIDED * arxiv_se
    sal_lim, sal_lim_se = weighted({s: lim[s] for s in TRACKS["sal"]}, pop)
    sal_any, sal_any_se = weighted({s: any_gap[s] for s in TRACKS["sal"]}, pop)
    s2_k, s2_n = lim["S2"]
    s2_p = binomial_p(s2_k, s2_n)
    b1_papers = Counter(x["paper"] for x in by["B1"])

    claim1 = gate and arxiv_lb > PRECISION_BAR
    claim2 = gate and s2_p < 0.05
    desc = {"B1": "arXiv, kept, gold none", "B2": "arXiv, kept, gold gap", "S1": "SAL-40, kept, SAL yes",
            "S2": "SAL-40, kept, SAL no, LLM gap", "S3": "SAL-40, kept, both none"}
    lines = [
        "# Human study results", "",
        f"Sample hash `{sample_hash}`. {len(key)} items, {len(dispute)} sent to the tie-breaker.", "",
        f"**Validity gate: {'PASSED' if gate else 'FAILED, so no claims can be made'}.** {decoy_yes} of {len(by['D'])} decoys were labelled a gap (limit {MAX_DECOY_YES}).", "",
        "## Primary tests", "",
        "| Test | Result | Outcome |", "|---|---|---|",
        f"| arXiv true precision, lower bound above {PRECISION_BAR:.0%} | {arxiv:.1%}, one-sided 95% lower bound {arxiv_lb:.1%} | {'Supported' if claim1 else 'Not supported'} |",
        f"| SAL no, LLM gap: more than 50% are real limitations | {s2_k} of {s2_n} ({s2_k / s2_n:.1%}), exact p = {s2_p:.4f} | {'Supported' if claim2 else 'Not supported'} |", "",
        "## True precision", "",
        "| Track | Reading | Estimate | Standard error |", "|---|---|---|---|",
        f"| arXiv | Any gap type | {arxiv:.1%} | {arxiv_se:.1%} |",
        f"| SAL-40 | Limitation only (primary) | {sal_lim:.1%} | {sal_lim_se:.1%} |",
        f"| SAL-40 | Any gap type | {sal_any:.1%} | {sal_any_se:.1%} |", "",
        "## Each group", "",
        "| Group | Population | Rated | Any gap | Limitation |", "|---|---|---|---|---|",
    ]
    for s in pop:
        lines.append(f"| {desc[s]} | {pop[s]} | {any_gap[s][1]} | {any_gap[s][0]} | {lim[s][0]} |")
    lines += ["", "## Rater agreement", "",
              f"Gap or not: {sum(a == b for a, b in pairs_gap) / len(key):.1%} agreement, Krippendorff's alpha {alpha(pairs_gap)}.",
              f"Including type: {sum(a == b for a, b in pairs_type) / len(key):.1%} agreement, Krippendorff's alpha {alpha(pairs_type)}.", "",
              "## Limits", "",
              f"The {len(by['B1'])} B1 items come from {len(b1_papers)} papers, and the top 5 hold {sum(c for _, c in b1_papers.most_common(5))}. Sampling ignored clustering within papers.",
              "Recall is not measured here. Each item was judged from one sentence with its two neighbours."]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def tiebreak(argv):
    key, sample_hash = load_key()
    a1, a2, dispute = two_raters(argv[0], argv[1], key, sample_hash)
    items = [x for x in key if x["id"] in set(dispute)]
    if not items:
        print("No disputed items, so no tie-breaker page is needed.")
        return
    write_page(OUT / "rater_TB.html", items, "TB", sample_hash, SEED)
    print(f"{len(items)} disputed items -> {OUT / 'rater_TB.html'}")


def simulate(argv):
    """Fake answers with known truth, to check the arithmetic. Nothing here is a real rating."""
    target = Path(argv[0])
    target.mkdir(parents=True, exist_ok=True)
    key, sample_hash = load_key()
    truth = {"B1": 0.5, "B2": 0.9, "S1": 0.95, "S2": 0.7, "S3": 0.25, "D": 0.05}
    rng = random.Random(1)
    base = {x["id"]: rng.random() < truth[x["stratum"]] for x in key}
    typed = {x["id"]: "limitation" if x["track"] == "sal" else rng.choice(sorted(TYPES)) for x in key}

    def fake(rater, noise):
        ans = {}
        for x in key:
            yes = base[x["id"]] != (rng.random() < noise)
            ans[x["id"]] = {"is_gap": yes, **({"type": typed[x["id"]]} if yes else {})}
        return {"sample_hash": sample_hash, "rater": rater, "answers": ans}
    r1, r2 = fake("R1", 0.08), fake("R2", 0.08)
    (target / "R1.json").write_text(json.dumps(r1), encoding="utf-8")
    (target / "R2.json").write_text(json.dumps(r2), encoding="utf-8")
    ids = [x["id"] for x in key if disputed(r1["answers"][x["id"]], r2["answers"][x["id"]])]
    tb = {i: {"is_gap": base[i], **({"type": typed[i]} if base[i] else {})} for i in ids}
    (target / "TB.json").write_text(json.dumps({"sample_hash": sample_hash, "rater": "TB", "answers": tb}), encoding="utf-8")
    pop = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))["populations"]
    print("Simulated", len(key), "items,", len(ids), "disputed. Expected arXiv precision about",
          round((pop["B1"] * truth["B1"] + pop["B2"] * truth["B2"]) / (pop["B1"] + pop["B2"]), 3))


if __name__ == "__main__":
    {"tiebreak": tiebreak, "analyze": analyze, "simulate": simulate}[sys.argv[1]](sys.argv[2:])
