"""Run field-specific rater sessions using the existing frozen sample and scorer."""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import human_study_analyze as scoring
from human_study_sample import OUT, SEED, write_page


FIELDS = {"MED": ("sal", 69, "Clinical medicine"), "CS": ("arxiv", 66, "Computer science / machine learning")}
PACK = OUT / "by_field"


def partition(key):
    parts = {code: [item for item in key if item["track"] == track] for code, (track, _, _) in FIELDS.items()}
    ids = {item["id"] for item in key}
    if len(ids) != len(key) or sum(map(len, parts.values())) != len(key):
        raise SystemExit("The frozen sample contains duplicate items or an unexpected track.")
    for code, (_, expected, _) in FIELDS.items():
        if len(parts[code]) != expected:
            raise SystemExit(f"{code}: expected {expected} frozen items, found {len(parts[code])}.")
    return parts


def save_page(directory, items, rater, sample_hash):
    path = directory / f"rater_{rater}.html"
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        candidate = Path(tmp) / path.name
        write_page(candidate, items, rater, sample_hash, SEED)
        # Match the byte format of the released pages on every platform.
        contents = candidate.read_text(encoding="utf-8").replace("\n", "\r\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != contents:
            raise SystemExit(f"{path} already exists with different contents. Use a fresh output directory.")
    else:
        with path.open("xb") as stream:
            stream.write(contents)
    print(f"{rater}: {len(items)} items -> {path}")


def read_rater(directory, rater, items, sample_hash):
    path = directory / f"answers_{rater}.json"
    if not path.is_file():
        raise SystemExit(f"Missing answer file: {path}")
    actual, answers = scoring.read_answers(path, {item["id"] for item in items}, sample_hash)
    if actual != rater:
        raise SystemExit(f"{path}: expected rater {rater}, got {actual}")
    return answers


def collect_pairs(directory, parts, sample_hash):
    return {
        code: tuple(read_rater(directory, f"{code}_{slot}", items, sample_hash) for slot in ("R1", "R2"))
        for code, items in parts.items()
    }


def write_analysis(parts, pairs, adjudicated, sample_hash, out):
    if out.exists():
        raise SystemExit(f"Result already exists: {out}. Choose a new output path.")
    # These slots assemble two votes per item for the original scorer. Each
    # slot contains one person's votes per field, not one person across fields.
    first = {item: vote for a, _ in pairs.values() for item, vote in a.items()}
    second = {item: vote for _, b in pairs.values() for item, vote in b.items()}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = []
        for slot, answers in (("R1", first), ("R2", second), ("TB", adjudicated)):
            if not answers:
                continue
            path = root / f"{slot}.json"
            path.write_text(json.dumps({"sample_hash": sample_hash, "rater": slot, "answers": answers}), encoding="utf-8")
            files.append(str(path))
        report = root / "analysis.md"
        with contextlib.redirect_stdout(io.StringIO()):
            scoring.analyze([*files, "--out", str(report)])
        body = report.read_text(encoding="utf-8")

    agreement = [
        "## Agreement by field before adjudication",
        "",
        "Each field used its own pair of raters. The pooled agreement below describes all item pairs together.",
        "",
        "| Field | Items | Disputed | Gap agreement | Gap alpha | Type agreement | Type alpha |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for code, items in parts.items():
        a, b = pairs[code]
        ids = [item["id"] for item in items]
        gap_pairs = [(a[i]["is_gap"], b[i]["is_gap"]) for i in ids]
        type_pairs = [tuple(v["type"] if v["is_gap"] else "none" for v in (a[i], b[i])) for i in ids]
        gaps = sum(x == y for x, y in gap_pairs) / len(ids)
        types = sum(x == y for x, y in type_pairs) / len(ids)
        disputes = sum(scoring.disputed(a[i], b[i]) for i in ids)
        gap_alpha, type_alpha = (scoring.alpha(values) for values in (gap_pairs, type_pairs))
        gap_text = "n/a" if gap_alpha is None else str(gap_alpha)
        type_text = "n/a" if type_alpha is None else str(type_alpha)
        agreement.append(f"| {FIELDS[code][2]} | {len(ids)} | {disputes} | {gaps:.1%} | {gap_text} | {types:.1%} | {type_text} |")
    title, details = body.split("\n", 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as stream:
        stream.write(title + "\n\n" + "\n".join(agreement) + "\n" + details)
    print(f"Analysis saved to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pages = commands.add_parser("pages", help="Create the four independent first-pass rater pages")
    pages.add_argument("--out", type=Path, default=PACK / "first_pass")
    tie = commands.add_parser("tiebreak", help="Create one page per field with disputed items")
    tie.add_argument("answers", type=Path)
    tie.add_argument("--out", type=Path, default=PACK / "tie_breakers")
    analysis = commands.add_parser("analyze", help="Score complete answers with the frozen protocol")
    analysis.add_argument("answers", type=Path)
    analysis.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    key, sample_hash = scoring.load_key()
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest["sample_hash"] != sample_hash or manifest["items"] != len(key):
        raise SystemExit("The frozen key does not match its manifest.")
    parts = partition(key)
    if args.command == "pages":
        for code, items in parts.items():
            for slot in ("R1", "R2"):
                save_page(args.out, items, f"{code}_{slot}", sample_hash)
        return

    pairs = collect_pairs(args.answers, parts, sample_hash)
    adjudicated = {}
    for code, items in parts.items():
        a, b = pairs[code]
        disputes = [item for item in items if scoring.disputed(a[item["id"]], b[item["id"]])]
        rater = f"{code}_TB"
        if not disputes:
            if args.command == "tiebreak":
                print(f"{code}: no disputes. No tie-breaker page needed.")
            elif (args.answers / f"answers_{rater}.json").exists():
                raise SystemExit(f"{code}: unexpected tie-breaker answers for a field without disputes.")
        elif args.command == "tiebreak":
            save_page(args.out, disputes, rater, sample_hash)
        else:
            adjudicated.update(read_rater(args.answers, rater, disputes, sample_hash))

    if args.command == "analyze":
        write_analysis(parts, pairs, adjudicated, sample_hash, args.out)


if __name__ == "__main__":
    main()
