"""Dataset adapters. Identity/count checks precede all model fitting."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


EXPECTED = {"sal": (40, 7968), "fws": (209, 1506), "arxiv": (28, 10388)}
LABELS = {"none", "limitation", "future_work", "research_gap"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def row(dataset, pid, idx, sentence, label, **metadata):
    if label not in LABELS:
        raise ValueError(f"Unknown gold type: {label}")
    if not isinstance(sentence, str) or not sentence.strip():
        raise ValueError(f"Empty sentence {pid}/{idx}")
    return dict(dataset=dataset, paper_id=str(pid), idx=int(idx), sentence=sentence,
                text_sha256=text_hash(sentence), label=label, y=int(label != "none"), **metadata)


def parse_sal_label(value):
    flags = ast.literal_eval(value)
    if not isinstance(flags, list) or any(type(x) is not bool for x in flags):
        raise ValueError(f"Unexpected SAL annotation: {value!r}")
    return "limitation" if any(flags) else "none"


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def sal_rows(path):
    indices = Counter()
    out = []
    for raw_idx, r in enumerate(csv_rows(path)):
        pid = r["pmid"]
        out.append(row("sal", pid, indices[pid], r["sentences"],
                       parse_sal_label(r["limitation_annotation"]),
                       section=r["section_name"], raw_idx=raw_idx,
                       source_sentence_id=r["sentence_id"]))
        indices[pid] += 1
    return out


def load_sal(root):
    base = root / "data/external/sal_rct"
    fixture_path = root / "stage_c_package/sal_criteria/sal_test_allmodels.json"
    paths = [base / f"{part}.csv" for part in ("train", "dev", "test")] + [fixture_path]
    train, val, raw_test = [sal_rows(p) for p in paths[:3]]
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    # Exact subsequence alignment, including duplicate occurrences. Cached fixture
    # keeps the evaluation sentence indices used by existing funnel predictions.
    by_pid = defaultdict(list)
    for r in raw_test:
        by_pid[r["paper_id"]].append(r)
    cursor, selected, out = Counter(), set(), []
    fixture_indices = Counter()
    for f in fixture:
        pid = str(f["pmcid"])
        candidates = by_pid[pid]
        found = next((j for j in range(cursor[pid], len(candidates))
                      if candidates[j]["sentence"].strip() == f["sent"].strip()), None)
        if found is None:
            raise ValueError(f"SAL fixture no longer aligns exactly: {pid}/{fixture_indices[pid]}")
        source = candidates[found]
        if source["y"] != int(f["label"]):
            raise ValueError(f"SAL fixture/raw label disagreement: {pid}/{found}")
        selected.add(source["raw_idx"])
        out.append(row("sal", pid, fixture_indices[pid], f["sent"], source["label"],
                       section=source["section"], coarse_section=f["sec"],
                       raw_idx=source["raw_idx"], source_sentence_id=source["source_sentence_id"]))
        fixture_indices[pid] += 1
        cursor[pid] = found + 1
    dropped = [r for r in raw_test if r["raw_idx"] not in selected]
    if len(raw_test) != 8882 or len(dropped) != 914 or any(r["y"] for r in dropped):
        raise ValueError("SAL raw-test/fixture discrepancy: expected 8882 raw, 914 negative omissions")
    return {"train": train, "val": val, "test": out}, paths, {
        "raw_test_rows": len(raw_test), "omitted_negative_fragments": len(dropped),
        "section_source": "raw section_name aligned by paper, exact text after outer whitespace strip, and order; model input is unchanged fixture text",
        "rules_limitation": "SAL CSV lacks paragraph boundaries; requires aligned structure file"}


def split_papers(rows, fraction=.1, seed=0):
    papers = sorted({r["paper_id"] for r in rows})
    if len(papers) < 2:
        raise ValueError("Need at least two training papers for validation")
    random.Random(seed).shuffle(papers)
    val_ids = set(papers[:max(1, round(len(papers) * fraction))])
    return ([r for r in rows if r["paper_id"] not in val_ids],
            [r for r in rows if r["paper_id"] in val_ids])


def load_fws(root, seed=0, protocol="current-gold"):
    corpus = root / "data/external/fws_corpus.csv"
    if protocol == "legacy":
        exclusions = root / "data/scibert_prep/train.jsonl"
        paths = [corpus, exclusions]
        training_texts = {r["sentence"].strip() for r in read_jsonl(exclusions)}
        transform = lambda text: text
        gold_stats = None
    elif protocol == "current-gold":
        gold_split, gold_paths, gold_stats = load_current_gold(root)
        paths = [corpus, *gold_paths]
        # Include EVERY sentence in the current train papers, not only the
        # positive rows listed by gap_gold_final. Never use val/test to filter.
        training_texts = {r["sentence"].strip().lower() for r in gold_split["train"]}
        transform = str.lower
    else:
        raise ValueError(f"Unknown FWS protocol {protocol}")
    indices, all_rows = Counter(), []
    for raw_idx, r in enumerate(csv_rows(corpus)):
        pid = r["id"]
        if r["label"] not in {"0", "1"}:
            raise ValueError(f"Unexpected FWS label {r['label']!r}")
        all_rows.append(row("fws", pid, indices[pid], transform(r["text"]),
                            "future_work" if r["label"] == "1" else "none",
                            raw_idx=raw_idx, source_text_sha256=text_hash(r["text"]),
                            section=r.get("section", r.get("section_name", ""))))
        indices[pid] += 1
    eligible = [r for r in all_rows if r["sentence"].strip() not in training_texts]
    # Reuse the sampling algorithm, but current-gold is explicitly a NEW
    # exclusion source / casing protocol, not a certificate of historical identity.
    papers = list(dict.fromkeys(r["paper_id"] for r in eligible))
    random.Random(0).shuffle(papers)
    by_pid = defaultdict(list)
    for r in eligible:
        by_pid[r["paper_id"]].append(r)
    test = []
    for pid in papers:
        test.extend(by_pid[pid])
        if len(test) >= 1500:
            break
    test_ids = {r["paper_id"] for r in test}
    training_pool = eligible if protocol == "current-gold" else all_rows
    train, val = split_papers([r for r in training_pool if r["paper_id"] not in test_ids], seed=seed)
    return {"train": train, "val": val, "test": test}, paths, {
        "protocol_id": "fws_current_gold_v1" if protocol == "current-gold" else "fws_legacy_breadth_v1",
        "test_protocol": "stable eligible CSV paper order; Random(0); complete paper groups until >=1500 rows",
        "exclusion_source": "current gold TRAIN papers, all sentences" if protocol == "current-gold" else "legacy scibert_prep/train.jsonl",
        "exclusion_scope": "all FWS splits" if protocol == "current-gold" else "test selection only",
        "text_transform": "lowercase every sentence before exclusion, training and inference" if protocol == "current-gold" else "original CSV case",
        "historical_test_identity": "not established; do not pair with old funnel outputs without identity checks",
        "gold_corpus_stats": gold_stats,
        "unique_exclusion_texts": len(training_texts),
        "excluded_test_sentences": len(all_rows) - len(eligible),
        "excluded_source_rows": [r["raw_idx"] for r in all_rows if r["sentence"].strip() in training_texts],
        "test_membership_sha256": text_hash(json.dumps([
            [r["paper_id"], r["idx"], r["source_text_sha256"]] for r in test], separators=(",", ":"))),
        "validation_protocol": "10% of remaining papers, sorted IDs then seeded shuffle",
        "validation_seed": seed}


def trim_refs(text):
    low, n = text.lower(), len(text)
    for kw in ("\nreferences", "\nbibliography", "references\n", "\nacknowledgment", "\nacknowledgement"):
        p = low.rfind(kw)
        if p > n * .5:
            n = min(n, p)
    return text[:n]


def load_arxiv(root, prefer_exports=True):
    """Use current branch's canonical splitter; verify every positive's exact text.

    The optional export path avoids dependence on future segmentation changes.
    Exports must contain every sentence, including negatives, with original IDs.
    """
    base = root / "data/gold"
    export = base / "task5b_arxiv"
    if prefer_exports and (export / "train.jsonl").exists():
        paths = [export / f"{p}.jsonl" for p in ("train", "val", "test")]
        split = {}
        for part, path in zip(("train", "val", "test"), paths):
            split[part] = [row("arxiv", r["paper_id"], r["idx"], r["sentence"], r["label"],
                               section=r.get("section", ""))
                           for r in read_jsonl(path) if r["domain"] == "arxiv_cs"]
        return split, paths, {"source": "complete gold_finalize/export_stageb_data exports, arxiv_cs only"}
    from gap2idea.pipeline.gap_prefilter import split_sentences
    paths = [base / "gap_gold_final.jsonl", base / "splits.json", root / "data/paper_texts.jsonl"]
    gold = read_jsonl(paths[0])
    splits = json.loads(paths[1].read_text(encoding="utf-8"))
    domain = {}
    labels = {}
    for r in gold:
        pid, idx = str(r["paper_id"]), int(r["idx"])
        if pid in domain and domain[pid] != r["domain"]:
            raise ValueError(f"Inconsistent domain for {pid}")
        domain[pid] = r["domain"]
        if (pid, idx) in labels:
            raise ValueError(f"Duplicate gold sentence {pid}/{idx}; provide finalized labels")
        labels[pid, idx] = r
    texts = {str(r["id"]): r["text"] for r in read_jsonl(paths[2])}
    out, used = {}, set()
    for part in ("train", "val", "test"):
        out[part] = []
        for raw_pid in splits[part]:
            pid = str(raw_pid)
            if pid not in domain:
                raise ValueError(f"No domain annotation for split paper {pid}; provide a complete export")
            if domain[pid] != "arxiv_cs":
                continue
            if pid not in texts:
                raise ValueError(f"Missing full text for {pid}")
            sents = split_sentences(trim_refs(texts[pid]))
            for idx, sentence in enumerate(sents):
                g = labels.get((pid, idx))
                if g and g["sentence"] != sentence:
                    raise ValueError(f"Gold/segmentation drift: {pid}/{idx}; use exact original exports")
                if g:
                    used.add((pid, idx))
                # Positive-only gold section fields must NOT become input features:
                # doing so would reveal which sentences were annotated positive.
                out[part].append(row("arxiv", pid, idx, sentence, g["label"] if g else "none", section=""))
    expected_gold = {key for key in labels if domain[key[0]] == "arxiv_cs"
                     and any(key[0] in map(str, splits[p]) for p in ("train", "val", "test"))}
    if used != expected_gold:
        raise ValueError("Some arXiv gold indices lie outside reconstructed sentences")
    return out, paths, {"source": "final positive gold + exact full texts + official paper splits"}


def load_current_gold(root):
    """Reconstruct the current 230-paper corpus without API clients or old data.

    The uploaded gold file lists positives only. Reconstruct ALL arXiv and SAL
    sentences, verify each positive against its original paper/index/text, then
    apply the explicit paper split. Public SAL fixture labels are not substituted
    for the current four-class labels.
    """
    arxiv, paths, _ = load_arxiv(root, prefer_exports=False)
    splits_path = root / "data/gold/splits.json"
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    gold = read_jsonl(root / "data/gold/gap_gold_final.jsonl")
    labels = {(str(g["paper_id"]), int(g["idx"])): g for g in gold}
    fixture_path = root / "stage_c_package/sal_criteria/sal_test_allmodels.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    paths = list(dict.fromkeys([*paths, fixture_path]))
    sal, indices, verified = defaultdict(list), Counter(), set()
    for f in fixture:
        pid = str(f["pmcid"])
        idx = indices[pid]
        indices[pid] += 1
        g = labels.get((pid, idx))
        if g and (g["sentence"] != f["sent"] or g["domain"] != "biomed_sal"):
            raise ValueError(f"Current SAL gold/fixture alignment drift: {pid}/{idx}")
        if g:
            verified.add((pid, idx))
        sal[pid].append(row("current_gold", pid, idx, f["sent"], g["label"] if g else "none",
                            domain="biomed_sal", section=f["sec"]))
    expected_sal = {k for k, g in labels.items() if g["domain"] == "biomed_sal"}
    if verified != expected_sal:
        raise ValueError("Some current SAL gold indices lie outside the cached fixture")
    corp = {}
    for rows in arxiv.values():
        for r in rows:
            corp.setdefault(r["paper_id"], []).append({**r, "dataset": "current_gold", "domain": "arxiv_cs"})
    if set(corp) & set(sal):
        raise ValueError("Paper IDs overlap between gold domains")
    corp.update(sal)
    parts, seen = {}, set()
    for part in ("train", "val", "test"):
        ids = list(map(str, splits[part]))
        if len(ids) != len(set(ids)) or seen & set(ids):
            raise ValueError("Current gold paper splits overlap or contain duplicates")
        if set(ids) - set(corp):
            raise ValueError(f"Current gold lacks full texts for {set(ids) - set(corp)}")
        seen.update(ids)
        parts[part] = [r for pid in ids for r in corp[pid]]
    if seen != set(corp):
        raise ValueError("Some current corpus papers have no split assignment")
    if len(labels) != len(gold) or set(k[0] for k in labels) - seen:
        raise ValueError("Current gold labels contain duplicate/unknown paper indices")
    stats = {p: dict(papers=len(splits[p]), sentences=len(rows),
                     positives=sum(r["y"] for r in rows),
                     domains=dict(Counter(r["domain"] for r in rows))) for p, rows in parts.items()}
    return parts, paths, stats


def apply_structure(split, path):
    """Attach externally recovered structure without changing any sentence ID.

    Require text hashes and exhaustive coverage. Raw LaTeX blocks alone are not
    sentence-aligned paragraph annotations; never guess their correspondence.
    """
    records = read_jsonl(path)
    mapping = {}
    for r in records:
        key = (str(r["paper_id"]), int(r["idx"]))
        if key in mapping:
            raise ValueError(f"Duplicate structure key: {key}")
        mapping[key] = r
    for rows in split.values():
        for r in rows:
            key = r["paper_id"], r["idx"]
            s = mapping.get(key)
            if s is None or s.get("text_sha256") != r["text_sha256"]:
                raise ValueError(f"Missing/mismatched sentence-aligned structure: {key}")
            for field in ("section", "top_section", "paragraph_id", "section_id"):
                if field not in s:
                    raise ValueError(f"Structure missing {field}: {key}")
                r[field] = s[field]
            r["structure_source"] = str(path)
            r["structure_quality"] = s.get("structure_quality", "sentence_aligned")


def attach_latex_sections(split, blocks_path, texts_path):
    """Locate section bodies in the ORIGINAL flat texts, preserving gold indices.

    The branch's LaTeX adapter provides top-level sections, not paragraphs. This
    enables only the declared section-proxy rule extension. Unlocated areas stay
    unknown and fail the lexical section gate; all test sentences stay in scoring.
    """
    normalize = lambda s: re.sub(r"\s+", " ", s).strip()
    sources = {str(r["id"]): r for r in read_jsonl(blocks_path)}
    texts = {str(r["id"]): normalize(r["text"]) for r in read_jsonl(texts_path)}
    by_pid = defaultdict(list)
    for rows in split.values():
        for r in rows:
            by_pid[r["paper_id"]].append(r)
    matched, total = 0, 0
    for pid, rows in by_pid.items():
        if pid not in texts or pid not in sources:
            raise ValueError(f"Missing original text/LaTeX status for {pid}")
        text, regions, cursor, heading = texts[pid], [], 0, ""
        for block in sources[pid]["blocks"]:
            if block["role"] == "heading":
                heading = block["text"]
            elif block["role"] == "text":
                body = normalize(block["text"])
                start = text.find(body, cursor)
                if start < 0:
                    raise ValueError(f"LaTeX block no longer exactly locates in flat text: {pid}")
                regions.append((start, start + len(body), heading))
                cursor = start + len(body)
        cursor = 0
        for r in sorted(rows, key=lambda r: r["idx"]):
            sentence = normalize(r["sentence"])
            start = text.find(sentence, cursor)
            if start < 0:
                raise ValueError(f"Canonical sentence no longer locates in original flat text: {pid}/{r['idx']}")
            end = start + len(sentence)
            overlaps = [max(0, min(end, b) - max(start, a)) for a, b, _ in regions]
            j = max(range(len(regions)), key=overlaps.__getitem__) if regions else None
            title = regions[j][2] if j is not None and overlaps[j] > 0 else ""
            r.update(section=title, top_section=title, structure_source="arxiv_latex_blocks",
                     structure_quality="section_proxy", latex_section_id=j if title else -1)
            matched += bool(title)
            total += 1
            cursor = end
    return dict(section_mapped_sentences=matched, total_sentences=total,
                note="Top-level LaTeX section-body overlap, no paragraph recovery; unmapped sentences retained")


def validate(split, dataset, strict=True):
    stats, paper_sets = {}, {}
    text_sets = {}
    for part in ("train", "val", "test"):
        rows = split[part]
        if not rows:
            raise ValueError(f"{dataset}/{part} is empty")
        keys = [(r["paper_id"], r["idx"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError(f"Duplicate paper/sentence keys in {dataset}/{part}")
        paper_sets[part] = {r["paper_id"] for r in rows}
        text_sets[part] = {r["sentence"].strip() for r in rows}
        stats[part] = dict(papers=len(paper_sets[part]), sentences=len(rows),
                           positives=sum(r["y"] for r in rows),
                           labels=dict(Counter(r["label"] for r in rows)))
        if part != "test" and len({r["y"] for r in rows}) != 2:
            raise ValueError(f"{dataset}/{part} must contain both classes")
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        if paper_sets[a] & paper_sets[b]:
            raise ValueError(f"Paper leakage {dataset}: {a}/{b}")
    if strict:
        observed = stats["test"]["papers"], stats["test"]["sentences"]
        if observed != EXPECTED[dataset]:
            raise ValueError(f"{dataset}: test {observed} != proposal {EXPECTED[dataset]}; do not silently substitute a split")
    stats["exact_text_overlap"] = {f"{a}/{b}": len(text_sets[a] & text_sets[b]) for a, b in pairs}
    return stats


def load_prepared(directory, dataset):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset:
        raise ValueError("Prepared dataset name mismatch")
    split = {}
    for part in ("train", "val", "test"):
        p = directory / f"{part}.jsonl"
        if digest(p) != manifest["prepared_sha256"][part]:
            raise ValueError(f"Prepared data changed since locking: {p}")
        split[part] = read_jsonl(p)
    validate(split, dataset, strict=manifest["mode"] == "full")
    return split, manifest
