# -*- coding: utf-8 -*-
"""Recover clean section structure for the arXiv gold papers from their LaTeX source.

Why: arXiv papers were ingested as *flat* PDF-extracted text (``data/paper_texts.jsonl``
has no section labels), so Stage A's whole-section keep (``_section_regions``) never
fires on them and the Stage-A ceiling caps ~56% vs biomed's 84%. 93% of the arXiv gold
papers publish LaTeX source on arXiv, which carries authored ``\\section{...}`` markup.

This script, for each arXiv gold paper:
  1. downloads the e-print source tarball from arXiv (cached in the scratchpad);
  2. parses the ordered top-level ``\\section`` / ``\\section*`` titles (+ ``\\appendix``);
  3. locates each title inside the paper's OWN flat text (reading order), so section
     boundaries are set WITHOUT changing the sentence text the gold was built from;
  4. emits biomed-style blocks ``[{role:heading,text:<title>}, {role:text,text:<body>} ...]``
     that ``slice_section_aware`` consumes exactly like SAL's per-sentence sections.

Keeping the paper's own flat-text sentences (not LaTeX-rendered ones) means the Stage-A
ceiling measurement isolates the effect of *sectioning* alone, with no text-source
mismatch against the gold.

Output: ``data/gold/arxiv_latex_blocks.jsonl`` ({id, src_format, n_sections, n_located, blocks})
plus a printed summary. PDF-only / unparseable papers are written with empty blocks
(src_format="pdf_only"/"no_source") so the benchmark falls back to flat text for them.

    python scripts/dataset/build_arxiv_latex_blocks.py            # download (cached) + build
    python scripts/dataset/build_arxiv_latex_blocks.py --no-net   # build from cache only
"""
from __future__ import annotations
import sys, os, re, io, json, gzip, time, tarfile, argparse, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gap2idea.pipeline.sections import _cut_before_references
from gap2idea.pipeline.gap_prefilter import split_sentences

TEXTS = ROOT / "data" / "paper_texts.jsonl"
OUT = ROOT / "data" / "gold" / "arxiv_latex_blocks.jsonl"
CACHE = Path(os.environ.get("ARXIV_SRC_CACHE",
             r"C:\Users\yazan\AppData\Local\Temp\claude"
             r"\C--Users-yazan-OneDrive-Desktop-thesis-Gap2Idea"
             r"\10eb8776-3b7e-46e0-bfa8-b53a787900b0\scratchpad\arxiv_src"))
UA = "gap2idea-research/1.0 (mailto:hamza132.1.sy@gmail.com)"
DELAY = 0.5  # polite gap between arXiv requests


# --------------------------------------------------------------------------- download
def fetch_source(arxiv_id: str, use_net: bool) -> bytes | None:
    """Return raw e-print bytes for ``arxiv_id`` (cached). None if unavailable."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{arxiv_id}.src"
    if cf.exists() and cf.stat().st_size > 0:
        return cf.read_bytes()
    if not use_net:
        return None
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except Exception as e:  # noqa: BLE001
        print(f"  [{arxiv_id}] download failed: {str(e)[:70]}", flush=True)
        return None
    cf.write_bytes(data)
    time.sleep(DELAY)
    return data


# --------------------------------------------------------------------------- unpack
def _read_tex_members(data: bytes) -> dict[str, str] | None:
    """Return {member_name: text} for all .tex files; None if ``data`` is a PDF."""
    if data[:4] == b"%PDF":
        return None
    # try tar (possibly gzipped)
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            out: dict[str, str] = {}
            for m in tf.getmembers():
                if not m.isfile() or not m.name.lower().endswith((".tex", ".ltx")):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                raw = f.read()
                out[m.name] = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if out:
                return out
    except (tarfile.TarError, EOFError, OSError):
        pass
    # single gzipped file (one .tex, not tarred)
    try:
        raw = gzip.decompress(data)
        if raw[:4] != b"%PDF":
            return {"main.tex": raw.decode("utf-8", "replace")}
    except OSError:
        pass
    # plain text (already a .tex)
    if b"\\documentclass" in data[:4000] or b"\\begin{document}" in data:
        return {"main.tex": data.decode("utf-8", "replace")}
    return None


# --------------------------------------------------------------------------- latex parse
_COMMENT = re.compile(r"(?<!\\)%.*")
_INPUT = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
_SECTION = re.compile(r"\\section\*?\s*\{")
_APPENDIX = re.compile(r"\\appendix\b")


def _strip_comments(tex: str) -> str:
    return _COMMENT.sub("", tex)


def _balanced(s: str, open_pos: int) -> tuple[str, int]:
    """Given ``s[open_pos] == '{'``, return (inner_text, index_after_closing_brace)."""
    depth, i, n = 0, open_pos, len(s)
    while i < n:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[open_pos + 1:i], i + 1
        i += 1
    return s[open_pos + 1:], n


def _resolve_main(members: dict[str, str]) -> str:
    """Concatenate the main .tex (the one with \\begin{document}) with its \\input files."""
    mains = [n for n, t in members.items() if "\\begin{document}" in t]
    if not mains:
        mains = sorted(members, key=lambda n: members[n].count("\\section"), reverse=True)[:1]
    if not mains:
        return ""
    main = max(mains, key=lambda n: members[n].count("\\section"))

    def _base(n: str) -> str:
        return Path(n).name.lower().rsplit(".tex", 1)[0].rsplit(".ltx", 1)[0]

    by_base = {}
    for n, t in members.items():
        by_base.setdefault(_base(n), t)

    seen: set[str] = set()

    def _expand(name: str, depth: int) -> str:
        if depth > 6 or name in seen:
            return ""
        seen.add(name)
        tex = members.get(name) or by_base.get(_base(name), "")
        tex = _strip_comments(tex)

        def _sub(m: re.Match) -> str:
            ref = _base(m.group(1))
            if ref in by_base:
                return "\n" + _expand(ref if ref in members else _find_name(ref), depth + 1) + "\n"
            return ""

        def _find_name(base: str) -> str:
            for n in members:
                if _base(n) == base:
                    return n
            return base

        return _INPUT.sub(_sub, tex)

    return _expand(main, 0)


_CLEAN_CMD = re.compile(r"\\[a-zA-Z@]+\s*(\*)?")       # drop commands
_CLEAN_BRACE = re.compile(r"[{}]")
_CLEAN_MATH = re.compile(r"\$[^$]*\$")
_CLEAN_LABEL = re.compile(r"\\label\s*\{[^}]*\}")


def _merge_dropcaps(title: str) -> str:
    """Rejoin a leading single letter split off by \\lettrine or {I}ntro grouping:
    'I ntroduction' -> 'Introduction', 'R andom H yperbolic' -> 'Random Hyperbolic'."""
    words = title.split()
    out: list[str] = []
    i = 0
    while i < len(words):
        if len(words[i]) == 1 and words[i].isalpha() and i + 1 < len(words):
            out.append(words[i] + words[i + 1])
            i += 2
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def _clean_title(raw: str) -> str:
    t = _CLEAN_LABEL.sub("", raw)
    t = _CLEAN_MATH.sub(" ", t)
    # drop \footnote{...} / \thanks{...} contents entirely
    t = re.sub(r"\\(?:footnote|thanks)\s*\{[^}]*\}", " ", t)
    t = _CLEAN_CMD.sub(" ", t)
    t = _CLEAN_BRACE.sub(" ", t)
    t = t.replace("~", " ").replace("\\&", "&")
    t = re.sub(r"\s+", " ", t).strip(" .:-")
    return _merge_dropcaps(t)


# LaTeX dimension / spacing spec leaking in from \@startsection / \titlespacing
# redefinitions (e.g. "section 1 -2.0ex plus -0.5ex minus -.2ex 1.5ex ...").
_DIMEN_JUNK = re.compile(r"\d\s*(?:ex|pt|em|sp|bp|cm|mm|in)\b|\bplus\b.*\bminus\b", re.IGNORECASE)


def _is_junk_title(t: str) -> bool:
    return bool(_DIMEN_JUNK.search(t)) or t.lower().startswith("section ")


def parse_sections(tex: str) -> list[str]:
    """Ordered top-level section titles (incl. a synthetic 'Appendix' at \\appendix)."""
    tex = _strip_comments(tex)
    marks: list[tuple[int, str]] = []
    for m in _SECTION.finditer(tex):
        title, _ = _balanced(tex, m.end() - 1)
        ct = _clean_title(title)
        if ct and not _is_junk_title(ct):
            marks.append((m.start(), ct))
    for m in _APPENDIX.finditer(tex):
        marks.append((m.start(), "Appendix"))
    marks.sort(key=lambda x: x[0])
    # de-dup consecutive identical titles
    out: list[str] = []
    for _, t in marks:
        if not out or out[-1].lower() != t.lower():
            out.append(t)
    return out


# --------------------------------------------------------------------------- segment flat text
_NUM_PREFIX = r"(?:\d+(?:\.\d+)*\.?\s+)?"


def _title_pattern(title: str) -> re.Pattern | None:
    words = re.findall(r"[A-Za-z]+", title)
    if not words:
        return None
    body = r"\s+".join(re.escape(w) for w in words)
    return re.compile(_NUM_PREFIX + body, re.IGNORECASE)


def segment(flat: str, titles: list[str]) -> tuple[list[dict], int]:
    """Locate ``titles`` in order inside ``flat`` and return (blocks, n_located).

    blocks alternate heading/text; body spans run from one located heading to the
    next. Sentences keep their original flat-text form (gold-aligned).
    """
    cut = _cut_before_references(flat)
    low = cut.lower()
    bounds: list[tuple[int, int, str]] = []  # (start, end_of_heading, title)
    cursor = 0
    for t in titles:
        pat = _title_pattern(t)
        if pat is None:
            continue
        m = pat.search(low, cursor)
        if not m:
            continue  # heading not found in flat text (scramble / PDF-only); skip
        bounds.append((m.start(), m.end(), t))
        cursor = m.end()
    if not bounds:
        return [], 0
    blocks: list[dict] = []
    for i, (hs, he, title) in enumerate(bounds):
        nxt = bounds[i + 1][0] if i + 1 < len(bounds) else len(cut)
        body = cut[he:nxt].strip()
        blocks.append({"role": "heading", "text": title})
        if body:
            blocks.append({"role": "text", "text": body})
    return blocks, len(bounds)


# --------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-net", action="store_true", help="use cached sources only")
    ap.add_argument("--limit", type=int, default=0, help="process only first N (debug)")
    args = ap.parse_args()

    ids = [json.loads(l)["id"] for l in TEXTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    flat = {json.loads(l)["id"]: json.loads(l)["text"]
            for l in TEXTS.read_text(encoding="utf-8").splitlines() if l.strip()}
    if args.limit:
        ids = ids[:args.limit]

    rows, stats = [], {"latex_ok": 0, "pdf_only": 0, "no_source": 0, "no_sections": 0}
    loc_rates = []
    for k, pid in enumerate(ids):
        data = fetch_source(pid, use_net=not args.no_net)
        if data is None:
            rows.append({"id": pid, "src_format": "no_source", "n_sections": 0,
                         "n_located": 0, "blocks": []})
            stats["no_source"] += 1
        else:
            members = _read_tex_members(data)
            if members is None:
                rows.append({"id": pid, "src_format": "pdf_only", "n_sections": 0,
                             "n_located": 0, "blocks": []})
                stats["pdf_only"] += 1
            else:
                tex = _resolve_main(members)
                titles = parse_sections(tex)
                blocks, n_loc = segment(flat[pid], titles)
                if n_loc == 0:
                    rows.append({"id": pid, "src_format": "no_sections", "n_sections": len(titles),
                                 "n_located": 0, "blocks": []})
                    stats["no_sections"] += 1
                else:
                    rows.append({"id": pid, "src_format": "latex", "n_sections": len(titles),
                                 "n_located": n_loc, "blocks": blocks})
                    stats["latex_ok"] += 1
                    loc_rates.append(n_loc / max(1, len(titles)))
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(ids)} processed", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(ids)
    avg_loc = sum(loc_rates) / len(loc_rates) if loc_rates else 0.0
    print("\n=== build_arxiv_latex_blocks summary ===")
    print(f"  papers:            {n}")
    print(f"  latex sections ok: {stats['latex_ok']} ({100*stats['latex_ok']/n:.1f}%)")
    print(f"  pdf_only:          {stats['pdf_only']}")
    print(f"  no_source/dl_fail: {stats['no_source']}")
    print(f"  parsed-but-0-found:{stats['no_sections']}")
    print(f"  mean title-localization rate (latex_ok papers): {avg_loc*100:.1f}%")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
