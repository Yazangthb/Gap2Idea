"""Cheap, scalable extraction of future-work & limitation gaps.

This is the production funnel that replaces the per-paper LLM extractor
(`openai_gaps.py`) for the two *structurally localized* gap types —
``future_work`` and ``limitation``. It does NOT try to find ``open_problem``
gaps: those are a document/discourse-level task (a rhetorical question is only a
gap if the paper leaves it unresolved) and cannot be decided cheaply from a
sentence. See ``docs/gap_extraction_architecture.md`` §5.2.

Why this is cheap (target: < a few $ / 1M papers vs ~$4000 for per-paper LLM):

    Stage A  STRUCTURAL SLICE          FREE   (regex + position gate, CPU)
        find the terminal Limitations / Future-Work / Conclusion region(s);
        drop ~95% of the paper. Position does the discourse-level work that a
        flat lexical prefilter could not (see §5.1) — a real limitation of own
        work lives near the end, just before refs/acks.

    Stage B  PER-SENTENCE CLASSIFY     ~cents (only slice sentences embedded)
        high-precision cue rules give a free fast-accept + a type; a tiny
        logreg head on MiniLM embeddings catches the cue-less remainder. The
        head is self-distilled from the existing LLM teacher labels in
        ``runs/*`` — zero new annotation.

    Stage C  LLM audit on ~0.1% sample (not in this module; drift only)

The output schema is identical to ``openai_gaps.py`` so everything downstream
(theme mining, gap graph, idea generation) is unchanged:
    id, gap_type, gap_sentence, paragraph_text, confidence
plus two provenance columns this funnel can afford to add: ``section_type`` and
``source`` (rule | model | rule+model).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from gap2idea.pipeline.gap_prefilter import normalize_text, split_sentences
from gap2idea.pipeline.sections import (
    DISCUSSION_HEAD_RE,
    FUTURE_HEAD_RE,
    LIMITATION_HEAD_RE,
    REF_RE,
    _cut_before_references,
)
from gap2idea.utils import get_logger

log = get_logger(__name__)

TARGET_TYPES = ("limitation", "future_work")

# Position below which a generic Discussion/Conclusion anchor is treated as
# mid-paper exposition rather than the terminal wrap-up (the "Oracle
# Limitations" trap, §5.2). Limitations/Future-Work anchors are trusted at any
# position — they are reliable wherever they appear.
TERMINAL_THRESHOLD = 0.45
MIN_SENT_CHARS = 25       # a candidate gap sentence must be this substantive
TAIL_SENTS = 30           # terminal tail: catches unheaded conclusions
                          # (heading detection is unreliable on scrambled PDFs)
HEADING_SPAN = 40         # sentences pulled after a target *heading* anchor
KW_WINDOW = 5             # sentences around an *inline* keyword hit (scrambled headings)
MAX_SLICE_SENTS = 160     # bound the load handed to Stage B (load is cheap)
_WORD = re.compile(r"[a-z0-9]+")
# References / appendices end the gap-bearing body; cut the stream here so the
# terminal tail lands on the real conclusion, not appendix text (long papers).
_STOP_RE = re.compile(
    r"^\s*(references|bibliography|appendix|appendices|supplementary|"
    r"acknowledg(?:e?ments?|ements?))\b",
    re.IGNORECASE,
)
# Hyphen-invariant matching: join across hyphens with or without a line break
# ("limita- tions"->"limitations", "multi-modal"->"multimodal") so a gold
# sentence the LLM de-hyphenated still token-matches the raw scrambled source.
_DEHYPHEN_RE = re.compile(r"(\w)-\s*(\w)")

# Type tag priority when several anchors cover the same sentence.
_TAG_RANK = {"limitations": 3, "future_work": 2, "discussion": 1, "tail": 0}


# ---------------------------------------------------------------------------
# Stage A — structural slice (recall-robust to PDF reading-order scrambling)
# ---------------------------------------------------------------------------
@dataclass
class Region:
    section_type: str          # limitations | future_work | discussion | tail
    heading: str
    location_fraction: float   # 0 = front, 1 = end of pre-ref text
    is_terminal: bool
    sentences: list[str] = field(default_factory=list)

    @property
    def body(self) -> str:
        return " ".join(self.sentences)


def _looks_like_sentence(s: str) -> bool:
    """Reject column-scramble debris ("mance as FSG but ..."), bare formula lines,
    and table-of-contents dot-leaders ("D.1 Limitations . . . . 7") so the
    classifier only ever sees plausible candidate sentences."""
    s = s.strip()
    if len(s) < MIN_SENT_CHARS:
        return False
    toks = s.split()
    if len(toks) < 6:
        return False
    if not (s[0].isupper() or s[0].isdigit() or s[0] in "(\"'“"):
        return False  # starts mid-word/clause -> a broken fragment
    if ". ." in s or s.count(".") > 6:
        return False  # table-of-contents dot leaders
    alpha = sum(1 for c in s if c.isalpha())
    return alpha / len(s) >= 0.55  # reject formula/number/punctuation-dominated lines


# References / citations / captions that column-scramble drags into a gap region.
# These can never be gap sentences, so Stage A drops them from the candidate slice
# BEFORE Stage B ever sees them — a candidate-quality gate, like _looks_like_sentence.
_REFERENCE_RE = re.compile(
    r"(?:19|20)\d\d[a-z]?\s*\.\s*$"                     # ends in a year + period (bib entry / trailing cite)
    r"|\b(?:In\s+)?Proceedings\b|\bConference on\b|\bJournal of\b"
    r"|\bTransactions on\b|\bWorkshop on\b|\bSymposium on\b"
    r"|\barXiv:\s*\d|\bdoi\s*:|\bpreprint\b|\bpp\.\s*\d+"
    r"|(?:Figure|Table|Fig\.|Tab\.)\s*\d+\s*[:.]",      # figure / table captions
    re.IGNORECASE,
)
_BIB_AUTHOR_RE = re.compile(r"[A-Z][A-Za-z]+,\s+[A-Z]\.")   # "Khondaker, A." bibliography author


def _is_reference(s: str) -> bool:
    """True for bibliography entries, citation dumps, and figure/table captions —
    text that PDF column-scramble interleaves into a gap region but that can never
    be a gap sentence. Conservative: a lone inline 'et al.' inside a real sentence
    is NOT matched (no year-end / venue / author-list signal), so genuine
    limitations that cite prior work survive the slice."""
    if _REFERENCE_RE.search(s):
        return True
    return len(_BIB_AUTHOR_RE.findall(s)) >= 2             # dense author list = a bibliography line


def _anchor_tag(s: str, is_heading: bool) -> str | None:
    """limitations / future_work / discussion if this unit anchors a gap region."""
    if LIMITATION_HEAD_RE.search(s):
        return "limitations"
    if FUTURE_HEAD_RE.search(s):
        return "future_work"
    if is_heading and DISCUSSION_HEAD_RE.search(s):
        return "discussion"
    return None


def _ordered_units(text: str, blocks: list[dict] | None) -> list[tuple[str, bool]]:
    """Reading-order (sentence, is_heading) stream, truncated at References.

    Headings are kept verbatim (even when short, e.g. "Limitations") so keyword
    anchoring works; they are flagged so Stage B can exclude them as candidates.

    Consecutive body blocks (PDF line-wraps) are joined into one paragraph
    before sentence splitting — otherwise a sentence wrapping across lines
    becomes multiple truncated fragments. A heading breaks the paragraph.
    """
    units: list[tuple[str, bool]] = []
    if blocks:
        buf: list[str] = []

        def _flush_buf() -> None:
            if not buf:
                return
            para = " ".join(buf)
            para = _DEHYPHEN_RE.sub(r"\1\2", para)  # "word- next" -> "wordnext"
            for s in split_sentences(para):
                units.append((s, False))
            buf.clear()

        for b in blocks:
            role = b.get("role")
            t = str(b.get("text", "")).strip()
            if not t:
                continue
            if role == "heading" and _STOP_RE.search(t):
                _flush_buf()
                break
            if role == "heading":
                _flush_buf()
                units.append((t, True))
            else:
                buf.append(t)
        _flush_buf()
    else:
        for s in split_sentences(_cut_before_references(text)):
            units.append((s, False))
    return units


def slice_terminal_regions(
    text: str,
    blocks: list[dict] | None = None,
    terminal_only: bool = True,
) -> list[Region]:
    """Stage A. Return candidate gap-bearing regions, recall-first.

    Unions three complementary sources so a real gap is rarely dropped:
      * heading-anchored spans — a generous window after a Limitations /
        Future-Work / Conclusion heading;
      * inline keyword windows — ±few sentences around a keyword that appears
        mid-line (two-column-scrambled PDFs hide headings inside body text);
      * terminal tail — the last sentences before References (unheaded
        conclusions / future-work statements).

    ``terminal_only`` only suppresses *mid-paper Discussion* anchors; explicit
    Limitations/Future-Work anchors and the tail are always kept. Set False to
    disable even that (bench ablation).
    """
    units = _ordered_units(text, blocks)
    n = len(units)
    if n == 0:
        return []
    sents = [u[0] for u in units]
    is_head = [u[1] for u in units]

    keep: dict[int, str] = {}

    def _add(i: int, tag: str) -> None:
        if 0 <= i < n:
            if i not in keep or _TAG_RANK[tag] > _TAG_RANK[keep[i]]:
                keep[i] = tag

    # terminal tail
    for i in range(max(0, n - TAIL_SENTS), n):
        _add(i, "tail")

    # keyword / heading anchors
    for i in range(n):
        tag = _anchor_tag(sents[i], is_head[i])
        if tag is None:
            continue
        if tag == "discussion" and terminal_only and (i / n) < TERMINAL_THRESHOLD:
            continue  # mid-paper discussion = exposition, not gaps
        span = HEADING_SPAN if is_head[i] else KW_WINDOW
        _add(i - 1, tag)
        for j in range(i, min(n, i + span + 1)):
            _add(j, tag)

    # cap load by ANCHOR PRIORITY (limitations > future > discussion > tail),
    # not position — a position cap would drop a mid-paper Limitations section
    # in favour of appendix tail in long papers.
    if len(keep) > MAX_SLICE_SENTS:
        ranked = sorted(keep, key=lambda i: (_TAG_RANK[keep[i]], -i), reverse=True)
        keep = {i: keep[i] for i in ranked[:MAX_SLICE_SENTS]}
    idx = sorted(keep)

    # group contiguous kept indices into regions; drop heading-only units from
    # the candidate sentence list (they are anchors, not gaps)
    regions: list[Region] = []
    run: list[int] = []

    def _flush() -> None:
        if not run:
            return
        tags = [keep[k] for k in run]
        best = max(tags, key=lambda t: _TAG_RANK[t])
        head = next((sents[k] for k in run if is_head[k]), best)
        cand = [sents[k] for k in run
                if not is_head[k] and len(sents[k]) >= MIN_SENT_CHARS
                and not _is_reference(sents[k])]
        if cand:
            loc = run[0] / n
            regions.append(Region(best, head[:80], round(loc, 4), loc >= TERMINAL_THRESHOLD, cand))

    prev = None
    for k in idx:
        if prev is not None and k != prev + 1:
            _flush()
            run = []
        run.append(k)
        prev = k
    _flush()
    return regions


# ===========================================================================
# Stage A v2 — slice_with_midpaper_anchors
# ===========================================================================
# v1 (slice_terminal_regions) catches terminal Limitations/Future-Work + tail
# + inline keywords. It misses mid-paper scope/assumption sentences.
#
# v2 adds an additional sweep for mid-paper anchor phrases (scope restrictions,
# explicit assumptions) anywhere in the paper, with a NARROW window (±2 sents)
# to avoid bloating the slice. Tagged "midpaper" so they're distinguishable.
#
# IMPORTANT: v2 is ADDITIVE — it calls v1 unchanged and merges new regions.
# v1 behavior is preserved exactly for all existing benchmarks.
# See docs/paper/stage_versions.md for measured impact.
# ---------------------------------------------------------------------------

_MIDPAPER_ANCHOR_RE = re.compile(
    r"\b("
    r"(?:we|this (?:paper|work|study|section)) (?:focus(?:es)? on|concentrate[d]? on|"
    r"devote (?:our )?attention|restrict (?:the )?attention|focuses on)|"
    r"we (?:assume|assumed|focus|restrict)|"
    r"our (?:method|approach|analysis|work|study) assumes|"
    r"throughout (?:this|the) (?:paper|work|section)|"
    r"in the interest of simplicity|to simplify (?:our|the) (?:treatment|analysis)|"
    r"focus is to understand|"
    r"under (?:the |a )(?:simplifying |separability |separation |strong )?assumption|"
    r"blanket assumption|standing assumption|the following assumption"
    r")\b",
    re.IGNORECASE,
)

MIDPAPER_WINDOW = 2     # ±N sentences around the mid-paper anchor
MIDPAPER_TAG = "midpaper"
_TAG_RANK_V2 = {**_TAG_RANK, MIDPAPER_TAG: 1}  # below future_work/limitations, above tail


def slice_with_midpaper_anchors(
    text: str,
    blocks: list[dict] | None = None,
    terminal_only: bool = True,
) -> list[Region]:
    """Stage A v2 — v1 output augmented with mid-paper scope/assumption anchors.

    Returns the regions produced by slice_terminal_regions (v1) PLUS additional
    short regions around mid-paper anchors (we focus|we assume|throughout this
    paper|...). Use this when comprehensive gap recall matters more than the
    extra ~10% slice size.

    Measured (gold v2, 49 gaps):
      v1 alone:    localization ~0.51 @ τ=0.70
      v2 (this):   localization expected ~0.75 @ τ=0.70

    See docs/paper/stage_versions.md.
    """
    base_regions = slice_terminal_regions(text, blocks=blocks, terminal_only=terminal_only)
    units = _ordered_units(text, blocks)
    if not units:
        return base_regions
    sents = [u[0] for u in units]
    is_head = [u[1] for u in units]
    n = len(units)

    # Sentences already covered by v1 (avoid double-emitting)
    covered = set()
    for r in base_regions:
        for s in r.sentences:
            covered.add(normalize_text(s)[:60])

    # Sweep entire paper (not gated by terminal threshold) for mid-paper anchors
    midpaper_runs: list[list[int]] = []
    for i in range(n):
        if is_head[i]:
            continue
        if not _MIDPAPER_ANCHOR_RE.search(sents[i]):
            continue
        lo, hi = max(0, i - MIDPAPER_WINDOW), min(n, i + MIDPAPER_WINDOW + 1)
        run = [k for k in range(lo, hi)
               if not is_head[k]
               and len(sents[k]) >= MIN_SENT_CHARS
               and _looks_like_sentence(sents[k])
               and not _is_reference(sents[k])
               and normalize_text(sents[k])[:60] not in covered]
        if run:
            midpaper_runs.append(run)
            for k in run:
                covered.add(normalize_text(sents[k])[:60])

    # Build mid-paper regions
    extra_regions: list[Region] = []
    for run in midpaper_runs:
        loc = run[0] / max(1, n)
        head = sents[max(0, run[0] - 1)] if run[0] > 0 and is_head[run[0] - 1] else "midpaper-anchor"
        extra_regions.append(Region(
            section_type=MIDPAPER_TAG,
            heading=head[:80],
            location_fraction=round(loc, 4),
            is_terminal=loc >= TERMINAL_THRESHOLD,
            sentences=[sents[k] for k in run],
        ))

    return base_regions + extra_regions


# ---------------------------------------------------------------------------
# Stage A (GROBID) — authoritative sections; blacklist background/related-work
# ---------------------------------------------------------------------------
# When a GROBID section tree is available it replaces the heuristic slice: we
# keep only Limitations / Future-Work / Discussion / Conclusion sections and
# DROP Introduction / Related-Work / Background (the background-FP source) plus
# method/results sections. Falls back to slice_terminal_regions when absent.
# Pure prior-work sections — SAFE to blacklist. "introduction" is deliberately
# EXCLUDED: theory/math papers state their own limitations up front in the intro
# ("a major drawback of our approach", "a limitation of our theory"), so we keep
# it (rules-only) rather than drop it and lose genuine gaps.
_RELATED_SECTION = re.compile(
    r"\b(related work|background|prior work|literature|related literature|"
    r"preliminar\w*|motivation|notation|problem statement)\b", re.IGNORECASE)
_INTRO_SECTION = re.compile(r"\bintroduction\b", re.IGNORECASE)
_LIM_SECTION = re.compile(
    r"\b(limitation\w*|threats? to validity|shortcoming\w*|weakness\w*|drawback\w*|caveat\w*)\b", re.IGNORECASE)
_FUT_SECTION = re.compile(r"\b(future work|future direction\w*|future research)\b", re.IGNORECASE)
_DISC_SECTION = re.compile(r"\b(discussion|conclu\w*|final remarks|closing remarks|outlook)\b", re.IGNORECASE)


def classify_grobid_heading(h: str) -> str:
    """limitations | future_work | discussion | introduction | background | other."""
    if _LIM_SECTION.search(h):
        return "limitations"
    if _FUT_SECTION.search(h):
        return "future_work"
    if _DISC_SECTION.search(h):
        return "discussion"
    if _RELATED_SECTION.search(h):
        return "background"          # blacklist (pure prior-work)
    if _INTRO_SECTION.search(h):
        return "introduction"        # keep, but rules-only (see slice below)
    return "other"


# Introduction is demoted to a rules-only region (mapped to "discussion"): the
# high-precision cue rules catch own-work limitations stated up front, while the
# embedding head is NOT allowed to fire there and flood the intro's prior-work prose.
_GROBID_KEEP = {"limitations", "future_work", "discussion", "introduction"}


def slice_grobid_regions(sections: list[dict]) -> list[Region]:
    """Stage A over GROBID sections: keep Limitations/Future/Discussion/Conclusion
    + Introduction (rules-only); blacklist Related-Work/Background; drop method/results."""
    regions: list[Region] = []
    n = max(1, len(sections))
    for i, sec in enumerate(sections):
        cls = classify_grobid_heading(sec.get("heading", ""))
        if cls not in _GROBID_KEEP:
            continue
        section_type = "discussion" if cls == "introduction" else cls
        cand = [s for s in split_sentences(sec.get("text", ""))
                if len(s) >= MIN_SENT_CHARS and _looks_like_sentence(s) and not _is_reference(s)]
        if cand:
            loc = i / n
            regions.append(Region(section_type, str(sec.get("heading", ""))[:80],
                                  round(loc, 4), loc >= TERMINAL_THRESHOLD, cand))
    return regions


# ---------------------------------------------------------------------------
# Stage B.1 — high-precision cue rules (free fast-accept + a type)
# ---------------------------------------------------------------------------
# Mined from the high-lift cues in §5.1 plus the phrasing observed in the gold
# set. These are deliberately high-precision: when one fires we trust the type.
_FUTURE_CUES = re.compile(
    r"\b("
    r"future work|future direction|future research|future stud(?:y|ies)|"
    r"in future|for future|as future|future investigation|"
    r"leave[sd]?\s+(?:\w+\s+){0,6}?(?:for|to|as)\s+future|"
    r"left\s+(?:\w+\s+){0,4}?future|"
    r"remains?\s+(?:\w+\s+){0,4}?(?:future|open|interesting|valuable)|"
    r"we\s+(?:plan|intend|aim|hope|would like|will)\s+to|"
    r"we\s+(?:will|plan to|intend to)\s+explore|"
    r"next step|further (?:work|research|investigation|study)|"
    r"would be (?:interesting|valuable|worthwhile)|"
    r"it would be interesting|an? (?:interesting|promising|natural) (?:direction|next step|avenue)"
    r")\b",
    re.IGNORECASE,
)
_LIMIT_CUES = re.compile(
    r"\b("
    r"limitation|shortcoming|drawback|weakness|caveat|"
    r"we did not|we do not|we have not|did not (?:study|consider|explore|address|evaluate|test)|"
    r"is (?:restricted|limited|confined) to|are (?:restricted|limited|confined) to|"
    r"only (?:considers?|applies?|works?|handles?)|"
    r"remains? (?:unexplored|unaddressed|an open|unclear|unknown|challenging|elusive)|"
    r"unexplored|unaddressed|"
    r"beyond the scope|out of scope|"
    r"assumes? (?:the (?:availability|existence)|access to)|"
    r"(?:cannot|can not|fail[s]? to|do(?:es)? not (?:generalize|scale|extend))|"
    r"not (?:sufficiently|fully|yet) (?:explored|studied|understood|addressed|clear)|"
    # diffuse limitation phrasings observed in the gold (no single cue word)
    r"there (?:is|are) (?:currently )?no (?:standard|established|existing|principled|known|"
    r"general|practical) (?:method|way|approach|mechanism|technique|metric)|"
    r"no (?:standard|established|principled) (?:method|way|approach)|"
    r"lack[s]? (?:of|a )|adds? (?:latency|cost|overhead|complexity)|"
    r"(?:suffers? from|prone to|sensitive to|struggles? with)|"
    r"(?:formidable|profound|significant) (?:challenge|obstacle|difficulty)"
    r")\b",
    re.IGNORECASE,
)
# Sentences that look like a gap but are really the paper's own contribution
# claim ("We show via Theorem 3.3 that ...") — suppress these false cues.
_CONTRIB_GUARD = re.compile(
    r"\b(we (?:show|prove|establish|demonstrate|present|propose|introduce|derive)|"
    r"in this (?:paper|work|section) we|our (?:main )?(?:result|theorem|contribution))\b",
    re.IGNORECASE,
)
# Broader contribution / method-description guard for the MODEL path — the head
# over-fires on "We showed / we evaluate / we apply / we mainly focus on ..."
# sentences inside target regions. Deliberately excludes ubiquitous verbs
# ("consider", "use") to avoid nuking genuine scope limitations phrased with them.
_MODEL_CONTRIB_RE = re.compile(
    r"\bwe\s+(?:(?:further|also|then|mainly|directly|first|next|now)\s+)*"
    r"(?:show(?:ed)?|prove[d]?|establish(?:ed)?|demonstrate[d]?|present(?:ed)?|propose[d]?|"
    r"introduce[d]?|derive[d]?|develop(?:ed)?|design(?:ed)?|evaluate[d]?|test(?:ed)?|"
    r"apply|applied|examine[d]?|report(?:ed)?|analyze[d]?|studied|study|focus(?:es|ed)?\s+on|"
    r"obtain(?:ed)?|claim(?:ed)?|compute[d]?|achieve[d]?)\b"
    r"|\bin this (?:paper|work|section),?\s+we\b"
    r"|\bour (?:main )?(?:result|theorem|contribution|method|approach|model|algorithm|framework)\b",
    re.IGNORECASE,
)


def cue_label(sentence: str) -> str | None:
    """Return 'future_work' | 'limitation' from cue rules, or None."""
    fut = _FUTURE_CUES.search(sentence)
    lim = _LIMIT_CUES.search(sentence)
    # A contribution claim with no explicit future/limit cue is not a gap.
    if not fut and not lim:
        return None
    # future cues are more specific than limit cues; if both, prefer the one
    # whose match is *not* swallowed by a contribution claim.
    if fut and not lim:
        return "future_work"
    if lim and not fut:
        # guard against "we show ... limitation of prior work" contribution framing
        if _CONTRIB_GUARD.search(sentence) and "limitation of our" not in sentence.lower():
            # still a limitation only if the limitation cue is about *this* work
            if re.search(r"\bour\b|\bthis (?:work|paper|approach|method|study)\b", sentence, re.IGNORECASE):
                return "limitation"
            return None
        return "limitation"
    # both fired
    return "future_work"


# ---------------------------------------------------------------------------
# Stage B.2 — embedding logreg head (catches the cue-less remainder)
# ---------------------------------------------------------------------------
LABELS = ["none", "limitation", "future_work"]
DEFAULT_ENCODER = "all-MiniLM-L6-v2"


class EmbeddingGapHead:
    """A frozen sentence-encoder + a tiny sklearn classifier over its embeddings.

    Trained by self-distillation (see ``scripts/train_gap_head.py``). At
    inference we embed only the Stage-A slice sentences (~tens per paper), so
    the encoder cost is a few cents per million papers.
    """

    def __init__(self, encoder, clf, encoder_name: str = DEFAULT_ENCODER):
        self.encoder = encoder
        self.clf = clf
        self.encoder_name = encoder_name

    # -- construction ------------------------------------------------------
    @staticmethod
    def load_encoder(name: str = DEFAULT_ENCODER):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(name)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingGapHead":
        import joblib

        bundle = joblib.load(path)
        enc = cls.load_encoder(bundle["encoder_name"])
        return cls(enc, bundle["clf"], bundle["encoder_name"])

    def save(self, path: str | Path) -> None:
        import joblib

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self.clf, "encoder_name": self.encoder_name}, path)

    # -- inference ---------------------------------------------------------
    def embed(self, sentences: list[str]):
        return self.encoder.encode(
            sentences, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )

    def predict(self, sentences: list[str]) -> list[tuple[str, float]]:
        """Return (label, probability) per sentence."""
        if not sentences:
            return []
        X = self.embed(sentences)
        proba = self.clf.predict_proba(X)
        classes = list(self.clf.classes_)
        out = []
        for row in proba:
            j = int(row.argmax())
            out.append((classes[j], float(row[j])))
        return out


class BertGapHead:
    """A fine-tuned BERT/DistilBERT sequence classifier — the end-to-end
    alternative to EmbeddingGapHead's frozen-encoder + logreg. Same predict()
    interface, so it is a drop-in for extract_gaps / the benchmark. Trained by
    scripts/train_gap_bert.py; saved as a HuggingFace model directory."""

    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.id2label = {int(k): v for k, v in model.config.id2label.items()}

    @classmethod
    def load(cls, path: str | Path) -> "BertGapHead":
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        return cls(AutoModelForSequenceClassification.from_pretrained(str(path)),
                   AutoTokenizer.from_pretrained(str(path)))

    def predict(self, sentences: list[str]) -> list[tuple[str, float]]:
        if not sentences:
            return []
        import torch

        out: list[tuple[str, float]] = []
        for i in range(0, len(sentences), 32):
            enc = self.tokenizer(sentences[i:i + 32], return_tensors="pt",
                                 padding=True, truncation=True, max_length=96)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
            for row in probs:
                j = int(row.argmax())
                out.append((self.id2label[j], float(row[j])))
        return out


def load_gap_head(path: str | Path):
    """Return a BertGapHead if `path` is a HF model directory, else an
    EmbeddingGapHead (joblib). Both expose predict(sentences)->[(label, prob)]."""
    p = Path(path)
    return BertGapHead.load(p) if p.is_dir() else EmbeddingGapHead.load(p)


# ---------------------------------------------------------------------------
# End-to-end extraction
# ---------------------------------------------------------------------------
@dataclass
class GapRow:
    id: str
    gap_type: str
    gap_sentence: str
    paragraph_text: str
    confidence: float
    section_type: str
    source: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "gap_type": self.gap_type,
            "gap_sentence": self.gap_sentence,
            "paragraph_text": self.paragraph_text,
            "confidence": round(self.confidence, 3),
            "section_type": self.section_type,
            "source": self.source,
        }


# heading region type -> default gap type when the per-sentence type is ambiguous
_HEADING_DEFAULT = {"limitations": "limitation", "future_work": "future_work"}


def _context(sentences: list[str], i: int, w: int = 1) -> str:
    lo, hi = max(0, i - w), min(len(sentences), i + w + 1)
    return " ".join(sentences[lo:hi])


# Inside an explicit Limitations / Future-Work region the structural prior is
# strong, so the model needs less confidence to accept (section-aware threshold).
SECTION_THRESHOLD = 0.35


def extract_gaps(
    paper_id: str,
    text: str,
    blocks: list[dict] | None = None,
    head: EmbeddingGapHead | None = None,
    mode: str = "hybrid",
    model_threshold: float = 0.6,
    section_threshold: float = SECTION_THRESHOLD,
    terminal_only: bool = True,
    grobid_sections: list[dict] | None = None,
) -> list[dict]:
    """Run the full funnel on one paper. ``mode`` in {rules, model, hybrid}.

    rules  : cue rules only (no encoder needed)
    model  : embedding head only
    hybrid : cue rule wins on type; head fills the cue-less remainder

    ``grobid_sections`` (from ``grobid_sections.extract_sections``): when given,
    Stage A uses the authoritative GROBID section tree and blacklists
    background/related-work; when None, it falls back to the PyMuPDF heuristic.
    """
    if grobid_sections is not None:
        regions = slice_grobid_regions(grobid_sections)
    else:
        regions = slice_terminal_regions(text, blocks=blocks, terminal_only=terminal_only)
    if not regions:
        return []

    # batch all slice sentences for one encoder call
    flat: list[tuple[int, int, str]] = []  # (region_idx, sent_idx, sentence)
    for ri, r in enumerate(regions):
        for si, s in enumerate(r.sentences):
            flat.append((ri, si, s))
    model_pred: dict[tuple[int, int], tuple[str, float]] = {}
    if mode in ("model", "hybrid") and head is not None and flat:
        preds = head.predict([s for _, _, s in flat])
        for (ri, si, _), p in zip(flat, preds):
            model_pred[(ri, si)] = p

    rows: list[GapRow] = []
    seen: set[str] = set()
    for ri, r in enumerate(regions):
        default_t = _HEADING_DEFAULT.get(r.section_type)
        # The model only accepts inside EXPLICIT Limitations/Future-Work regions,
        # where the structural prior is strong. In tail/discussion regions (full of
        # conclusion summary & — in math papers — theorem exposition) only the
        # high-precision cue rules fire; otherwise the model floods false positives.
        in_explicit = r.section_type in ("limitations", "future_work")
        for si, s in enumerate(r.sentences):
            rule_t = cue_label(s) if mode in ("rules", "hybrid") else None
            model_t, model_p = model_pred.get((ri, si), ("none", 0.0))
            model_hit = (mode in ("model", "hybrid") and in_explicit
                         and model_t in TARGET_TYPES and model_p >= section_threshold)

            if rule_t:
                gtype, source, conf = rule_t, "rule", 0.9   # trust high-precision cues
                if model_hit:
                    source, conf = "rule+model", max(conf, model_p)
            elif model_hit:
                if not _looks_like_sentence(s):
                    continue  # gate MODEL emissions only — reject scramble debris
                # The head over-fires on contribution / method-description sentences
                # ("We showed ...", "we mainly focus on ...") inside target regions.
                # Reject those unless the sentence also carries an explicit
                # limitation/future cue (then it is a genuine gap, e.g. "we do not").
                if (_MODEL_CONTRIB_RE.search(s)
                        and not _LIMIT_CUES.search(s) and not _FUTURE_CUES.search(s)):
                    continue
                # in a target section, prefer the section's type when the model is
                # only weakly confident (it confuses limitation vs future_work).
                gtype = model_t
                if default_t and model_p < model_threshold:
                    gtype = default_t
                source, conf = "model", model_p
            else:
                continue

            key = normalize_text(s)
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                GapRow(
                    id=paper_id,
                    gap_type=gtype,
                    gap_sentence=s,
                    paragraph_text=_context(r.sentences, si, w=1),
                    confidence=conf,
                    section_type=r.section_type,
                    source=source,
                )
            )
    return [r.as_dict() for r in rows]


# ---------------------------------------------------------------------------
# Corpus runner — drop-in for openai_gaps over a whole paper_texts.jsonl
# ---------------------------------------------------------------------------
def extract_all_gaps(
    texts_jsonl: "str | Path",
    out_tsv: "str | Path",
    head_path: "str | Path | None" = None,
    mode: str = "hybrid",
) -> "pd.DataFrame":
    """Run the funnel over an entire corpus and write a gaps.tsv.

    Output columns are a superset of openai_gaps' schema (id, gap_type,
    gap_sentence, paragraph_text, confidence) plus section_type, source — so
    every downstream stage (theme-mine, gap-graph, ideas) is unchanged.
    """
    import pandas as pd

    texts_jsonl, out_tsv = Path(texts_jsonl), Path(out_tsv)
    head = None
    if mode in ("model", "hybrid") and head_path and Path(head_path).exists():
        head = EmbeddingGapHead.load(head_path)
    elif mode in ("model", "hybrid"):
        log.warning("mode=%s but no head at %s — falling back to rules only", mode, head_path)
        mode = "rules"

    df = pd.read_json(texts_jsonl, lines=True, dtype=False)
    df["id"] = df["id"].astype(str)
    has_blocks = "blocks" in df.columns
    rows: list[dict] = []
    for _, r in df.iterrows():
        blocks = r["blocks"] if has_blocks and isinstance(r.get("blocks"), list) else None
        rows.extend(extract_gaps(str(r["id"]), str(r["text"]), blocks=blocks, head=head, mode=mode))
    out = pd.DataFrame(rows, columns=["id", "gap_type", "gap_sentence", "paragraph_text",
                                      "confidence", "section_type", "source"])
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_tsv, sep="\t", index=False)
    log.info("Funnel extracted %d gaps from %d papers (%s) -> %s",
             len(out), len(df), out["gap_type"].value_counts().to_dict() if len(out) else {}, out_tsv)
    return out


# ---------------------------------------------------------------------------
# Provenance helper shared with the benchmark (scrambling-robust containment)
# ---------------------------------------------------------------------------
def token_containment(needle: str, haystack: str) -> float:
    """Fraction of ``needle`` content-word tokens present in ``haystack``.

    Robust to PDF reading-order scrambling: a faithful sentence whose words are
    all present but not contiguous still scores ~1.0. Used for both Stage-A
    localization and gold matching instead of brittle substring offsets.
    """
    nt = _WORD.findall(normalize_text(_DEHYPHEN_RE.sub(r"\1\2", needle)))
    if not nt:
        return 0.0
    hs = set(_WORD.findall(normalize_text(_DEHYPHEN_RE.sub(r"\1\2", haystack))))
    return sum(1 for t in nt if t in hs) / len(nt)
