"""Tier 0 of the gap-extraction funnel: a recall-tuned lexical prefilter.

A sentence is a *candidate* gap if any of its 1..max_n word n-grams is in a
mined cue-phrase dictionary. This is the cheap first stage that drops the
~90% of sentences that cannot be gaps, before any model runs.

Design priority: RECALL. Tier 0 is the recall ceiling for everything
downstream — a real gap dropped here is lost forever. Precision is expected to
be low; Tier 1 handles precision.

This module is the single source of truth for text normalization and sentence
splitting. The miner (`scripts/mine_tier0_dictionary.py`) and the evaluator
(`scripts/eval_tier0.py`) MUST import these helpers so a phrase mined from
training matches identically at inference.

Matching is O(sentence length), independent of dictionary size: we build the
set of the sentence's n-grams and intersect with the phrase set. For
production-scale millions of papers this is already cheap; if the dictionary
grows huge, swap the set-intersection for Aho-Corasick (e.g. pyahocorasick).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Normalization  (KEEP IN SYNC across miner / matcher / eval)
# ---------------------------------------------------------------------------
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]*\}\}")                 # {{formula:..}} {{cite:..}}
_MATH_RE = re.compile(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]*\$")    # \(..\)  \[..\]  $..$
_NONWORD_RE = re.compile(r"[^a-z0-9'\- ]+")
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """Lowercase, strip math/citation placeholders and punctuation, collapse ws."""
    if not s:
        return ""
    s = s.lower()
    s = _PLACEHOLDER_RE.sub(" ", s)
    s = _MATH_RE.sub(" ", s)
    s = _NONWORD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def tokenize(s: str) -> list[str]:
    """Normalize then split on whitespace."""
    return normalize_text(s).split()


def ngrams(tokens: list[str], n_min: int = 1, n_max: int = 3) -> set[str]:
    """All space-joined n-grams for n in [n_min, n_max]."""
    out: set[str] = set()
    L = len(tokens)
    for n in range(n_min, n_max + 1):
        if n > L:
            break
        for i in range(L - n + 1):
            out.add(" ".join(tokens[i : i + n]))
    return out


# ---------------------------------------------------------------------------
# Sentence splitting (scientific-text aware)
# ---------------------------------------------------------------------------
_ABBREVIATIONS = [
    "e.g.", "i.e.", "cf.", "vs.", "etc.", "et al.", "Fig.", "Figs.",
    "Eq.", "Eqs.", "Sec.", "Secs.", "Ref.", "Refs.", "Tab.", "Tabs.",
    "Dr.", "Mr.", "Mrs.", "Prof.", "St.", "No.", "Vol.", "pp.",
    "approx.", "viz.", "resp.", "U.S.", "U.K.",
]
_ABBR_SENTINEL = "\x00"
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\{(\[\$])")
_RAW_WS_RE = re.compile(r"\s+")


def split_sentences(text: str, min_chars: int = 15) -> list[str]:
    """Split scientific text into sentences (raw, not normalized)."""
    if not text:
        return []
    norm = _RAW_WS_RE.sub(" ", text).strip()
    for abbr in _ABBREVIATIONS:
        norm = norm.replace(abbr, abbr.replace(".", _ABBR_SENTINEL))
    parts = _SPLIT_RE.split(norm)
    out = []
    for p in parts:
        s = p.replace(_ABBR_SENTINEL, ".").strip()
        if len(s) >= min_chars:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------
class GapPrefilter:
    """Loads a mined cue-phrase dictionary and flags candidate gap sentences."""

    def __init__(self, phrases: set[str] | list[str], max_n: int = 3):
        self.phrases: set[str] = set(phrases)
        self.max_n = max_n

    @classmethod
    def from_json(cls, path: str | Path) -> "GapPrefilter":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        phrases = [p["phrase"] if isinstance(p, dict) else p for p in data["dictionary"]]
        return cls(phrases=phrases, max_n=int(data.get("max_n", 3)))

    def matched_phrases(self, sentence: str) -> set[str]:
        grams = ngrams(tokenize(sentence), 1, self.max_n)
        return grams & self.phrases

    def is_candidate(self, sentence: str) -> bool:
        grams = ngrams(tokenize(sentence), 1, self.max_n)
        # Early-exit intersection
        return not grams.isdisjoint(self.phrases)

    def filter(self, sentences: list[str]) -> list[int]:
        """Return indices of sentences flagged as candidate gaps."""
        return [i for i, s in enumerate(sentences) if self.is_candidate(s)]
