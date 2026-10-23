"""SQuAD-style answer normalization, EM and token F1."""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_PUNCT = set(string.punctuation) | {"‘", "’", "“", "”", "–", "—"}


def normalize(s: str | None) -> str:
    """Lowercase, strip accents, remove punctuation and articles, collapse whitespace."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = "".join(ch if ch not in _PUNCT else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str | None, golds: list[str]) -> bool:
    p = normalize(pred)
    return bool(p) and any(p == normalize(g) for g in golds)


def token_f1(pred: str | None, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def max_f1(pred: str | None, golds: list[str]) -> float:
    return max((token_f1(pred, g) for g in golds), default=0.0)


def hop_answer_matches(pred: str | None, golds: list[str], f1_threshold: float = 0.8, max_extra_words: int = 3) -> bool:
    """Does a model's hop answer match a gold hop answer?

    Exact match or token-F1 >= threshold (doc §6.1), after dropping parentheticals the model
    adds ("Vatican City (assumed ...)"). Also accepts whole-word containment in either direction
    when the longer side has at most `max_extra_words` extra words ("Da Nang, Vietnam" for
    "Da Nang", "Vivaldi" for "Antonio Vivaldi"), so sentences that merely mention it do not match.
    """
    if not pred:
        return False
    core = re.sub(r"\([^)]*\)", " ", pred).strip() or pred
    if exact_match(core, golds) or max_f1(core, golds) >= f1_threshold:
        return True
    p = normalize(core)
    for g in golds:
        ng = normalize(g)
        if not ng or not p:
            continue
        # either side may carry a few extra words: "Da Nang, Vietnam" ~ "Da Nang", "Vivaldi" ~ "Antonio Vivaldi"
        short, long_ = sorted([p, ng], key=lambda s: len(s.split()))
        if f" {short} " in f" {long_} " and len(long_.split()) <= len(short.split()) + max_extra_words:
            return True
    return False


def overlaps(a: str, b: str) -> bool:
    """True if normalized a equals, contains, or is contained in normalized b.

    Plain substring (strict on purpose): "Art" counts as inside "Arthur".
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return na in nb or nb in na
