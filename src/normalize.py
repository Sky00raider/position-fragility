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


def overlaps(a: str, b: str) -> bool:
    """True if normalized a equals, contains, or is contained in normalized b.

    Plain substring (strict on purpose): "Art" counts as inside "Arthur".
    """
    na, nb = normalize(a), normalize(b)
    if not na or not nb:
        return False
    return na in nb or nb in na
