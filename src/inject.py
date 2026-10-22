"""Choose planted values (doc §6.4) and check validity rules (doc §6.5).

in_context : same type tag, from the item's 4 shown distractor paragraphs
             (titles first, then NER/regex strings; closest length wins).
off_context: same type tag, a gold hop answer of another item that appears nowhere
             in this item's question or shown paragraphs (closest length wins).
"""
from __future__ import annotations

import re

from .normalize import normalize, overlaps
from .store import cell_seed

YEAR_FORMAT = re.compile(r"^\d{4}$")


def gold_strings(item: dict) -> list[str]:
    return [item["answer"], *item.get("answer_aliases", []), *(h["answer"] for h in item["hops"])]


def _casing(s: str) -> str:
    letters = [w for w in s.split() if w[:1].isalpha()]
    if not letters:
        return "none"
    if all(w[0].isupper() for w in letters):
        return "title"
    if all(w.islower() for w in letters):
        return "lower"
    return "mixed"


def format_ok(orig: str, new: str) -> bool:
    """Rule 3: same casing style and same trailing punctuation; digits stay digits."""
    if bool(YEAR_FORMAT.match(orig)) != bool(YEAR_FORMAT.match(new)):
        return False
    if any(c.isdigit() for c in orig) != any(c.isdigit() for c in new):
        return False
    if (orig[-1:] in ".,;") != (new[-1:] in ".,;"):
        return False
    co, cn = _casing(orig), _casing(new)
    # same style, or one side "mixed" (e.g. "Battle of Hastings") vs title/lower
    return co == cn or ("mixed" in (co, cn) and "none" not in (co, cn))


def check_validity(item: dict, orig_value: str, new_value: str, gold_tag: str,
                   new_tag: str, clean_chain: str) -> str | None:
    """Return None if valid, else the name of the first rule that fails."""
    if not normalize(new_value):
        return "empty_value"
    if any(overlaps(new_value, g) for g in gold_strings(item)) or overlaps(new_value, orig_value):
        return "rule1_overlaps_gold"
    if new_tag != gold_tag:
        return "rule2_type_mismatch"
    if not format_ok(orig_value, new_value):
        return "rule3_format"
    if normalize(new_value) in normalize(clean_chain):
        return "rule4_in_clean_chain"
    return None


def _ranked(cands: list[dict], orig: str, prefer_titles: bool, salt: str) -> list[dict]:
    """Sort candidates: titles first (in_context), then length bucket (|diff|//3), then a
    seeded tiebreak per item x hop, so ties don't always pick the same value."""
    return sorted(cands, key=lambda c: (
        0 if (prefer_titles and c.get("source") == "title") else 1,
        abs(len(c["text"]) - len(orig)) // 3,
        cell_seed(salt, c["text"])))


def pick_value(item: dict, k: int, orig_value: str, error_type: str, clean_chain: str,
               off_pool: list[dict]) -> tuple[str | None, str | None, int]:
    """Return (new_value, drop_reason, n_candidates_tried).

    Walks the ranked candidates of the right type and returns the first one that passes
    every validity rule. If none passes, reports the rule that failed for the best candidate.
    """
    tag = item["hops"][k - 1]["tag"]
    salt = f"13|{item['id']}|{k}|{error_type}"
    if error_type == "in_context":
        pool = [c for c in item["distractor_candidates"] if c["tag"] == tag]
        ranked = _ranked(pool, orig_value, prefer_titles=True, salt=salt)
    else:
        context = normalize(item["question"] + " " + " ".join(
            p["title"] + " " + p["paragraph_text"] for p in item["context"]))
        pool = [c for c in off_pool if c["tag"] == tag and c["item_id"] != item["id"]
                and normalize(c["text"]) and normalize(c["text"]) not in context]
        ranked = _ranked(pool, orig_value, prefer_titles=False, salt=salt)
    if not ranked:
        return None, f"no_candidate_with_tag_{tag}", 0
    first_reason = None
    for c in ranked:
        reason = check_validity(item, orig_value, c["text"], tag, c["tag"], clean_chain)
        if reason is None:
            return c["text"], None, len(ranked)
        first_reason = first_reason or reason
    return None, f"all_candidates_invalid:{first_reason}", len(ranked)


def build_off_pool(items: list[dict]) -> list[dict]:
    """Gold hop answers of all prepared items whose type tag is confident (context and
    standalone NER agree), skipping sentence-like answers (annotation noise)."""
    seen, pool = set(), []
    for it in items:
        for h in it["hops"]:
            if not h.get("tag_confident", True) or len(h["answer"].split()) > 6:
                continue
            key = (h["answer"].lower(), h["tag"])
            if key not in seen:
                seen.add(key)
                pool.append({"text": h["answer"], "tag": h["tag"], "item_id": it["id"]})
    return pool
