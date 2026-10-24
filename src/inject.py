"""Choose planted values (doc §6.4) and check validity rules (doc §6.5).

in_context : same type tag, from the item's 4 shown distractor paragraphs
             (titles first, then NER/regex strings; closest length wins).
off_context: same type tag, a gold hop answer of another item that appears nowhere
             in this item's question or shown paragraphs; answers to the same Wikidata
             relation ("X >> country") first, then closest length.
With vetting on (config `vet_values`), a candidate must also pass the LLM judge in src/vet.py,
because tags alone let a state replace a city or a network replace a person.
"""
from __future__ import annotations

import re

from .normalize import normalize, overlaps
from .store import cell_seed
from .vet import resolve_refs, vet_key

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


def relation(subquestion: str) -> str | None:
    """What a hop asks for, used to prefer off_context values that answer the same kind of question.

    Wikidata-style hops: "Yaxing Coach >> headquarters location" -> "rel:headquarters location".
    Natural-language hops: first two words, "What city did ..." -> "q:what city".
    """
    if ">>" in subquestion:
        r = subquestion.split(">>", 1)[1].strip().lower()
        return f"rel:{r}" if r else None
    words = re.findall(r"[a-z]+", subquestion.lower())
    return f"q:{' '.join(words[:2])}" if len(words) >= 2 else None


def _ranked(cands: list[dict], orig: str, prefer_titles: bool, salt: str, rel: str | None = None) -> list[dict]:
    """Sort candidates: same relation first (off_context), titles first (in_context), then
    length bucket (|diff|//3), then a seeded tiebreak per item x hop so ties vary."""
    return sorted(cands, key=lambda c: (
        0 if (rel is not None and c.get("relation") == rel) else 1,
        0 if (prefer_titles and c.get("source") == "title") else 1,
        abs(len(c["text"]) - len(orig)) // 3,
        cell_seed(salt, c["text"])))


def judge_question(item: dict, k: int, gold_idx: int | None = None) -> tuple[str, str]:
    """(sub-question with #N refs filled in, gold hop answer): what the vet judge sees."""
    gold_hop = item["hops"][k - 1 if gold_idx is None else gold_idx]
    return resolve_refs(gold_hop.get("question", ""), [h["answer"] for h in item["hops"]]), gold_hop["answer"]


def valid_candidates(item: dict, k: int, orig_value: str, error_type: str, clean_chain: str,
                     off_pool: list[dict], gold_idx: int | None = None):
    """Yield (candidate text, None) for each ranked candidate of the right tag that passes every
    validity rule, in rank order. If none passes, yields once (None, drop reason) at the end."""
    gold_hop = item["hops"][k - 1 if gold_idx is None else gold_idx]   # type of the gold hop at this position
    tag = gold_hop["tag"]
    salt = f"13|{item['id']}|{k}|{error_type}"
    if error_type == "in_context":
        pool = [c for c in item["distractor_candidates"] if c["tag"] == tag]
        ranked = _ranked(pool, orig_value, prefer_titles=True, salt=salt)
    else:
        context = normalize(item["question"] + " " + " ".join(
            p["title"] + " " + p["paragraph_text"] for p in item["context"]))
        pool = [c for c in off_pool if c["tag"] == tag and c["item_id"] != item["id"]
                and normalize(c["text"]) and normalize(c["text"]) not in context]
        ranked = _ranked(pool, orig_value, prefer_titles=False, salt=salt,
                         rel=relation(gold_hop.get("question", "")))
    if not ranked:
        yield None, f"no_candidate_with_tag_{tag}"
        return
    first_reason, any_valid = None, False
    for c in ranked:
        reason = check_validity(item, orig_value, c["text"], tag, c["tag"], clean_chain)
        if reason is None:
            any_valid = True
            yield c["text"], None
        first_reason = first_reason or reason
    if not any_valid:
        yield None, f"all_candidates_invalid:{first_reason}"


def pick_value(item: dict, k: int, orig_value: str, error_type: str, clean_chain: str,
               off_pool: list[dict], gold_idx: int | None = None,
               vetted: dict[str, bool] | None = None) -> tuple[str | None, str | None, int]:
    """Return (new_value, drop_reason, n_valid_candidates_seen).

    Without `vetted`: the first candidate that passes every validity rule.
    With `vetted` (vet cache, see src/vet.py): the first valid candidate the judge accepted.
    Valid candidates are vetted in rank order, so a candidate with no verdict means the vet step
    stopped here (its per-cell limit) and every earlier one was rejected.
    """
    subq, gold = judge_question(item, k, gold_idx) if vetted is not None else ("", "")
    n = 0
    for text, reason in valid_candidates(item, k, orig_value, error_type, clean_chain, off_pool, gold_idx):
        if text is None:
            return None, reason, n
        n += 1
        if vetted is None:
            return text, None, n
        verdict = vetted.get(vet_key(subq, gold, text))
        if verdict is None:
            return None, ("vet_rejected_all_vetted" if n > 1 else "not_vetted"), n
        if verdict:
            return text, None, n
    return None, "vet_rejected_all", n


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
                pool.append({"text": h["answer"], "tag": h["tag"], "item_id": it["id"],
                             "relation": relation(h.get("question", ""))})
    return pool
