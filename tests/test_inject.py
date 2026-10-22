"""Planted value never equals / contains / is contained in any gold answer, alias or gold hop answer."""
import pytest

from src.data import items_path, load_items
from src.inject import build_off_pool, check_validity, format_ok, gold_strings, pick_value
from src.normalize import overlaps


def make_item(iid, hops, answer, aliases=(), cands=(), context_text="filler"):
    return {
        "id": iid, "question": "q?", "answer": answer, "answer_aliases": list(aliases), "K": len(hops),
        "hops": [{"answer": a, "tag": t} for a, t in hops],
        "context": [{"title": "T", "paragraph_text": context_text}],
        "distractor_candidates": [{"text": x, "tag": t, "source": s} for x, t, s in cands],
    }


def assert_never_overlaps(item, value):
    for g in gold_strings(item):
        assert not overlaps(value, g), f"planted {value!r} overlaps gold {g!r}"


def test_rule1_rejects_substring_and_superstring():
    item = make_item("a", [("Paris", "GPE"), ("France", "GPE"), ("Europe", "LOC")], "Europe",
                     aliases=["EU"])
    for bad in ["Paris", "paris, texas", "Par", "France", "the France"]:
        assert check_validity(item, "Paris", bad, "GPE", "GPE", "") == "rule1_overlaps_gold"
    assert check_validity(item, "Paris", "Lyon", "GPE", "GPE", "") is None


def test_in_context_skips_overlapping_candidates():
    item = make_item("a", [("Aston Villa", "ORG"), ("Birmingham City", "ORG"), ("1 December 2010", "DATE_DMY")],
                     "1 December 2010",
                     cands=[("Aston Villa F.C.", "ORG", "title"), ("Birmingham", "ORG", "ner"),
                            ("Leeds United", "ORG", "ner"), ("Chelsea", "ORG", "title")])
    v, reason, _ = pick_value(item, 1, "Aston Villa", "in_context", "Hop 1: x -> Aston Villa", [])
    assert reason is None and v in {"Leeds United", "Chelsea"}
    assert_never_overlaps(item, v)


def test_rule4_rejects_value_in_clean_chain():
    item = make_item("a", [("Lyon", "GPE"), ("x", "OTHER"), ("y", "OTHER")], "y",
                     cands=[("Nice", "GPE", "title")])
    v, reason, _ = pick_value(item, 1, "Lyon", "in_context", "Hop 1: near Nice -> Lyon", [])
    assert v is None and "rule4" in reason


def test_off_context_not_in_context_and_type_matched():
    a = make_item("a", [("1990", "YEAR"), ("x", "OTHER"), ("y", "OTHER")], "y", context_text="founded 1987")
    b = make_item("b", [("1987", "YEAR"), ("1991", "YEAR"), ("Rome", "GPE")], "Rome")
    pool = build_off_pool([a, b])
    v, reason, _ = pick_value(a, 1, "1990", "off_context", "Hop 1: when -> 1990", pool)
    assert v == "1991"   # 1987 appears in a's context; Rome is the wrong type
    assert_never_overlaps(a, v)


def test_format_rule():
    assert format_ok("1990", "1987")
    assert not format_ok("1990", "March 1987")
    assert not format_ok("Paris", "paris")


@pytest.mark.skipif(not items_path("dev").exists(), reason="run prepare-data first")
def test_real_items_never_overlap_gold():
    items = load_items(["dev"], [3, 4], 100)
    pool = build_off_pool(items)
    n_planted = 0
    for it in items:
        for k, h in enumerate(it["hops"], start=1):
            for et in ("in_context", "off_context"):
                v, reason, _ = pick_value(it, k, h["answer"], et, "", pool)
                if v is not None:
                    n_planted += 1
                    assert_never_overlaps(it, v)
    assert n_planted > 100
