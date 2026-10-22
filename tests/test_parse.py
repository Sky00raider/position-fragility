from src.parse import extract_final, hops_well_formed, parse_hops, split_think


def final(text, think=False):
    _, ans, _ = split_think(text, think)
    return extract_final(ans)


def test_plain():
    assert final("Hop 1: a -> b\nFinal answer: Paris") == "Paris"


def test_bold_label_and_trailing_period():
    assert final("**Final answer:** Paris.") == "Paris"
    assert final("**Final Answer**: **Paris**") == "Paris"


def test_last_final_answer_wins():
    assert final("Final answer: Rome\nWait, that's wrong.\nFinal answer: Paris") == "Paris"


def test_answer_is_prefix_and_quotes():
    assert final('Final answer: The answer is "1990".') == "1990"


def test_label_alone_then_next_line():
    assert final("Final answer:\n  Paris\n") == "Paris"


def test_no_final_answer():
    assert final("Hop 1: a -> b") is None
    assert final("Final answer:   ") is None


def test_think_mode_only_after_close():
    # final answer only inside the think block does not count
    assert final("<think>\nFinal answer: Paris\n</think>\n\nI am not sure.", think=True) is None
    assert final("<think>\nhmm\n</think>\n\nFinal answer: Paris", think=True) == "Paris"


def test_think_mode_truncated():
    think, ans, closed = split_think("<think>\nstill thinking", True)
    assert not closed and ans == "" and extract_final(ans) is None


def test_unicode_answer():
    assert final("Final answer: São Paulo") == "São Paulo"
    assert final("Final answer: “Hamlet”") == "Hamlet"


def test_hop_lines_extra_arrow_and_markdown():
    text = ("Hop 1: Who wrote A -> B? -> Jane Doe\n"
            "**Hop 2:** Where was Jane Doe born? -> **London**.\n"
            "Hop 3: country -> England\nFinal answer: England")
    hops = parse_hops(text)
    assert [h.answer for h in hops] == ["Jane Doe", "London", "England"]
    assert hops_well_formed(hops, 3)
    # answer spans point at the right text
    assert text[hops[1].ans_start:hops[1].ans_end] == "London"
    assert text[hops[0].ans_start:hops[0].ans_end] == "Jane Doe"


def test_hop_numbering_must_be_1_to_K():
    hops = parse_hops("Hop 1: a -> x\nHop 3: c -> z\nHop 4: d -> w")
    assert not hops_well_formed(hops, 3)


def test_unicode_arrow():
    hops = parse_hops("Hop 1: who → Alice")
    assert hops[0].answer == "Alice"
