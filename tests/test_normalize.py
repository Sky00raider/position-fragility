from src.normalize import exact_match, normalize, overlaps, token_f1


def test_squad_normalize():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"
    assert normalize("Café") == "cafe"


def test_em_aliases():
    assert exact_match("the USA", ["United States", "USA"])
    assert not exact_match(None, ["x"])
    assert not exact_match("", [""])


def test_f1():
    assert token_f1("Barack Obama", "Obama") == 2 / 3


def test_overlaps_both_directions():
    assert overlaps("Paris", "Paris, Texas")
    assert overlaps("Paris, Texas", "paris")
    assert overlaps("Art", "Arthur")          # strict on purpose
    assert not overlaps("London", "Paris")


def test_hop_answer_matches():
    from src.normalize import hop_answer_matches
    assert hop_answer_matches("Da Nang, Vietnam", ["Da Nang"])
    assert hop_answer_matches("22 times", ["22"])
    assert hop_answer_matches("Vatican City (assumed based on his role as Pope)", ["Vatican City"])
    assert hop_answer_matches("Pope Pius XI", ["Pius XI"])
    assert not hop_answer_matches("The headquarters of the Warsaw Treaty armed forces was in Warsaw, Poland.", ["Poland"])
    assert not hop_answer_matches("ABC News", ["NBC"])
    assert not hop_answer_matches(None, ["x"])


def test_hop_answer_matches_shorter_pred():
    from src.normalize import hop_answer_matches
    assert hop_answer_matches("Vivaldi", ["Antonio Vivaldi"])
    assert hop_answer_matches("15th century", ["built in the 15th century"])
    assert not hop_answer_matches("Poland", ["the USSR"])


def test_align_hops_respects_dependencies():
    from src.pipeline import align_hops
    # 3hop2 shape: hops 1 and 2 independent, hop 3 depends on both
    item = {"K": 3, "answer": "C", "answer_aliases": [],
            "hops": [{"answer": "A", "depends_on": []}, {"answer": "B", "depends_on": []},
                     {"answer": "C", "depends_on": [1, 2]}]}
    assert align_hops(["A", "B", "C"], item, 0.8) == ([0, 1, 2], None)
    assert align_hops(["B", "A", "C"], item, 0.8) == ([1, 0, 2], None)          # independent hops swapped: fine
    assert align_hops(["A", "C", "B"], item, 0.8)[1] == "hop_order_violates_dependencies"
    assert align_hops(["A", "X", "C"], item, 0.8)[1] == "hop_mismatch_2"
