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
