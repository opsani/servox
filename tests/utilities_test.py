from servo.utilities import join_to_series


def test_join_to_series_empty() -> None:
    words = []
    assert join_to_series(words) == ""


def test_join_to_series_one() -> None:
    words = ["this"]
    assert join_to_series(words) == "this"


def test_join_to_series_two() -> None:
    words = ["this", "that"]
    assert join_to_series(words) == "this and that"


def test_join_to_series_three() -> None:
    words = ["this", "that", "the other"]
    assert join_to_series(words) == "this, that, and the other"


def test_join_to_series_three_or() -> None:
    words = ["this", "that", "the other"]
    assert join_to_series(words, conjunction="or") == "this, that, or the other"


def test_join_to_series_three_no_oxford_comma() -> None:
    words = ["this", "that", "the other"]
    assert join_to_series(words, oxford_comma=False) == "this, that and the other"
