"""Scorers for corpus evals (no models or data needed)."""
from src.eval_corpus import _ids, score


def test_number_ignores_dates_decimals_and_tags():
    s = {"type": "number", "v": 5}
    assert score("Walid worked on exactly 5 projects [G1].", s)
    assert score("He has five projects.", s) and score("Er hat fünf Projekte.", s)
    assert not score("Born 05.05.2001; grade 2,5 [P5].", s)
    assert not score("At least two projects [P1].", s)


def test_set_and_contains_with_variants_and_forbidden_terms():
    s = {"type": "set", "v": ["Cluster Computing", "Computer Vision", "Software Engineering"],
         "forbid": ["Advanced Software Engineering"]}
    assert score("Cluster Computing, Computer Vision 1 and Software Engineering.", s)
    assert not score("Cluster Computing, Computer Vision 1, Advanced Software Engineering.", s)
    assert score("Er spricht Arabisch und Deutsch.", {"type": "contains_all", "v": [["Arabic", "Arabisch"], ["German", "Deutsch"]]})
    assert not score("Python only", {"type": "contains_any", "v": ["Neo4j"]})


def test_duration_must_not_be_rounded():
    s = {"type": "duration", "years": 2, "months": 9}
    assert score("2 years 9 months [C1]", s) and score("33 months", s)
    assert not score("3 years [C1]", s) and not score("about 2 years", s)
    assert score("10 Monate (315 Tage)", {"type": "duration", "years": 0, "months": 10})


def test_refusals_are_recognised():
    s = {"type": "refuse"}
    assert score("I don't have enough information.", s)
    assert score("The documents do not mention his salary.", s)
    assert score("There is no grade for Cluster Computing yet; it is only approved.", s)
    assert not score("Walid earns 40,000 euros.", s)


def test_id_ranges():
    assert _ids("1-3,26") == {1, 2, 3, 26} and _ids(None) is None
