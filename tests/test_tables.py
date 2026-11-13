"""The general table reader: rows, merges, copies, legends and answers (no models or databases)."""
from src import tables
from src.tables import cell_class, find_tables, to_number

TRANSCRIPT = """Overview of results - page 1
Course Status Credits Grade
Databases PNr 1001 P 5 1,7
Statistics PNr 1002 P 5 2,3
Compilers PNr 1003 A
Operating Systems PNr 1004 P 5 3,0
Status: P = passed, F = failed, A = admitted
Footer line one
Footer line two
Footer line three
Overview of results - page 2
Course Status Credits Grade
Networks PNr 1005 P 5 1,0
Robotics PNr 1006 A
Ergebnisübersicht - Seite 1
Kurs Status Punkte Note
Datenbanken PNr 1001 P 5 1,7
Statistik PNr 1002 P 5 2,3
Übersetzerbau PNr 1003 A
Betriebssysteme PNr 1004 P 5 3,0
Status: P = bestanden, F = nicht bestanden, A = zugelassen
"""

FINANCIALS = """Konzern-Gewinn- und Verlustrechnung (in Mio. €)
Posten 2024 2023
Umsatzerlöse 12.430 11.205
Herstellungskosten (7.812) (7.390)
Forschung und Entwicklung 1.104 998
Operatives Ergebnis 2.214 1.877
"""


def test_cells_and_numbers():
    assert [cell_class(x) for x in ["1,7", "1001", "12.430", "(7.812)", "06.05.2001", "BE", "Nein", "-", "12%", "Mainz"]] == \
        ["dec", "num", "num", "num", "date", "code", "yesno", "dash", "pct", None]
    assert to_number("2,3") == 2.3 and to_number("12.430") == 12430 and to_number("(7.812)") == -7812
    assert to_number("1,234") == 1234 and to_number("12%") == 12


def test_pages_merge_and_the_translation_is_a_copy():
    (t,) = find_tables(TRANSCRIPT, "D1", "Results")
    assert [r["name"] for r in t.rows] == ["Databases", "Statistics", "Compilers", "Operating Systems",
                                          "Networks", "Robotics"]
    assert t.rows[0]["aliases"] == ["Datenbanken"] and t.rows[2]["cells"] == ["1003", "A"]
    assert t.legend == {"P": "passed / bestanden", "A": "admitted / zugelassen"}


def test_answers_come_from_the_table_not_the_model():
    (t,) = find_tables(TRANSCRIPT, "D1", "Results")
    t.columns, t.name_column = ["exam_number", "status", "credits", "grade"], "course"
    facts = tables.facts("Which courses are admitted but not yet passed, and how many credits so far?", [t])
    s = facts[-1]
    assert "P (passed / bestanden) ×4" in s and "A (admitted / zugelassen) ×2 = Compilers, Robotics" in s
    assert "credits: 4 rows have a value, total 20" in s
    assert "grade: 4 rows have a value" in s and "lowest 1,0 (Networks), highest 3,0 (Operating Systems)" in s
    assert "no grade for 2: Compilers, Robotics" in s
    assert "exam_number" not in s                      # ids are not added up
    (row, _summary) = tables.facts("What grade did they get in Statistik?", [t])   # found by its alias
    assert "Statistics: exam_number 1002; status P (passed / bestanden); credits 5; grade 2,3" in row
    (row, _summary) = tables.facts("What grade in Compilers?", [t])
    assert "no value for credits, grade" in row
    assert tables.facts("How old is he?", [t]) == []


def test_any_table_with_the_same_shape_works():
    (t,) = find_tables(FINANCIALS, "D2", "Annual report")
    assert t.rows[1] == {"name": "Herstellungskosten", "cells": ["(7.812)", "(7.390)"], "aliases": []}
    t.columns, t.name_column = ["fy2024", "fy2023"], "line item"
    s = tables.facts("Welcher Posten war 2024 am höchsten?", [t])[-1]
    assert "fy2024: 4 rows have a value" in s and "highest 12.430 (Umsatzerlöse)" in s
    assert "lowest (7.812) (Herstellungskosten)" in s


def test_column_names_from_the_model_are_checked():
    (t,) = find_tables(TRANSCRIPT, "D1", "Results")
    tables.name_columns(t, chat=lambda p: '{"name_column": "Course", "columns": ["Exam Nr", "status", "credits", "grade"]}')
    assert t.columns == ["exam_nr", "status", "credits", "grade"] and t.name_column == "course"
    tables.name_columns(t, chat=lambda p: '{"columns": ["only one"]}')          # wrong count
    assert t.columns == ["num_1", "code_2", "num_3", "dec_4"]
    tables.name_columns(t, chat=lambda p: "not json")
    assert t.columns == ["num_1", "code_2", "num_3", "dec_4"]


def test_prose_is_not_a_table():
    text = "Walid lives in Mainz since 2024.\nHe likes football 3 times a week.\nPhone 0151 2345 678\n"
    assert find_tables(text) == []


def test_unique_numbers_are_ids_even_without_a_name():
    (t,) = find_tables(TRANSCRIPT, "D1", "Results")
    t.columns = ["num_1", "code_2", "num_3", "dec_4"]                  # the model gave no names
    s = tables.summarize(t)
    assert "num_1" not in s and "num_3: 4 rows have a value, total 20" in s
