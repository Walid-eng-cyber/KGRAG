"""The answer step's tools and skills (pure Python, no models or databases)."""
from __future__ import annotations

from datetime import date

from src.skills import skill_for
from src.tools import TOOLS, run_tools
from src.tools.dates import all_periods, span

TODAY = date(2026, 11, 12)

# How the CV reads after text extraction (one passage, whitespace collapsed).
CV = ("Room Booking Hochschule Mainz | GitHub 15.03.2026 – 24.06.2026 ● Konzeption eines KI-Assistenten. "
      "Wissenschaftliche Hilfskraft (10 Std./Woche) i3mainz, Hochschule Mainz | Projekt VAMoS: KI in Produktions Supply Chain ab 02/2027 (Zusage) Mainz "
      "Mitarbeiter im Rotationssystem als Werkstudent (Gastronomie) Jamys Burger, Mainz. seit 15.02.2026 Mainz "
      "Mitarbeiter im Rotationssystem als Werkstudent (Gastronomie) McDonalds 02.11.2024 – 12.09.2025 Mainz "
      "Praktikum als Softwareentwickler (40 Std./Woche) General Treasury of the Kingdom (TGR) "
      "19.06.2023 – 18.07.2023 Rabat, Marokko ● Entity Matching. "
      "Praktikum als Softwareentwickler (40 Std./Woche) Association of African Fisheries "
      "01.07.2022 – 30.09.2022 Safi, Marokko. "
      "M.Sc. Angewandte Informatik Hochschule Mainz 01.09.2024 – AKTUELL Mainz, Deutschland. "
      "B.Sc. Mathematik und Informatik University Cadi Ayyad 01.09.2019 – 01.06.2022 Safi, Marokko")
EV = [("P1", CV)]


def ask(q, tools=None, graph=None, ev=EV):
    return run_tools(q, ev, graph, tools or list(TOOLS), TODAY)


def test_periods_are_read_with_their_labels():
    ps = {p.label: p for p in all_periods(EV, TODAY)}
    bsc = next(p for label, p in ps.items() if "B.Sc." in label)
    assert (bsc.start, bsc.end, bsc.ongoing) == (date(2019, 9, 1), date(2022, 6, 1), False)
    msc = next(p for label, p in ps.items() if "M.Sc." in label)
    assert msc.ongoing and msc.end == TODAY
    jamys = next(p for label, p in ps.items() if "Jamys" in label)
    assert jamys.ongoing and jamys.start == date(2026, 2, 15)
    vamos = next(p for label, p in ps.items() if "VAMoS" in label)
    assert vamos.planned and vamos.start == date(2027, 2, 1)
    assert span(date(2019, 9, 1), date(2022, 6, 1)).startswith("2 years 9 months")


def test_duration_picks_the_period_the_question_is_about():
    facts, used = ask("how long did walid study his bachelor?")
    assert used == ["duration"] and "B.Sc." in facts[0][0] and "2 years 9 months" in facts[0][0]
    facts, _ = ask("how long was the internship at the General Treasury?")
    assert "TGR" in facts[0][0] and "1 month" in facts[0][0]
    facts, _ = ask("wie lange arbeitet er schon bei Jamys Burger?")
    assert "Jamys" in facts[0][0] and "ongoing" in facts[0][0] and "8 months" in facts[0][0]


def test_timeline_answers_current_latest_first_and_order():
    facts = [f for f, _ in ask("what is walid's current job?", ["timeline"])[0]]
    assert "Jamys" in facts[0] and "M.Sc." not in facts[0]          # a job question skips the degree
    now = [f for f, _ in ask("what is walid doing at the moment?", ["timeline"])[0]]
    assert "M.Sc." in now[0] and "Jamys" in now[0] and "VAMoS" in now[1] and now[1].startswith("Planned")
    (latest, _), = ask("what was his most recent project?", ["timeline"])[0]
    assert "Room Booking" in latest
    (intern, _), = ask("what was walid's first internship?", ["timeline"])[0]
    assert "Fisheries" in intern and "McDonalds" not in intern
    (first, _), = ask("what did he do first?", ["timeline"])[0]
    assert "B.Sc." in first
    (order, _), = ask("show his career timeline", ["timeline"])[0]
    assert order.index("B.Sc.") < order.index("TGR") < order.index("Room Booking") < order.index("VAMoS")


def test_experience_unions_work_periods_and_skips_study_and_plans():
    (fact, _), = ask("how many years of experience does walid have?", ["experience"])[0]
    for job in ("McDonalds", "TGR", "Fisheries", "Jamys"):
        assert job in fact
    assert "B.Sc." not in fact and "VAMoS" not in fact and "Room Booking" not in fact
    # 92 + 30 + 315 + 271 days
    assert "(708 days)" in fact and "1 year 11 months" in fact


def test_aggregate_does_the_arithmetic_on_graph_rows():
    g = {"rows": [{"insider": "A", "value": 1000.0, "shares": 10}, {"insider": "B", "value": 5000.5, "shares": 40},
                  {"insider": "C", "value": 250.0, "shares": 5}]}
    facts, used = ask("what is the total and the largest trade value?", ["aggregate"], g)
    assert used == ["aggregate"]
    assert "value: total 6,250, highest 5,000 (B)" in facts[0][0]
    assert ask("who traded?", ["aggregate"], g) == ([], [])          # no arithmetic asked
    assert ask("total?", ["aggregate"], {"rows": [{"value": 3}]}) == ([], [])   # one row: nothing to add


def test_tools_only_run_when_triggered_and_allowed():
    assert ask("where does walid live?") == ([], [])
    assert ask("how old is walid?", ["duration"]) == ([], [])          # age tool not in this skill


def test_a_failing_tool_never_breaks_the_answer(monkeypatch):
    import src.tools as tools
    boom = tools.Tool("boom", "x", tools._rx("boom"), lambda *a: 1 / 0)
    monkeypatch.setitem(tools.TOOLS, "boom", boom)
    assert run_tools("boom", EV, None, ["boom"], TODAY) == ([], [])


def test_skills_follow_the_corpus():
    assert "experience" in skill_for({"kind": "documents", "schema": {"preset": "career"}})["tools"]
    assert skill_for({"kind": "sec_insiders", "schema": {}})["label"] == "SEC insider trades"
    assert set(skill_for({"kind": "documents", "schema": {"preset": "custom"}})["tools"]) == set(TOOLS) - {"in_year"}
    assert skill_for(None)["label"] == "Company filings"


def test_next_planned_and_year_checks():
    (nxt, _), = ask("what position will walid start next?", ["timeline"])[0]
    assert nxt.startswith("Next, planned") and "VAMoS" in nxt
    (y, _), = ask("which company did walid work for in 2018?", ["in_year"])[0]
    assert "No dated period" in y and "2018" in y
    (y, _), = ask("what was he doing in 2023?", ["in_year"])[0]
    assert "TGR" in y and "B.Sc." not in y


def test_degree_words_beat_names_that_contain_them():
    ev = [("P1", "Draft Master AI Hochschule Mainz | GitHub 01.03.2025 – 01.07.2025 ● Chat app. "
                 "STUDIUM M.Sc. Angewandte Informatik Hochschule Mainz 01.09.2024 – AKTUELL Mainz.")]
    (fact, _), = run_tools("How long has he been studying for his master's?", ev, None, ["duration"], TODAY)[0]
    assert "M.Sc." in fact and "Draft Master" not in fact
