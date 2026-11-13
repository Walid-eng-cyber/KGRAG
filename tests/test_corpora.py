"""Corpus schemas, profiles and the corpus-scoped graph helpers (no database needed)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server.main as main
from server.corpora import CorpusStore
from server.graph_view import build_graph
from src import corpus_graph, corpus_schema
from src.answer import verbalize_graph


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "corpus_store", CorpusStore(tmp_path / "documents.db"))
    return TestClient(main.app)


def test_presets_are_valid_schemas():
    for key in corpus_schema.PRESETS:
        s = corpus_schema.from_preset(key)
        assert s["preset"] == key and s["entities"] and s["relations"]
        assert all(r[0] in s["entities"] and r[2] in s["entities"] for r in s["relations"])


def test_custom_schema_is_normalized_and_checked():
    s = corpus_schema.validate(["person", "Skill set", "skill set"], [["person", "has skill", "skill set"]])
    assert s == {"entities": ["PERSON", "SKILL_SET"], "relations": [["PERSON", "HAS_SKILL", "SKILL_SET"]]}
    with pytest.raises(corpus_schema.SchemaError, match="isn't in the entity list"):
        corpus_schema.validate(["PERSON"], [["PERSON", "WORKS_AT", "COMPANY"]])
    with pytest.raises(corpus_schema.SchemaError, match="at least one relation"):
        corpus_schema.validate(["PERSON"], [])
    with pytest.raises(corpus_schema.SchemaError, match="valid relation"):
        corpus_schema.validate(["A"], [["A", "MENTIONS", "A"]])   # reserved for chunk links
    with pytest.raises(corpus_schema.SchemaError, match="valid type"):
        corpus_schema.validate(["9LIVES"], [["9LIVES", "X", "9LIVES"]])


def test_create_corpus_with_preset_or_custom_schema(client):
    r = client.post("/api/corpora", json={"name": "Walid", "description": "My CV", "preset": "career"})
    c = r.json()["corpus"]
    assert c["kind"] == "documents" and c["schema"]["preset"] == "career" and "SKILL" in c["schema"]["entities"]
    custom = client.post("/api/corpora", json={"name": "Recipes", "entities": ["dish", "ingredient"],
                                               "relations": [["dish", "uses", "ingredient"]]}).json()["corpus"]
    assert custom["schema"] == {"preset": "custom", "entities": ["DISH", "INGREDIENT"],
                                "relations": [["DISH", "USES", "INGREDIENT"]]}
    bad = client.post("/api/corpora", json={"name": "Bad", "entities": ["A"], "relations": [["A", "R", "B"]]})
    assert bad.status_code == 422 and "entity list" in bad.json()["detail"]
    presets = client.get("/api/corpora").json()["presets"]
    assert set(presets) == {"company", "career", "general"}


def test_update_corpus_schema_and_name(client):
    cid = client.post("/api/corpora", json={"name": "X", "preset": "general"}).json()["corpus"]["id"]
    r = client.patch(f"/api/corpora/{cid}", json={"name": "Y", "entities": ["PERSON", "PET"],
                                                   "relations": [["PERSON", "OWNS", "PET"]]})
    c = r.json()["corpus"]
    assert c["name"] == "Y" and c["schema"]["entities"] == ["PERSON", "PET"] and c["schema"]["preset"] == "custom"


def test_sec_corpus_profile_uses_insider_examples(client, monkeypatch):
    c = main.corpus_store.create("SEC", "", "sec_insiders", {"preset": "sec_insiders", "entities": ["COMPANY"],
                                                             "relations": [["COMPANY", "X", "COMPANY"]]}, None)
    p = client.get(f"/api/corpora/{c['id']}/profile").json()
    assert p["examples"] and "insider" in p["examples"][0]["question"].lower()
    assert client.post(f"/api/corpora/{c['id']}/documents", files={"file": ("a.txt", b"x" * 300)}).status_code == 400


def test_document_corpus_profile_suggests_from_extracted_entities(client, monkeypatch):
    cid = client.post("/api/corpora", json={"name": "W", "preset": "career"}).json()["corpus"]["id"]
    monkeypatch.setattr(corpus_graph, "profile", lambda c: {
        "types": [{"type": "SKILL", "n": 9}, {"type": "INSTITUTION", "n": 2}],
        "top_entities": [{"name": "Walid Beladdad", "type": "PERSON", "degree": 12},
                         {"name": "Hochschule Fulda", "type": "INSTITUTION", "degree": 3}],
        "relations": [{"relation": "HAS_SKILL", "n": 9}]})
    qs = [e["question"] for e in client.get(f"/api/corpora/{cid}/profile").json()["examples"]]
    assert "What is Walid Beladdad connected to?" in qs
    assert "How is Walid Beladdad related to Hochschule Fulda?" in qs
    assert "Which skill entries are in these documents?" in qs


class _Ent:
    def __init__(self, name, label):
        self.name, self.label = name, label


class _Rel:
    def __init__(self, s, label, t):
        self.source_id, self.label, self.target_id = s, label, t


class _Node:
    def __init__(self, ents, rels):
        self.metadata = {"nodes": ents, "relations": rels}


def test_triples_from_keeps_only_safe_well_formed_triples():
    ents, rels = corpus_graph.triples_from(_Node(
        [_Ent("Walid", "PERSON"), _Ent("Python", "SKILL"), _Ent("x", "bad label; DROP")],
        [_Rel("Walid", "HAS_SKILL", "Python"), _Rel("Walid", "HAS_SKILL", "Unknown"),
         _Rel("Walid", "REL`) DETACH DELETE n //", "Python")]))
    assert ents == [{"name": "Walid", "label": "PERSON"}, {"name": "Python", "label": "SKILL"}]
    assert rels == [{"source": "Walid", "target": "Python", "type": "HAS_SKILL"}]


def test_corpus_results_verbalize_and_draw_with_their_own_types():
    g = {"source": "corpus", "query_type": "entity_connections", "params": {"name": "Walid"}, "cypher": "M",
         "rows": [{"entity": "Walid", "_e_type": "PERSON", "relation": "HAS_SKILL", "outgoing": True,
                   "other": "Python", "_o_type": "SKILL", "_src": "c1"},
                  {"entity": "Walid", "_e_type": "PERSON", "relation": "STUDIED_AT", "outgoing": True,
                   "other": "Hochschule Fulda", "_o_type": "INSTITUTION", "_src": "c2"}]}
    assert verbalize_graph(g) == ["Walid has skill Python.", "Walid studied at Hochschule Fulda."]
    view = build_graph(g)
    assert {n["label"]: n["type"] for n in view["nodes"]} == {"Walid": "PERSON", "Python": "SKILL",
                                                              "Hochschule Fulda": "INSTITUTION"}
    assert [e["tag"] for e in view["edges"]] == ["G1", "G2"] and view["layout"] == "radial"
    count = {"source": "corpus", "query_type": "count_of_type", "params": {"type": "SKILL"},
             "rows": [{"type": "SKILL", "count": 2, "names": ["Neo4j", "Python"]}]}
    assert verbalize_graph(count) == ["This corpus has exactly 2 skill entities: Neo4j; Python."]
    assert build_graph(count)["metrics"] == [{"label": "Skill", "value": 2, "tag": "G1"}]


def test_legacy_path_results_are_not_treated_as_corpus_results():
    legacy = {"query_type": "path_between", "params": {"a": "A", "b": "B"},
              "rows": [{"path": ["A", "X", "B"], "relations": ["FACES_RISK", "FACES_RISK"]}]}
    assert corpus_graph.edges(legacy) is None and corpus_graph.verbalize(legacy) is None


# ---------- entity resolution within a corpus ----------

def _merged_names(nodes):
    by = {n["id"]: n["name"] for n in nodes}
    return sorted(sorted([by[k]] + [by[d] for d in dups]) for k, dups in corpus_graph.plan_merges(nodes))


def test_plan_merges_on_real_cv_names():
    n = lambda i, name, t: {"id": str(i), "name": name, "type": t}   # noqa: E731
    nodes = [
        n(1, "Walid", "PERSON"), n(2, "Walid Beladdad", "PERSON"), n(3, "walid Beladdad", "PERSON"), n(4, "walid", "PERSON"),
        n(5, "Prof. Dr.-Ing. Jens Strackeljan", "PERSON"),
        n(6, "NBA-Bewertungsvorhersage", "PROJECT"), n(7, "NBA-Bewertungsvorhersage – ML-Projekt", "PROJECT"),
        n(8, "Knowledge-Graph RAG (GraphRAG)", "PROJECT"), n(9, "Knowledge-Graph RAG (GraphRAG) for Corporate Documents", "PROJECT"),
        n(10, "Room Booking", "PROJECT"), n(11, "Draft Master AI", "PROJECT"),
        n(12, "General Treasury of the Kingdom", "ORGANIZATION"), n(13, "General Treasury of the Kingdom (TGR)", "ORGANIZATION"),
        n(14, "Hochschule Mainz", "ORGANIZATION"), n(15, "HOCHSCHULE MAINZ", "ORGANIZATION"),
        n(16, "i3mainz, Hochschule Mainz", "ORGANIZATION"),        # a different unit: must NOT merge
        n(17, "Hochschule Mainz", "INSTITUTION"),                 # different type: must NOT merge
    ]
    assert _merged_names(nodes) == [
        ["General Treasury of the Kingdom", "General Treasury of the Kingdom (TGR)"],
        ["HOCHSCHULE MAINZ", "Hochschule Mainz"],
        ["Knowledge-Graph RAG (GraphRAG)", "Knowledge-Graph RAG (GraphRAG) for Corporate Documents"],
        ["NBA-Bewertungsvorhersage", "NBA-Bewertungsvorhersage – ML-Projekt"],
        ["Walid", "Walid Beladdad", "walid", "walid Beladdad"],
    ]
    keep = {nodes[int(k) - 1]["name"] for k, _ in corpus_graph.plan_merges(nodes)}
    assert {"Walid Beladdad", "Hochschule Mainz", "NBA-Bewertungsvorhersage – ML-Projekt"} <= keep


def test_first_name_is_not_merged_when_ambiguous():
    nodes = [{"id": "1", "name": "Anna", "type": "PERSON"}, {"id": "2", "name": "Anna Schmidt", "type": "PERSON"},
             {"id": "3", "name": "Anna Weber", "type": "PERSON"}]
    assert corpus_graph.plan_merges(nodes) == []


def test_pick_type_votes_with_the_schema():
    schema = corpus_schema.from_preset("career")
    # "Hochschule Mainz" was tagged PROJECT and ORGANIZATION; its links say organization
    incident = [("in", "WORKED_ON", "PERSON"), ("in", "WORKED_AT", "PERSON"), ("in", "WORKED_AT", "PERSON")]
    assert corpus_graph.pick_type(["PROJECT", "ORGANIZATION"], incident, schema) == "ORGANIZATION"


def test_type_and_relation_are_read_from_the_question():
    career = corpus_schema.from_preset("career")
    rels = sorted({r[1] for r in career["relations"]})
    assert corpus_graph.guess_type("how many project did walid work at ?", career["entities"]) == "PROJECT"
    assert corpus_graph.guess_type("in which hochschule did walid study?", career["entities"]) == "INSTITUTION"
    assert corpus_graph.guess_type("which programming skills does he have", career["entities"]) == "SKILL"
    assert corpus_graph.guess_relation("where did he studied at", rels) == "STUDIED_AT"


def test_linked_entities_are_stated_exactly():
    g = {"source": "corpus", "query_type": "entities_linked_of_type", "params": {"name": "Walid Beladdad", "type": "PROJECT"},
         "rows": [{"entity": "Walid Beladdad", "_e_type": "PERSON", "name": p, "type": "PROJECT", "relations": ["WORKED_ON"]}
                  for p in ["Draft Master AI", "Room Booking"]]}
    assert verbalize_graph(g) == ["Walid Beladdad is linked to exactly 2 project entities in this corpus: "
                                  "Draft Master AI; Room Booking."]
    view = build_graph(g)
    assert view["layout"] == "radial" and {e["tag"] for e in view["edges"]} == {"G1"}


def test_people_named_in_the_question_are_found_without_the_planner():
    people = ["Walid Beladdad", "Prof. Dr.-Ing. Jens Strackeljan", "Anna Schmidt", "Anna Weber"]
    assert corpus_graph.mentioned_people("how many project did walid work at ?", people) == ["Walid Beladdad"]
    assert corpus_graph.mentioned_people("what did Walid Beladdad study", people) == ["Walid Beladdad"]
    assert corpus_graph.mentioned_people("which courses did anna take", people) == []      # ambiguous
    assert corpus_graph.mentioned_people("how many projects are there", people) == []


# ---------- facts computed in code ----------

def test_age_is_computed_from_a_birth_date():
    from datetime import date
    from src.tools import run_tools
    from src.tools.dates import parse_date
    derived_facts = lambda q, ev, d: run_tools(q, ev, None, ["age"], d)[0]   # noqa: E731
    today = date(2026, 11, 12)
    for text in ("geboren am 06.05.2001", "Geburtsdatum: 06.05.2001", "born on May 6, 2001", "Born: 6 Mai 2001"):
        facts = derived_facts("how old is walid ?", [("P1", f"Walid Beladdad, {text}, Mainz")], today)
        assert facts and "25 years old" in facts[0][0] and facts[0][1] == "P1", text
    assert "24 years old" in derived_facts("wie alt ist er", [("P2", "geboren am 13.11.2001")], today)[0][0]
    assert derived_facts("where does walid live", [("P1", "geboren am 06.05.2001")], today) == []
    assert parse_date("31.02.2001") is None


def test_retry_narration_is_never_shown():
    from src.answer import _META
    assert _META.search("Since there is no 'G3', I will keep the original text, and cite it with a valid label.")
    assert not _META.search("Walid studied at Hochschule Mainz [P1].")


def test_keyword_query_and_rank_fusion():
    from src.vectorize import _kw_query, fuse
    q = _kw_query("What was the title of Walid's bachelor thesis?")
    assert q.startswith("titl:* | walid:* | bachelor:*") and "bachelorarbeit:*" in q      # stems + synonyms
    assert "bachelor:*" in _kw_query("Wo hat er seinen Bachelor gemacht?")
    assert _kw_query("is it?") is None
    a, b, c = ({"chunk_id": x} for x in "abc")
    # c is 2nd in both lists, a is 1st in one only: c wins
    assert [r["chunk_id"] for r in fuse([[a, c], [b, c]], 3)][0] == "c"


def test_bm25_ignores_words_every_chunk_has():
    from src.vectorize import bm25
    docs = [{"text": "Walid Beladdad grades: Machine Learning 4,0"}, {"text": "Walid Beladdad lives in Mainz"},
            {"text": "Walid Beladdad Bachelorarbeit: Neo4Py"}]
    s = bm25("walid thesis", docs, 3, {"walid": 3, "thesis": 0, "bachelorarbeit": 1, "arbeit": 1,
                                         "masterarbeit": 0})
    assert s.index(max(s)) == 2 and max(s[0], s[1]) < s[2] / 5    # the name alone barely counts


def test_type_labels_are_not_entities():
    assert corpus_graph.is_type_label("Persönliches Projekt", "PROJECT")
    assert corpus_graph.is_type_label("Side project", "PROJECT") and corpus_graph.is_type_label("Projekt", "PROJECT")
    assert not corpus_graph.is_type_label("Room Booking", "PROJECT")
    assert not corpus_graph.is_type_label("Project Apollo", "PROJECT")
    assert not corpus_graph.is_type_label("NBA-Bewertungsvorhersage – ML-Projekt", "PROJECT")
    assert not corpus_graph.is_type_label("Persönliches Projekt", "SKILL")


def test_long_passages_show_the_part_that_matches_the_question():
    from src.answer import best_window
    text = "Intro " * 300 + "Mitautoren: Mahmoud Eslami, Ali Sayyadi. " + "Outro " * 100
    w = best_window(text, "Who are the co-authors (Mitautoren) of the publication?", 300)
    assert "Eslami" in w and w.startswith("…") and len(w) <= 302
    assert best_window("short", "anything", 300) == "short"


def test_near_duplicate_chunks_take_one_slot():
    from src.vectorize import diversify
    cv = "Walid Beladdad Masterstudent Angewandte Informatik Projekte Room Booking Docker React TypeScript"
    rows = [{"chunk_id": "a", "text": cv}, {"chunk_id": "b", "text": cv + " Seite"},
            {"chunk_id": "c", "text": "Sprachen Arabisch Muttersprache Deutsch C1 Englisch Französisch"}]
    assert [r["chunk_id"] for r in diversify(rows, 2)] == ["a", "c"]


def test_language_is_detected_from_function_words():
    from src.vectorize import language
    assert language("What did he achieve in the competition and with which team?") == "en"
    assert language("Was hat er im Turnen erreicht und mit der Mannschaft gewonnen?") == "de"
    assert language("Quel est le diplôme de la personne et où est-il?") == "fr"
    assert language("Python FastAPI Neo4j") is None


def test_stems_let_prefixes_match_other_word_forms():
    from src.vectorize import stem
    assert stem("turnens") == "turn" and stem("deutschen") == "deutsch" and stem("projects") == "project"
    assert stem("python") == "python" and stem("ects") == "ects"
