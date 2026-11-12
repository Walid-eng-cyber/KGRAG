"""SEC Form 3/4/5 loader, verbalizer and graph-view tests (no database or SEC download)."""
from __future__ import annotations

from server.graph_view import build_graph
from src.answer import verbalize_graph
from src.form4 import build_records, company_aliases, is_10b5_1, is_entity, parse_date, person_aliases


def sub(acc, cik="1045810", name="NVIDIA CORP", ticker="NVDA", form="4", filed="05-AUG-2026", plan="1"):
    return {"ACCESSION_NUMBER": acc, "FILING_DATE": filed, "DOCUMENT_TYPE": form, "ISSUERCIK": cik,
            "ISSUERNAME": name, "ISSUERTRADINGSYMBOL": ticker, "AFF10B5ONE": plan}


def owner(acc, cik, name, rel="Officer", title=""):
    return {"ACCESSION_NUMBER": acc, "RPTOWNERCIK": cik, "RPTOWNERNAME": name,
            "RPTOWNER_RELATIONSHIP": rel, "RPTOWNER_TITLE": title}


def trade(acc, sk, code="S", shares="1,000", price="180.50", ad="D", date="01-AUG-2026"):
    return {"ACCESSION_NUMBER": acc, "NONDERIV_TRANS_SK": sk, "SECURITY_TITLE": "Common Stock",
            "TRANS_DATE": date, "TRANS_CODE": code, "TRANS_SHARES": shares, "TRANS_PRICEPERSHARE": price,
            "TRANS_ACQUIRED_DISP_CD": ad, "SHRS_OWND_FOLWNG_TRANS": "50000"}


TABLES = {
    "SUBMISSION.tsv": [sub("A1"), sub("A2", form="4/A"), sub("A3", plan="0", filed="20-SEP-2026"),
                       sub("B1", cik="1318605", name="TESLA, INC.", ticker="tsla", plan="")],
    "REPORTINGOWNER.tsv": [
        owner("A1", "0001197649", "HUANG JEN HSUN", "Director,Officer", "President and CEO"),
        owner("A2", "0001197649", "HUANG JEN HSUN"),
        owner("A3", "0001197649", "HUANG JEN HSUN", "Director"),
        owner("B1", "0000102909", "VANGUARD GROUP INC", "TenPercentOwner"),
        owner("B1", "0000999999", "SMITH JOHN A JR", "Director"),
    ],
    "NONDERIV_TRANS.tsv": [trade("A1", "1"), trade("A1", "2", code="F", price="0"),
                           trade("A2", "1"), trade("B1", "1", code="P", ad="A", price="250")],
}


def test_parsing_helpers():
    assert parse_date("28-SEP-2026") == "2026-09-28" and parse_date("2026-09-28") == "2026-09-28"
    assert parse_date("not a date") is None
    assert is_entity("VANGUARD GROUP INC") and is_entity("Blackrock Fund Advisors") and not is_entity("HUANG JEN HSUN")
    assert person_aliases("HUANG JEN HSUN") == ["Jen Hsun Huang", "Jen Huang"]
    assert "John Smith Jr" in person_aliases("SMITH JOHN A JR")
    assert company_aliases("NVIDIA CORP", "nvda") == ["NVDA", "Nvidia"]
    assert company_aliases("TESLA, INC.", "TSLA") == ["TSLA", "Tesla"]
    assert (is_10b5_1("1"), is_10b5_1("false"), is_10b5_1("")) == (True, False, None)


def test_build_records_skips_amendments_and_merges_roles():
    r = build_records(TABLES)
    assert {i["ticker"] for i in r["issuers"]} == {"NVDA", "TSLA"}
    kinds = {i["name"]: i["kind"] for i in r["insiders"]}
    assert kinds == {"HUANG JEN HSUN": "PERSON", "VANGUARD GROUP INC": "COMPANY", "SMITH JOHN A JR": "PERSON"}
    huang = next(x for x in r["roles"] if x["insider"] == "1197649")
    assert huang["roles"] == ["Director", "Officer"] and huang["title"] == "President and CEO"
    assert huang["last_filed"] == "2026-09-20"
    ids = {t["id"] for t in r["trades"]}
    assert ids == {"A1#1", "A1#2", "B1#1"}  # the 4/A amendment (A2) is skipped
    assert len(build_records(TABLES, include_amendments=True)["trades"]) == 4


def test_trade_fields_and_joint_filings():
    t = {x["id"]: x for x in build_records(TABLES)["trades"]}
    sale = t["A1#1"]
    assert (sale["code_label"], sale["shares"], sale["price"], sale["value"]) == ("open-market sale", 1000.0, 180.5, 180500.0)
    assert sale["date"] == "2026-08-01" and sale["plan_10b5_1"] is True and not sale["acquired"]
    assert t["A1#2"]["value"] is None  # tax withholding at $0 has no value
    joint = t["B1#1"]
    assert joint["insider"] == "102909" and joint["co_filers"] == ["SMITH JOHN A JR"]  # counted once
    assert joint["acquired"] and joint["plan_10b5_1"] is None


def g(qt, rows, **params):
    return {"query_type": qt, "rows": rows, "params": params, "cypher": "MATCH ..."}


TRADES = [{"_insider_type": "PERSON", "insider": "HUANG JEN HSUN", "date": "2026-08-01", "type": "open-market sale",
           "shares": 1000.0, "price": 180.5, "value": 180500.0, "plan_10b5_1": True, "_src": "0001-26-000001"},
          {"_insider_type": "COMPANY", "insider": "VANGUARD GROUP INC", "date": "2026-07-01", "type": "open-market purchase",
           "shares": 10.0, "price": None, "value": None, "plan_10b5_1": False, "_src": "0001-26-000002"}]


def test_trades_are_verbalized_one_citable_fact_per_row():
    out = verbalize_graph(g("insider_trades_at", TRADES, name="NVIDIA CORP"))
    assert len(out) == 2
    assert out[0] == ("On 2026-08-01, HUANG JEN HSUN reported an open-market sale of 1,000 shares of NVIDIA CORP "
                      "at $180.50 (about $180,500) under a 10b5-1 trading plan. [SEC filing 0001-26-000001]")
    assert "at $" not in out[1] and "10b5-1" not in out[1]


def test_net_activity_statements_and_metrics():
    row = {"shares_bought": 100.0, "value_bought": 25000.0, "shares_sold": 3000.0, "value_sold": 540000.0,
           "buyers": 1, "sellers": 2, "first_date": "2026-07-01", "last_date": "2026-09-28"}
    out = verbalize_graph(g("insider_net_activity", [row], name="TESLA, INC."))
    assert len(out) == 2 and "insiders were sellers of about $515,000" in out[1]
    view = build_graph(g("insider_net_activity", [row], name="TESLA, INC."))
    assert view["layout"] == "none" and [m["tag"] for m in view["metrics"]] == ["G1", "G2"]
    none = verbalize_graph(g("insider_net_activity", [{"shares_bought": 0, "shares_sold": 0}], name="X"))
    assert "No open-market" in none[0]


def test_graph_view_types_insiders_and_tags_by_row():
    view = build_graph(g("insider_trades_at", TRADES, name="NVIDIA CORP"))
    types = {n["label"]: n["type"] for n in view["nodes"]}
    assert types == {"HUANG JEN HSUN": "PERSON", "VANGUARD GROUP INC": "COMPANY", "NVIDIA CORP": "COMPANY"}
    assert [e["tag"] for e in view["edges"]] == ["G1", "G2"] and view["layout"] == "bipartite"


def test_shared_insiders_two_hop_edges_share_the_row_tag():
    rows = [{"company": "ARISTA NETWORKS", "shared_insiders": ["COXE TENCH"]},
            {"company": "OTHER CO", "shared_insiders": ["A PERSON", "B PERSON"]}]
    view = build_graph(g("companies_sharing_insiders", rows, name="NVIDIA CORP"))
    assert [e["tag"] for e in view["edges"]] == ["G1", "G1", "G2", "G2", "G2", "G2"]
    assert all(e["relation"] == "INSIDER_OF" for e in view["edges"])
    stmts = verbalize_graph(g("companies_sharing_insiders", rows, name="NVIDIA CORP"))
    assert stmts[0] == "ARISTA NETWORKS shares these insiders with NVIDIA CORP: COXE TENCH."


def test_insiders_of_is_one_statement():
    rows = [{"_insider_type": "PERSON", "insider": "KRESS COLETTE", "roles": ["Officer"], "title": "EVP & CFO"},
            {"_insider_type": "PERSON", "insider": "COXE TENCH", "roles": ["Director"], "title": None}]
    assert verbalize_graph(g("insiders_of", rows, name="NVIDIA CORP")) == [
        "NVIDIA CORP's reported insiders include: KRESS COLETTE (EVP & CFO), COXE TENCH (Director)."]
    assert {e["tag"] for e in build_graph(g("insiders_of", rows, name="NVIDIA CORP"))["edges"]} == {"G1"}


def test_router_sends_insider_questions_to_the_graph():
    from src.router import Path, _heuristic
    for q in ["Which NVIDIA insiders sold shares?", "Did Jensen Huang sell stock?",
              "What were the largest insider purchases?", "Which companies share a director with Apple?"]:
        assert _heuristic(q) == Path.GRAPH, q


def test_typo_prices_are_flagged_suspect():
    from src.form4 import flag_suspect_prices
    t = [{"issuer": "1", "price": p, "value": p * 100} for p in (100.0, 101.0, 99.0, 2_110_482.0)]
    t.append({"issuer": "2", "price": 40e6, "value": 40e6 * 40e6})   # absurd total value
    t.append({"issuer": "3", "price": None, "value": None})
    t.append({"issuer": "4", "price": 52316.7, "value": 52316.7 * 34800})   # no baseline, absurd per-share price
    flag_suspect_prices(t)
    assert [x["price_suspect"] for x in t] == [False, False, False, True, True, False, True]
