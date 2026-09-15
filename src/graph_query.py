"""Safe, parameterized graph queries — the hardened graph path.

Security control: the LLM NEVER emits Cypher. Every query is a fixed,
hand-written template in TEMPLATES below. The model only (a) picks a template by
key and (b) names the entities/values to fill in. Those values are:
  1. resolved to real node ids/names against the graph (using the `aliases` we
     built in entity resolution — so "Microsoft" -> "MICROSOFT CORP"), and
  2. passed to Neo4j as **bound parameters** ($name, $names, $risk), never
     string-concatenated into the query.

This blocks Cypher injection (the model can't smuggle a write/DROP into the
query — the Cypher is a constant) and makes results reproducible: the same
question resolves to the same template + params + rows every time.
"""
from __future__ import annotations

import difflib
import json

import ollama
from neo4j import GraphDatabase

import config

# --- The template library. Cypher is CONSTANT; only $params vary. -----------
# Each entry: cypher + how to fill it (entity_slots / list_slot / value_slot).
TEMPLATES: dict[str, dict] = {
    "subsidiaries_of": {
        "cypher": "MATCH (c:COMPANY {name:$name})-[:HAS_SUBSIDIARY]->(s) "
                  "RETURN s.name AS subsidiary",
        "entity_slots": ["name"],
        "desc": "list the subsidiaries of a company",
    },
    "count_subsidiaries": {
        "cypher": "MATCH (c:COMPANY {name:$name})-[:HAS_SUBSIDIARY]->(s) "
                  "RETURN count(s) AS count",
        "entity_slots": ["name"],
        "desc": "how many subsidiaries a company has",
    },
    "risks_of": {
        "cypher": "MATCH (c {name:$name})-[:FACES_RISK]->(r) RETURN r.name AS risk",
        "entity_slots": ["name"],
        "desc": "the risks a company faces",
    },
    "executives_of": {
        "cypher": "MATCH (c {name:$name})-[:HAS_EXECUTIVE]->(p) "
                  "RETURN p.name AS executive",
        "entity_slots": ["name"],
        "desc": "the executives of a company",
    },
    "competitors_of": {
        "cypher": "MATCH (c {name:$name})-[:COMPETES_WITH]-(x) "
                  "RETURN DISTINCT x.name AS competitor",
        "entity_slots": ["name"],
        "desc": "the competitors of a company",
    },
    "regulators_of": {
        "cypher": "MATCH (c {name:$name})-[:REGULATED_BY]->(a) "
                  "RETURN a.name AS regulator",
        "entity_slots": ["name"],
        "desc": "the regulators of a company",
    },
    "neighbors": {
        "cypher": "MATCH (a {name:$name})-[r]-(b) WHERE NOT b:Chunk "
                  "RETURN type(r) AS relation, b.name AS connected_to LIMIT 50",
        "entity_slots": ["name"],
        "desc": "what an entity is connected to (a connection question)",
    },
    "path_between": {
        "cypher": "MATCH (a {name:$a}), (b {name:$b}), "
                  "p = shortestPath((a)-[*..5]-(b)) "
                  "RETURN [n IN nodes(p) | n.name] AS path, "
                  "[r IN relationships(p) | type(r)] AS relations",
        "entity_slots": ["a", "b"],
        "desc": "how two entities are connected (multi-hop path)",
    },
    "compare_subsidiary_counts": {
        "cypher": "MATCH (c:COMPANY)-[:HAS_SUBSIDIARY]->(s) WHERE c.name IN $names "
                  "RETURN c.name AS company, count(s) AS count ORDER BY count DESC",
        "list_slot": "names",
        "desc": "compare subsidiary counts across two or more companies",
    },
    "companies_facing_risk": {
        "cypher": "MATCH (c)-[:FACES_RISK]->(r) "
                  "WHERE toLower(r.name) CONTAINS toLower($risk) "
                  "RETURN DISTINCT c.name AS company",
        "value_slot": "risk",
        "desc": "which/how many companies face a given risk (value = risk keyword)",
    },
}


def _templates_doc() -> str:
    lines = []
    for key, t in TEMPLATES.items():
        if "entity_slots" in t:
            need = f"entities: {t['entity_slots']}"
        elif "list_slot" in t:
            need = "entities: [two or more]"
        else:
            need = f"value: <{t['value_slot']}>"
        lines.append(f"- {key} ({need}) — {t['desc']}")
    return "\n".join(lines)


# --- Entity resolution: surface name -> real node name ----------------------

_RESOLVE = """
MATCH (n) WHERE NOT n:Chunk AND n.name IS NOT NULL
WITH n, [l IN labels(n) WHERE NOT l STARTS WITH '__'][0] AS label
WHERE toLower(n.name) = toLower($q)
   OR any(a IN coalesce(n.aliases, []) WHERE toLower(a) = toLower($q))
   OR toLower(n.name) CONTAINS toLower($q)
   OR toLower($q) CONTAINS toLower(n.name)
RETURN n.name AS name, label,
   CASE WHEN toLower(n.name) = toLower($q) THEN 3
        WHEN any(a IN coalesce(n.aliases, []) WHERE toLower(a) = toLower($q)) THEN 2
        ELSE 1 END AS score
ORDER BY score DESC, size(n.name) ASC
LIMIT 1
"""


def resolve_entity(driver, surface: str) -> str | None:
    with driver.session() as s:
        rec = s.run(_RESOLVE, q=surface).single()
    return rec["name"] if rec else None


# --- Planning: model picks a template + names, NEVER Cypher -----------------

def plan(question: str) -> dict:
    system = (
        "Convert the question into a graph-query PLAN. Pick exactly one "
        "query_type from the list and extract the entity name(s) and/or value "
        "it needs. NEVER write Cypher.\n"
        f"Query types:\n{_templates_doc()}\n"
        'Reply ONLY JSON: {"query_type":"...","entities":["..."],"value":"..."}.'
    )
    client = ollama.Client(host=config.OLLAMA_BASE_URL)
    resp = client.chat(
        model=config.ROUTER_MODEL,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": question}],
        format="json",
    )
    return json.loads(resp["message"]["content"])


# --- Build bound params from the plan + resolve entities --------------------

def _display(cypher: str, params: dict) -> str:
    """A read-only rendering of the query for logging (NOT executed)."""
    out = cypher
    for k, v in params.items():
        out = out.replace(f"${k}", json.dumps(v))
    return out


def run_graph_query(question: str) -> dict:
    try:
        p = plan(question)
        qtype = str(p.get("query_type", "")).strip()
    except Exception as e:
        return {"query_type": None, "cypher": None, "rows": [],
                "error": f"plan failed: {str(e)[:80]}"}

    if qtype not in TEMPLATES:
        # Tolerate model typos by snapping to the nearest KNOWN key. This never
        # widens the attack surface — the result is still a fixed template.
        close = difflib.get_close_matches(qtype, TEMPLATES.keys(), n=1, cutoff=0.7)
        if not close:
            return {"query_type": qtype, "cypher": None, "rows": [],
                    "error": "no template for this query type"}
        qtype = close[0]

    tmpl = TEMPLATES[qtype]
    surfaces = p.get("entities") or []
    driver = GraphDatabase.driver(
        config.NEO4J_URI, auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD))
    try:
        resolved = [r for r in (resolve_entity(driver, s) for s in surfaces) if r]
        params: dict = {}

        if "entity_slots" in tmpl:
            slots = tmpl["entity_slots"]
            if len(resolved) < len(slots):
                return {"query_type": qtype, "cypher": None, "rows": [],
                        "error": f"could not resolve entities {surfaces} -> {resolved}"}
            params = dict(zip(slots, resolved))
        elif "list_slot" in tmpl:
            if len(resolved) < 2:
                return {"query_type": qtype, "cypher": None, "rows": [],
                        "error": f"need >=2 entities, resolved {resolved}"}
            params = {tmpl["list_slot"]: resolved}
        elif "value_slot" in tmpl:
            params = {tmpl["value_slot"]: p.get("value", "")}

        with driver.session() as s:
            rows = [dict(r) for r in s.run(tmpl["cypher"], **params)][:50]
        return {"query_type": qtype, "cypher": _display(tmpl["cypher"], params),
                "params": params, "resolved": resolved, "rows": rows}
    except Exception as e:
        return {"query_type": qtype, "cypher": tmpl["cypher"], "rows": [],
                "error": str(e)[:120]}
    finally:
        driver.close()


if __name__ == "__main__":
    for q in [
        "How many subsidiaries does Microsoft have?",
        "What are Apple's subsidiaries?",
        "Which companies face supply chain risk?",
        "Compare the subsidiaries of Apple and Microsoft.",
    ]:
        r = run_graph_query(q)
        print(f"\nQ: {q}")
        print(f"   type={r['query_type']}  resolved={r.get('resolved')}")
        print(f"   cypher={r['cypher']}")
        print(f"   rows={r['rows'][:6]}" + (f"  err={r.get('error')}" if r.get('error') else ""))
