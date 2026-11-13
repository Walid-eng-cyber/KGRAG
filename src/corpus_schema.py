"""Extraction schemas for corpora: built-in presets plus validated custom schemas.

A schema says which entity types the LLM may extract and which
(subject, RELATION, object) shapes are allowed. Every corpus stores its own, so
a CV corpus extracts skills and degrees while a filings corpus extracts
subsidiaries and risks.
"""
from __future__ import annotations

import re

from src.schema import VALIDATION_SCHEMA as COMPANY_RELATIONS

PRESETS: dict[str, dict] = {
    "company": {
        "label": "Company filings",
        "description": "Annual reports, 10-Ks, investor documents: subsidiaries, executives, risks, products.",
        "entities": ["COMPANY", "PERSON", "PRODUCT", "SEGMENT", "RISK", "AUDITOR", "LOCATION",
                     "GOVERNMENT_AGENCY", "STOCK_EXCHANGE"],
        "relations": [list(t) for t in COMPANY_RELATIONS],
    },
    "career": {
        "label": "CV & career",
        "description": "CVs, transcripts, certificates: education, jobs, skills, projects.",
        "entities": ["PERSON", "ORGANIZATION", "INSTITUTION", "DEGREE", "COURSE", "ROLE", "SKILL",
                     "PROJECT", "CERTIFICATION", "LANGUAGE", "LOCATION"],
        "relations": [
            ["PERSON", "WORKED_AT", "ORGANIZATION"], ["PERSON", "HELD_ROLE", "ROLE"],
            ["ROLE", "AT", "ORGANIZATION"], ["PERSON", "STUDIED_AT", "INSTITUTION"],
            ["PERSON", "EARNED", "DEGREE"], ["DEGREE", "FROM", "INSTITUTION"],
            ["PERSON", "TOOK_COURSE", "COURSE"], ["COURSE", "PART_OF", "DEGREE"],
            ["PERSON", "HAS_SKILL", "SKILL"], ["PERSON", "WORKED_ON", "PROJECT"],
            ["PROJECT", "USED_SKILL", "SKILL"], ["PROJECT", "DONE_AT", "ORGANIZATION"],
            ["PROJECT", "DONE_AT", "INSTITUTION"], ["PERSON", "HOLDS", "CERTIFICATION"],
            ["PERSON", "SPEAKS", "LANGUAGE"], ["PERSON", "LIVES_IN", "LOCATION"],
            ["ORGANIZATION", "LOCATED_IN", "LOCATION"], ["INSTITUTION", "LOCATED_IN", "LOCATION"],
        ],
    },
    "general": {
        "label": "General",
        "description": "Any documents: people, organizations, places, events, products and concepts.",
        "entities": ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "PRODUCT", "CONCEPT", "DOCUMENT"],
        "relations": [
            ["PERSON", "WORKS_FOR", "ORGANIZATION"], ["PERSON", "MEMBER_OF", "ORGANIZATION"],
            ["ORGANIZATION", "PART_OF", "ORGANIZATION"], ["ORGANIZATION", "LOCATED_IN", "LOCATION"],
            ["PERSON", "LOCATED_IN", "LOCATION"], ["EVENT", "TOOK_PLACE_IN", "LOCATION"],
            ["PERSON", "PARTICIPATED_IN", "EVENT"], ["ORGANIZATION", "PRODUCES", "PRODUCT"],
            ["PERSON", "CREATED", "PRODUCT"], ["CONCEPT", "RELATED_TO", "CONCEPT"],
            ["DOCUMENT", "DISCUSSES", "CONCEPT"], ["PERSON", "KNOWS", "PERSON"],
        ],
    },
}

_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
MAX_ENTITIES, MAX_RELATIONS = 20, 40


class SchemaError(ValueError):
    pass


def _norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9_]", "_", str(name).strip().upper().replace(" ", "_")).strip("_")


def validate(entities: list[str], relations: list[list[str]]) -> dict:
    """Normalize and check a custom schema. Names become UPPER_SNAKE_CASE.
    Raises SchemaError with a message fit for the UI."""
    ents = list(dict.fromkeys(_norm(e) for e in entities if str(e).strip()))
    if not ents:
        raise SchemaError("Add at least one entity type.")
    if len(ents) > MAX_ENTITIES:
        raise SchemaError(f"At most {MAX_ENTITIES} entity types.")
    for e in ents:
        if not _NAME.match(e):
            raise SchemaError(f"“{e}” isn't a valid type name (letters, digits, underscores; starts with a letter).")
    rels, seen = [], set()
    for r in relations:
        if len(r) != 3:
            raise SchemaError("Each relation needs a subject type, a relation name and an object type.")
        s, rel, o = (_norm(x) for x in r)
        if s not in ents or o not in ents:
            raise SchemaError(f"Relation {s} {rel} {o} uses a type that isn't in the entity list.")
        if not _NAME.match(rel) or rel in {"MENTIONS", "INSIDER_OF", "TRADED"}:
            raise SchemaError(f"“{rel}” isn't a valid relation name.")
        if (s, rel, o) not in seen:
            seen.add((s, rel, o))
            rels.append([s, rel, o])
    if not rels:
        raise SchemaError("Add at least one relation.")
    if len(rels) > MAX_RELATIONS:
        raise SchemaError(f"At most {MAX_RELATIONS} relations.")
    return {"entities": ents, "relations": rels}


def from_preset(key: str) -> dict:
    if key not in PRESETS:
        raise SchemaError(f"Unknown preset “{key}”.")
    p = PRESETS[key]
    return {"preset": key, **validate(p["entities"], p["relations"])}


def extraction_hint(schema: dict, corpus_name: str, description: str) -> str:
    """The instruction placed before each document for the extractor."""
    about = f" The documents are about: {description.strip()}." if description.strip() else ""
    return (f"You are reading a document from the collection “{corpus_name}”.{about} Extract only facts "
            f"that are explicitly stated, as entities of these types: {', '.join(schema['entities'])}. "
            "Use each entity's full, common name.")
