"""The SEC knowledge-graph ontology.

This is the single most important file for extraction *quality*. Instead of
letting the LLM invent arbitrary entities and relationships (which produces a
noisy, unqueryable graph), we constrain it to a fixed schema. The extractor is
only allowed to emit entities of these types, relations of these types, and
subject->relation->object triples that appear in VALIDATION_SCHEMA.
"""
from __future__ import annotations

from typing import Literal

# --- Node (entity) types ---
Entities = Literal[
    "COMPANY",            # the filer, subsidiaries, competitors, partners
    "PERSON",             # executives, directors
    "PRODUCT",            # products / services / brands
    "SEGMENT",            # reportable business segments
    "RISK",               # risk factors (Item 1A)
    "AUDITOR",            # independent registered public accounting firm
    "LOCATION",           # cities / states / countries
    "GOVERNMENT_AGENCY",  # regulators (SEC, FDA, FTC, ...)
    "STOCK_EXCHANGE",     # NYSE, Nasdaq, ...
]

# --- Relationship (edge) types ---
Relations = Literal[
    "HAS_SUBSIDIARY",
    "COMPETES_WITH",
    "PARTNERS_WITH",
    "ACQUIRED",
    "SUPPLIES_TO",
    "HAS_EXECUTIVE",
    "HAS_DIRECTOR",
    "OFFERS_PRODUCT",
    "OPERATES_SEGMENT",
    "FACES_RISK",
    "AUDITED_BY",
    "HEADQUARTERED_IN",
    "OPERATES_IN",
    "REGULATED_BY",
    "LISTED_ON",
]

# --- Which (subject, relation, object) shapes are allowed ---
# strict=True in the extractor rejects any triple not listed here, which is
# what keeps the graph clean and consistently queryable.
VALIDATION_SCHEMA: list[tuple[str, str, str]] = [
    ("COMPANY", "HAS_SUBSIDIARY", "COMPANY"),
    ("COMPANY", "COMPETES_WITH", "COMPANY"),
    ("COMPANY", "PARTNERS_WITH", "COMPANY"),
    ("COMPANY", "ACQUIRED", "COMPANY"),
    ("COMPANY", "SUPPLIES_TO", "COMPANY"),
    ("COMPANY", "HAS_EXECUTIVE", "PERSON"),
    ("COMPANY", "HAS_DIRECTOR", "PERSON"),
    ("COMPANY", "OFFERS_PRODUCT", "PRODUCT"),
    ("COMPANY", "OPERATES_SEGMENT", "SEGMENT"),
    ("COMPANY", "FACES_RISK", "RISK"),
    ("COMPANY", "AUDITED_BY", "AUDITOR"),
    ("COMPANY", "HEADQUARTERED_IN", "LOCATION"),
    ("COMPANY", "OPERATES_IN", "LOCATION"),
    ("COMPANY", "REGULATED_BY", "GOVERNMENT_AGENCY"),
    ("COMPANY", "LISTED_ON", "STOCK_EXCHANGE"),
    ("SEGMENT", "OFFERS_PRODUCT", "PRODUCT"),
]

# A short natural-language description injected into the extraction prompt so
# the model understands the domain and applies the labels consistently.
EXTRACTION_HINT = (
    "You are reading a company's SEC 10-K annual report. Extract only facts that "
    "are explicitly stated. Identify the filing company, its subsidiaries, "
    "competitors, partners, executives and directors, products, business "
    "segments, auditor, regulators, stock exchanges, headquarters/operating "
    "locations, and the key risk factors it discloses. Use the company's common "
    "name (e.g. 'Apple' not 'Apple Inc., a California corporation')."
)
