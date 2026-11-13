"""Layout-aware chunking (pure text, no models)."""
from src.chunking import blocks, chunk_texts, is_heading, strip_boilerplate

CV = """Jane Doe
Data engineer, Berlin
PROJECTS
Fraud Detector
Personal project | GitHub
01.03.2024 – 01.07.2024
● Built a streaming model.
Technologies: Python | Kafka | Docker
Price Tracker
Company X | GitHub
01.08.2024 – 01.10.2024
● Scraped and stored prices.
Technologies: Go | PostgreSQL
Jane Doe · CV · Page 1
\f
EXPERIENCE
Data Engineer
Company X, Berlin
since 01/2025
● Pipelines in Airflow.
Jane Doe · CV · Page 2
"""


def test_headings():
    assert is_heading("BERUFSERFAHRUNG") and is_heading("HOBBYS UND INTERESSEN") and is_heading("RISK FACTORS")
    assert not is_heading("Room Booking") and not is_heading("01.09.2024 – AKTUELL") and not is_heading("PNR")


def test_each_entry_stays_whole_with_its_fields():
    bs = blocks(strip_boilerplate(CV))
    titles = [(s, b.split("\n")[0]) for s, b in bs]
    assert ("PROJECTS", "Fraud Detector") in titles and ("PROJECTS", "Price Tracker") in titles
    assert ("EXPERIENCE", "Data Engineer") in titles
    fraud = next(b for s, b in bs if b.startswith("Fraud Detector"))
    assert "Technologies: Python | Kafka | Docker" in fraud and "Price Tracker" not in fraud


def test_repeated_footer_is_kept_once():
    s = strip_boilerplate(CV)
    assert s.count("Jane Doe · CV · Page") == 1


def test_chunks_carry_their_section_and_fall_back_for_prose():
    cs = chunk_texts(strip_boilerplate(CV), max_chars=200, min_chars=50)
    assert any("PROJECTS\nFraud Detector" in c for c in cs)          # a short intro joins the next section
    assert any(c.startswith("PROJECTS\nPrice Tracker") for c in cs)    # a continued section repeats its heading
    assert any(c.startswith("EXPERIENCE") for c in cs)
    letter = "Dear Sir,\nthank you for your letter.\nBest regards\n\fPage two of the letter.\n"
    assert chunk_texts(letter) == ["Dear Sir,\nthank you for your letter.\nBest regards", "Page two of the letter."]
