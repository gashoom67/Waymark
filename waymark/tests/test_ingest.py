from __future__ import annotations

from waymark.ingest.chunker import Chunker, PageText
from waymark.ingest.images import analyse_image


def test_long_pdf_keeps_far_apart_facts(ingested):
    survey = ingested["shifting_sands_survey.pdf"]
    assert survey.pages >= 100
    texts = " ".join(n.text for n in survey.notes)
    assert "Dr. Lin Chen" in texts
    assert "Liwa Oasis Research Station" in texts
    pages = [n.page for n in survey.notes if "Liwa Oasis Research Station" in n.text]
    assert max(pages) > 60


def test_image_is_read_and_described(corpus_files):
    poster = next(p for p in corpus_files if p.suffix == ".png")
    report = analyse_image(poster)
    assert "Mercury Labs" in report.text
    assert len(report.lines) == 3
    assert report.kind in {"document", "diagram", "screenshot"}
    assert report.palette and report.palette[0]["hex"].startswith("#")
    assert all(0.0 <= x <= 1.0 for line in report.lines for point in line["box"] for x in point)


def test_headings_become_passage_titles():
    page = PageText(page=1, text="Qasr Al Sarab Field Laboratory\nThe lab was set up near Hamim in 2004 and works on aquifers in the western region.\n\nIntroduction\nThis part has a generic heading.")
    chunks = Chunker(target=200).chunk(page)
    assert chunks[0].title == "Qasr Al Sarab Field Laboratory"
    assert chunks[0].text.startswith("The lab was set up")
    assert chunks[-1].title == ""


def test_hyphenated_line_breaks_are_joined():
    page = PageText(page=1, text="The survey measured dis-\nambiguation errors across\nall twenty papers in the set.")
    chunks = Chunker().chunk(page)
    assert "disambiguation" in chunks[0].text
