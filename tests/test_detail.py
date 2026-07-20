from datetime import date
from pathlib import Path

from bridges_rag.ingest.detail import merge_detail, parse_detail
from bridges_rag.ingest.models import Paper

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_detail_single_author():
    html = (FIXTURES / "detail_regular.html").read_text(encoding="utf-8")
    detail = parse_detail(html)

    assert detail.title == "Algorithmic Generation of Interlaced Stellations"
    assert detail.authors == ["Fahreddin Başeğmez"]
    assert detail.isbn == "978-1-938664-51-9"
    assert detail.issn == "1099-6702"
    assert detail.pdf_url == "https://archive.bridgesmathart.org/2025/bridges2025-29.pdf"
    assert detail.abstract is not None
    assert "SI" in detail.abstract
    assert detail.publication_date == date(2025, 7, 1)
    assert detail.conference_title == "Proceedings of Bridges 2025: Mathematics and the Arts"


def test_parse_detail_multiple_citation_author_tags():
    html = (FIXTURES / "detail_multi_author.html").read_text(encoding="utf-8")
    detail = parse_detail(html)

    assert detail.authors == ["Nuno Bastos", "Andreia Hall"]


def test_merge_detail_prefers_detail_fields_over_listing_stub():
    stub = Paper(
        paper_id="bridges2025-29",
        year=2025,
        title="mangled listing title",
        authors=["Mangled Name"],
        category="Regular Papers",
        first_page=29,
        last_page=36,
        detail_url="https://archive.bridgesmathart.org/2025/bridges2025-29.html",
        pdf_url="https://archive.bridgesmathart.org/2025/bridges2025-29.pdf",
    )
    html = (FIXTURES / "detail_regular.html").read_text(encoding="utf-8")
    merged = merge_detail(stub, parse_detail(html))

    assert merged.title == "Algorithmic Generation of Interlaced Stellations"
    assert merged.authors == ["Fahreddin Başeğmez"]
    assert merged.abstract is not None
    assert merged.isbn == "978-1-938664-51-9"
    assert merged.publication_date == date(2025, 7, 1)
    assert merged.conference_title == "Proceedings of Bridges 2025: Mathematics and the Arts"
    # fields not touched by detail parsing are preserved from the listing stub
    assert merged.first_page == 29
    assert merged.last_page == 36
    assert merged.category == "Regular Papers"
