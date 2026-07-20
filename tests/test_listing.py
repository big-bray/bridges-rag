from pathlib import Path

from bridges_rag.ingest.listing import _split_authors, parse_listing
from bridges_rag.ingest.models import FRONT_MATTER_CATEGORY

FIXTURE = Path(__file__).parent / "fixtures" / "listing_2025.html"
BASE_URL = "https://archive.bridgesmathart.org/2025/"


def _parse_fixture():
    html = FIXTURE.read_text(encoding="utf-8")
    return parse_listing(html, year=2025, base_url=BASE_URL)


def test_parses_all_entries_in_order():
    papers = _parse_fixture()
    assert [p.paper_id for p in papers] == [
        "frontmatter",
        "bridges2025-1",
        "bridges2025-3",
        "bridges2025-101",
        "bridges2025-575",
    ]


def test_front_matter_has_no_detail_page_and_direct_pdf_url():
    papers = _parse_fixture()
    front_matter = papers[0]
    assert front_matter.category == FRONT_MATTER_CATEGORY
    assert front_matter.detail_url is None
    assert front_matter.pdf_url == BASE_URL + "frontmatter.pdf"
    assert front_matter.authors == ["The Editors"]
    assert front_matter.first_page is None
    assert front_matter.last_page is None


def test_category_carries_forward_to_following_entries():
    papers = _parse_fixture()
    assert papers[1].category == "Invited Papers"
    assert papers[2].category == "Invited Papers"
    assert papers[3].category == "Regular Papers"
    assert papers[4].category == "Workshop Papers"


def test_detail_and_pdf_urls_derived_from_html_href():
    papers = _parse_fixture()
    paper = papers[1]
    assert paper.detail_url == BASE_URL + "bridges2025-1.html"
    assert paper.pdf_url == BASE_URL + "bridges2025-1.pdf"


def test_single_page_paper_page_range():
    papers = _parse_fixture()
    paper = papers[1]
    assert paper.first_page == 1
    assert paper.last_page == 1
    assert paper.page_count == 1


def test_multi_page_paper_page_range_and_count():
    papers = _parse_fixture()
    paper = papers[3]
    assert paper.first_page == 101
    assert paper.last_page == 108
    assert paper.page_count == 8


def test_title_with_html_entities_and_smart_quotes():
    papers = _parse_fixture()
    assert papers[2].title == "Eindhoven and Escher’s Connection with the World of Mathematics"


def test_split_authors_two_names_no_comma():
    assert _split_authors("Micky Piller and Kristoffel Lieten") == [
        "Micky Piller",
        "Kristoffel Lieten",
    ]


def test_split_authors_oxford_comma_list():
    assert _split_authors("Loe Feijs, Rong-Hao Liang, Holly Krueger, and Marina Toeters") == [
        "Loe Feijs",
        "Rong-Hao Liang",
        "Holly Krueger",
        "Marina Toeters",
    ]


def test_split_authors_single_name():
    assert _split_authors("Brigitte Kock") == ["Brigitte Kock"]
