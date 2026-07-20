from bridges_rag.chunk.models import Chunk
from bridges_rag.search.citation import format_citation


def _chunk(**overrides: object) -> Chunk:
    defaults = dict(
        chunk_id="bridges2025-1-0000",
        paper_id="bridges2025-1",
        chunk_index=0,
        heading="# Intro",
        text="# Intro\n\nSome text.",
        pdf_pages=[1],
        proceedings_pages=[29, 30],
        title="A Paper Title",
        authors=["Ada Lovelace"],
        year=2025,
    )
    defaults.update(overrides)
    return Chunk(**defaults)  # type: ignore[arg-type]


def test_format_citation_single_author_and_page_range():
    citation = format_citation(_chunk())
    assert citation == 'Ada Lovelace. "A Paper Title." Bridges 2025, pp. 29-30.'


def test_format_citation_two_authors():
    citation = format_citation(_chunk(authors=["Ada Lovelace", "Alan Turing"]))
    assert citation.startswith("Ada Lovelace and Alan Turing.")


def test_format_citation_three_or_more_authors():
    citation = format_citation(_chunk(authors=["Ada Lovelace", "Alan Turing", "Grace Hopper"]))
    assert citation.startswith("Ada Lovelace, Alan Turing, and Grace Hopper.")


def test_format_citation_single_page():
    citation = format_citation(_chunk(proceedings_pages=[29]))
    assert citation == 'Ada Lovelace. "A Paper Title." Bridges 2025, p. 29.'


def test_format_citation_no_pages():
    citation = format_citation(_chunk(proceedings_pages=[]))
    assert citation == 'Ada Lovelace. "A Paper Title." Bridges 2025.'


def test_format_citation_no_authors():
    citation = format_citation(_chunk(authors=[]))
    assert citation.startswith("Unknown.")
