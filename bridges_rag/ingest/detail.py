"""Parse a Bridges archive paper detail page.

Detail pages carry Google Scholar-style `citation_*` meta tags plus an
abstract div. These are more reliable than the listing page's free-text
byline, so callers should prefer the values here (`merge_detail`) over the
listing-derived stub wherever a detail page exists. Editor/publisher/address
aren't in these meta tags — those come from `bridges_rag.ingest.bibtex`.
"""

from __future__ import annotations

from datetime import date, datetime

from bs4 import BeautifulSoup

from bridges_rag.ingest.models import Paper


class DetailMetadata:
    """Fields scraped from a paper's own detail page."""

    def __init__(
        self,
        *,
        title: str | None,
        authors: list[str],
        abstract: str | None,
        isbn: str | None,
        issn: str | None,
        publication_date: date | None,
        conference_title: str | None,
        pdf_url: str | None,
    ) -> None:
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.isbn = isbn
        self.issn = issn
        self.publication_date = publication_date
        self.conference_title = conference_title
        self.pdf_url = pdf_url


def parse_detail(html: str) -> DetailMetadata:
    soup = BeautifulSoup(html, "html.parser")

    authors = [
        content.strip()
        for tag in soup.find_all("meta", attrs={"name": "citation_author"})
        if (content := tag.get("content"))
    ]

    abstract_el = soup.find("div", class_="abstract")
    abstract = abstract_el.get_text(" ", strip=True) if abstract_el else None

    return DetailMetadata(
        title=_meta_content(soup, "citation_title"),
        authors=authors,
        abstract=abstract,
        isbn=_meta_content(soup, "citation_isbn"),
        issn=_meta_content(soup, "citation_issn"),
        publication_date=_parse_citation_date(_meta_content(soup, "citation_publication_date")),
        conference_title=_meta_content(soup, "citation_conference_title"),
        pdf_url=_meta_content(soup, "citation_pdf_url"),
    )


def _parse_citation_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y/%m/%d").date()
    except ValueError:
        return None


def _meta_content(soup: BeautifulSoup, name: str) -> str | None:
    tag = soup.find("meta", attrs={"name": name})
    if tag is None:
        return None
    content = tag.get("content")
    return content.strip() if isinstance(content, str) else None


def merge_detail(paper: Paper, detail: DetailMetadata) -> Paper:
    """Return a copy of `paper` with detail-page fields filled in.

    Title/authors/pdf_url from the detail page win when present, since the
    listing page's free-text byline is a lossier source.
    """
    return paper.model_copy(
        update={
            "title": detail.title or paper.title,
            "authors": detail.authors or paper.authors,
            "abstract": detail.abstract,
            "isbn": detail.isbn,
            "issn": detail.issn,
            "publication_date": detail.publication_date,
            "conference_title": detail.conference_title,
            "pdf_url": detail.pdf_url or paper.pdf_url,
        }
    )
