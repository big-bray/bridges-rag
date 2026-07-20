"""Parse a Bridges archive paper detail page.

Detail pages carry Google Scholar-style `citation_*` meta tags plus an
abstract div. These are more reliable than the listing page's free-text
byline, so callers should prefer the values here (`merge_detail`) over the
listing-derived stub wherever a detail page exists.
"""

from __future__ import annotations

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
        pdf_url: str | None,
    ) -> None:
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.isbn = isbn
        self.issn = issn
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
        pdf_url=_meta_content(soup, "citation_pdf_url"),
    )


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
            "pdf_url": detail.pdf_url or paper.pdf_url,
        }
    )
