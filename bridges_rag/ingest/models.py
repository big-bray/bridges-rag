"""Pydantic models for scraped Bridges archive metadata."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, computed_field

FRONT_MATTER_CATEGORY = "Front Matter"


class Paper(BaseModel):
    """Metadata for a single paper in a Bridges proceedings year.

    Populated in three passes: `parse_listing` fills the fields available on
    the year's listing page, then (for entries with a `detail_url`)
    `parse_detail` fills in the abstract/isbn/issn/dates from the paper's own
    page, and `parse_bibtex` fills in editors/publisher/address from its
    BibTeX citation file — the only source for those three. `pdf_sha256` and
    `pdf_path` are filled in last, once the PDF has been downloaded.
    """

    paper_id: str
    year: int
    title: str
    authors: list[str]
    category: str
    first_page: int | None = None
    last_page: int | None = None
    abstract: str | None = None
    detail_url: str | None = None
    pdf_url: str
    isbn: str | None = None
    issn: str | None = None
    publication_date: date | None = None
    conference_title: str | None = None
    bibtex_url: str | None = None
    bibtex: str | None = None
    editors: list[str] = []
    publisher: str | None = None
    address: str | None = None
    pdf_sha256: str | None = None
    pdf_path: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def page_count(self) -> int | None:
        if self.first_page is None or self.last_page is None:
            return None
        return self.last_page - self.first_page + 1
