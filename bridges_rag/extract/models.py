"""Pydantic models for extracted per-paper markdown."""

from __future__ import annotations

from pydantic import BaseModel


class PageMarkdown(BaseModel):
    """Markdown text for a single page of a paper's PDF."""

    pdf_page: int
    """1-indexed page number within the downloaded PDF."""

    proceedings_page: int | None = None
    """Published page number in the proceedings, derived from the paper's
    `first_page` (from the ingest manifest) when known."""

    markdown: str


class ExtractedPaper(BaseModel):
    """Result of extracting markdown from one paper's PDF."""

    paper_id: str
    pages: list[PageMarkdown]
    error: str | None = None
    """Set instead of raising when extraction fails, so one bad PDF doesn't
    stop the rest of a batch."""

    @property
    def ok(self) -> bool:
        return self.error is None
