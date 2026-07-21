"""Pydantic model for a chunk of paper text ready for embedding."""

from __future__ import annotations

from pydantic import BaseModel


class Chunk(BaseModel):
    """A heading-scoped, token-budgeted span of a paper's markdown, with metadata
    denormalized on so it can be embedded and indexed without a join back to the paper."""

    chunk_id: str
    paper_id: str
    chunk_index: int
    heading: str | None
    text: str
    pdf_pages: list[int]
    proceedings_pages: list[int]
    title: str
    authors: list[str]
    year: int
