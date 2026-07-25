"""Data models for figures extracted from paper PDFs."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel


@dataclass
class ExtractedFigure:
    """A raster figure found on one page of a PDF, before it's persisted to disk.

    Image bytes are transient — never written to the figure manifest, only to a file
    under the figures directory."""

    pdf_page: int
    proceedings_page: int | None
    bbox: tuple[float, float, float, float]
    caption: str | None
    image_bytes: bytes
    image_ext: str
    width: int
    height: int


class Figure(BaseModel):
    """A figure extracted from a paper's PDF, with metadata denormalized on so it
    can be indexed and searched without a join back to the paper."""

    figure_id: str
    paper_id: str
    pdf_page: int
    proceedings_page: int | None = None
    bbox: tuple[float, float, float, float]
    """(x0, y0, x1, y1) in PDF page points."""
    caption: str | None = None
    image_path: str
    """Path to the saved image, relative to the data directory."""
    image_ext: str
    width: int
    height: int
    checksum: str
    """sha256 of the saved image bytes."""
    title: str
    authors: list[str]
    year: int
