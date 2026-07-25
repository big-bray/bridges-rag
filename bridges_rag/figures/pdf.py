"""Extract embedded figures and their nearby captions from a PDF."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from bridges_rag.figures.models import ExtractedFigure

MIN_IMAGE_DIM = 32
"""Pixel width/height below which an embedded image is treated as a decorative
icon or rule line rather than a figure."""

CAPTION_MAX_DISTANCE = 150.0
"""Points a "Figure N" text block may sit above/below an image and still count
as its caption."""

_CAPTION_RE = re.compile(r"^\s*fig(?:ure)?\.?\s*\d+", re.IGNORECASE)


def _caption_for(
    image_rect: pymupdf.Rect,
    text_blocks: list[tuple[float, float, float, float, str, int, int]],
    max_distance: float,
) -> str | None:
    """Nearest "Figure N" text block to `image_rect` by vertical gap, within `max_distance`."""
    best_text: str | None = None
    best_distance: float | None = None
    for _x0, y0, _x1, y1, text, *_ in text_blocks:
        candidate = text.strip()
        if not _CAPTION_RE.match(candidate):
            continue
        if y0 >= image_rect.y1:
            distance = y0 - image_rect.y1
        elif y1 <= image_rect.y0:
            distance = image_rect.y0 - y1
        else:
            distance = 0.0
        if best_distance is None or distance < best_distance:
            best_distance, best_text = distance, candidate
    if best_text is None or (best_distance is not None and best_distance > max_distance):
        return None
    return best_text


def _extract_figures_from_doc(
    doc: pymupdf.Document,
    *,
    first_page: int | None = None,
    min_dim: int = MIN_IMAGE_DIM,
    caption_max_distance: float = CAPTION_MAX_DISTANCE,
) -> list[ExtractedFigure]:
    figures = []
    for page in doc.pages():
        pdf_page = page.number + 1
        proceedings_page = first_page + pdf_page - 1 if first_page is not None else None
        text_blocks = page.get_text("blocks")  # type: ignore[no-untyped-call]

        seen_xrefs: set[int] = set()
        for img in page.get_images(full=True):  # type: ignore[no-untyped-call]
            xref = img[0]
            if xref in seen_xrefs:
                # Identical pixel content is deduplicated to one xref by PyMuPDF even when
                # inserted at multiple placements, so get_images() can list it more than
                # once; get_image_rects() below already returns every placement in one call.
                continue
            seen_xrefs.add(xref)

            info = doc.extract_image(xref)  # type: ignore[no-untyped-call]
            if info["width"] < min_dim or info["height"] < min_dim:
                continue

            for rect in page.get_image_rects(xref):
                figures.append(
                    ExtractedFigure(
                        pdf_page=pdf_page,
                        proceedings_page=proceedings_page,
                        bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                        caption=_caption_for(rect, text_blocks, caption_max_distance),
                        image_bytes=info["image"],
                        image_ext=info["ext"],
                        width=info["width"],
                        height=info["height"],
                    )
                )
    return figures


def extract_figures(pdf_path: Path, *, first_page: int | None = None) -> list[ExtractedFigure]:
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        return _extract_figures_from_doc(doc, first_page=first_page)


def extract_figures_from_stream(
    data: bytes, *, first_page: int | None = None
) -> list[ExtractedFigure]:
    with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        return _extract_figures_from_doc(doc, first_page=first_page)
