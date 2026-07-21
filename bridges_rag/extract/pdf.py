"""Extract per-page markdown text from a PDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pymupdf4llm

from bridges_rag.extract.models import PageMarkdown


def _extract_pages_from_doc(
    doc: pymupdf.Document, *, first_page: int | None = None
) -> list[PageMarkdown]:
    chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)

    pages = []
    for chunk in chunks:
        pdf_page = chunk["metadata"]["page_number"]
        proceedings_page = first_page + pdf_page - 1 if first_page is not None else None
        pages.append(
            PageMarkdown(
                pdf_page=pdf_page,
                proceedings_page=proceedings_page,
                markdown=chunk["text"],
            )
        )
    return pages


def extract_pages(pdf_path: Path, *, first_page: int | None = None) -> list[PageMarkdown]:
    with pymupdf.open(pdf_path) as doc:  # type: ignore[no-untyped-call]
        return _extract_pages_from_doc(doc, first_page=first_page)


def extract_pages_from_stream(data: bytes, *, first_page: int | None = None) -> list[PageMarkdown]:
    """Same as `extract_pages`, but from an in-memory PDF that is never written to disk."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        return _extract_pages_from_doc(doc, first_page=first_page)
