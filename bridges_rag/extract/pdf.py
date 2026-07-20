"""Extract per-page markdown text from a PDF."""

from __future__ import annotations

from pathlib import Path

import pymupdf4llm

from bridges_rag.extract.models import PageMarkdown


def extract_pages(pdf_path: Path, *, first_page: int | None = None) -> list[PageMarkdown]:
    chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

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
