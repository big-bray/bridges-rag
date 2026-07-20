"""Format a chunk's paper metadata as a human-readable citation."""

from __future__ import annotations

from bridges_rag.chunk.models import Chunk


def format_citation(chunk: Chunk) -> str:
    """e.g. 'Ada Lovelace and Alan Turing. "A Paper Title." Bridges 2025, pp. 29-31.'"""
    citation = f'{_format_authors(chunk.authors)}. "{chunk.title}." Bridges {chunk.year}'
    pages = _format_pages(chunk.proceedings_pages)
    if pages:
        citation += f", {pages}"
    return citation + "."


def _format_authors(authors: list[str]) -> str:
    if not authors:
        return "Unknown"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return f"{', '.join(authors[:-1])}, and {authors[-1]}"


def _format_pages(pages: list[int]) -> str:
    if not pages:
        return ""
    if len(pages) == 1:
        return f"p. {pages[0]}"
    return f"pp. {min(pages)}-{max(pages)}"
