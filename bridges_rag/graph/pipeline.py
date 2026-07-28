from __future__ import annotations

from collections.abc import Iterable, Sequence

from bridges_rag.chunk.models import Chunk
from bridges_rag.ingest.models import Paper


def entity_lists(papers: Iterable[Paper]) -> tuple[list[str], list[str]]:
    """Known author names and paper titles to entity-link queries against."""
    authors: set[str] = set()
    titles: set[str] = set()
    for paper in papers:
        authors.update(paper.authors)
        titles.add(paper.title)
    return sorted(authors), sorted(titles)


def representative_chunks(chunks: Sequence[Chunk]) -> dict[str, Chunk]:
    """One chunk per paper (the first, by chunk_index), for graph-only hits that
    the base vector retriever didn't surface any chunk for."""
    by_paper: dict[str, Chunk] = {}
    for chunk in chunks:
        existing = by_paper.get(chunk.paper_id)
        if existing is None or chunk.chunk_index < existing.chunk_index:
            by_paper[chunk.paper_id] = chunk
    return by_paper
