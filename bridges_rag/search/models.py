"""Pydantic model for a ranked search result."""

from __future__ import annotations

from pydantic import BaseModel

from bridges_rag.chunk.models import Chunk


class SearchResult(BaseModel):
    """A chunk returned for a query, with its similarity score."""

    chunk: Chunk
    score: float
