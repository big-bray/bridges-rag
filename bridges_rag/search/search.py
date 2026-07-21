"""Query -> embed -> Qdrant search -> ranked results."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from bridges_rag.chunk.models import Chunk
from bridges_rag.embed.embedder import Embedder
from bridges_rag.index.qdrant import DEFAULT_COLLECTION
from bridges_rag.search.models import SearchResult


def build_filter(*, author: str | None = None, year: int | None = None) -> qmodels.Filter | None:
    """Build a Qdrant payload filter from optional author/year constraints."""
    must: list[qmodels.Condition] = []
    if author:
        must.append(qmodels.FieldCondition(key="authors", match=qmodels.MatchValue(value=author)))
    if year is not None:
        must.append(qmodels.FieldCondition(key="year", match=qmodels.MatchValue(value=year)))
    return qmodels.Filter(must=must) if must else None


def search(
    client: QdrantClient,
    embedder: Embedder,
    query: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    top_k: int = 10,
    author: str | None = None,
    year: int | None = None,
) -> list[SearchResult]:
    """Embed `query` and return the top-k matching chunks, most relevant first."""
    vector = embedder.embed_query(query).tolist()
    response = client.query_points(
        collection_name=collection,
        query=vector,
        query_filter=build_filter(author=author, year=year),
        limit=top_k,
        with_payload=True,
    )
    return [
        SearchResult(chunk=Chunk.model_validate(point.payload), score=point.score)
        for point in response.points
    ]
