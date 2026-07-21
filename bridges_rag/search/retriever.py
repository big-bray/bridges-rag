"""Retriever protocol: the interface the eval harness benchmarks against."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient

from bridges_rag.embed.embedder import Embedder
from bridges_rag.embed.sparse import SparseEmbedder
from bridges_rag.index.qdrant import DEFAULT_COLLECTION
from bridges_rag.search.models import SearchResult
from bridges_rag.search.search import search


class Retriever(Protocol):
    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        """Return the top-k results for `question`, most relevant first."""
        ...


@dataclass
class DenseRetriever:
    """Reference Retriever: dense vector search over Qdrant."""

    client: QdrantClient
    embedder: Embedder
    collection: str = DEFAULT_COLLECTION

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        return search(self.client, self.embedder, question, collection=self.collection, top_k=top_k)


@dataclass
class HybridRetriever:
    """Dense + BM25 sparse retrieval, fused server-side via Qdrant RRF.

    The collection must have been indexed with sparse vectors
    (`ensure_collection(..., with_sparse=True)` / `index --hybrid`).
    """

    client: QdrantClient
    embedder: Embedder
    sparse_embedder: SparseEmbedder
    collection: str = DEFAULT_COLLECTION
    prefetch_limit: int = 50

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        return search(
            self.client,
            self.embedder,
            question,
            collection=self.collection,
            top_k=top_k,
            hybrid=True,
            sparse_embedder=self.sparse_embedder,
            prefetch_limit=self.prefetch_limit,
        )
