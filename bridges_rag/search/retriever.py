"""Retriever protocol: the interface the eval harness benchmarks against.

Dense vector search is the reference implementation. Hybrid search, reranking,
and the Stream C graph retriever all become benchmarkable by implementing
this same protocol, with no changes to eval/runner.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient

from bridges_rag.embed.embedder import Embedder
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
        return search(
            self.client, self.embedder, question, collection=self.collection, top_k=top_k
        )
