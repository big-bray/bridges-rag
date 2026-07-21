"""Cross-encoder reranking: retrieve a wide candidate pool cheaply, rerank precisely.

A cross-encoder scores (query, passage) pairs jointly instead of comparing
independently-embedded vectors, which is more precise but too slow to run over
a whole collection — so it only ever reranks a small pool from a base Retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sentence_transformers import CrossEncoder

from bridges_rag.search.models import SearchResult
from bridges_rag.search.retriever import Retriever

DEFAULT_RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"


class Reranker(Protocol):
    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        """Score and reorder `results` against `query`, returning the top-k."""
        ...


class CrossEncoderReranker:
    """Reference Reranker: a cross-encoder scoring (query, passage) pairs directly."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        if not results:
            return []
        pairs = [[query, result.chunk.text] for result in results]
        scores = self._model.predict(pairs)
        reranked = sorted(zip(results, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in reranked[:top_k]
        ]


@dataclass
class RerankRetriever:
    """Retriever: fetch a candidate pool via `base`, then rerank it to top-k."""

    base: Retriever
    reranker: Reranker
    pool_size: int = 100

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        candidates = self.base.retrieve(question, top_k=max(self.pool_size, top_k))
        return self.reranker.rerank(question, candidates, top_k=top_k)
