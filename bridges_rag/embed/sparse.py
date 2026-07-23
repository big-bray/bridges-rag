"""BM25 sparse embeddings via FastEmbed, for hybrid dense+sparse retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from fastembed import SparseTextEmbedding
from qdrant_client.http import models as qmodels

DEFAULT_SPARSE_MODEL_NAME = "Qdrant/bm25"


class SparseEmbedder:
    """Batch text -> BM25 sparse vector using FastEmbed."""

    def __init__(self, model_name: str = DEFAULT_SPARSE_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)

    def embed_passages(self, texts: Sequence[str]) -> list[qmodels.SparseVector]:
        return [
            qmodels.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
            for e in self._model.embed(list(texts))
        ]

    def embed_query(self, text: str) -> qmodels.SparseVector:
        (embedding,) = self._model.query_embed([text])
        return qmodels.SparseVector(
            indices=embedding.indices.tolist(), values=embedding.values.tolist()
        )
