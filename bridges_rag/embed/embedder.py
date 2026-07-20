"""Wraps sentence-transformers for batch embedding with progress reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# bge models are trained asymmetrically: passages are embedded as-is, but queries
# need this instruction prefix to land in the same retrieval space. Passage-side
# embedding happens here (indexing); query-side is Milestone 3's concern.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Batch text -> vector embedding using a local sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        size = self._model.get_embedding_dimension()
        assert size is not None
        return cast(int, size)

    def embed_passages(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        return cast(
            np.ndarray,
            self._model.encode(
                list(texts),
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
        )

    def embed_query(self, text: str) -> np.ndarray:
        result = self._model.encode([BGE_QUERY_PREFIX + text], normalize_embeddings=True)
        return cast(np.ndarray, result[0])
