"""Wraps sentence-transformers for batch embedding with progress reporting."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Some models are trained asymmetrically: query and passage text need different
# instruction prefixes to land in the same retrieval space. Models not listed here
# (gte-large, all-MiniLM-L6-v2) are symmetric and need no prefix.
QUERY_PREFIXES: dict[str, str] = {
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-large-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "intfloat/e5-base-v2": "query: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_query: ",
}

PASSAGE_PREFIXES: dict[str, str] = {
    "intfloat/e5-base-v2": "passage: ",
    "nomic-ai/nomic-embed-text-v1.5": "search_document: ",
}

# Models whose sentence-transformers integration requires executing model-repo code
# (e.g. a custom rotary-embedding implementation) rather than a stock architecture.
TRUST_REMOTE_CODE_MODELS = {"nomic-ai/nomic-embed-text-v1.5"}


def query_prefix(model_name: str) -> str:
    """The query-side instruction prefix for `model_name`, or "" if it doesn't need one."""
    return QUERY_PREFIXES.get(model_name, "")


def passage_prefix(model_name: str) -> str:
    """The passage-side instruction prefix for `model_name`, or "" if it doesn't need one."""
    return PASSAGE_PREFIXES.get(model_name, "")


class Embedder:
    """Batch text -> vector embedding using a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=model_name in TRUST_REMOTE_CODE_MODELS,
        )
        self._query_prefix = query_prefix(model_name)
        self._passage_prefix = passage_prefix(model_name)

    @property
    def dimension(self) -> int:
        size = self._model.get_embedding_dimension()
        assert size is not None
        return cast(int, size)

    def embed_passages(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        return cast(
            np.ndarray,
            self._model.encode(
                [self._passage_prefix + text for text in texts],
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
            ),
        )

    def embed_query(self, text: str) -> np.ndarray:
        result = self._model.encode([self._query_prefix + text], normalize_embeddings=True)
        return cast(np.ndarray, result[0])
