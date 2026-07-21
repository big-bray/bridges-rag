"""Qdrant collection setup and chunk upserts."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from bridges_rag.chunk.models import Chunk

DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "bridges_papers"

# Named vectors: every collection has a dense vector; sparse (BM25, for hybrid
# search) is opt-in per collection via ensure_collection(..., with_sparse=True).
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Payload fields that support metadata filtering (author, title, year).
_KEYWORD_INDEX_FIELDS = ("authors", "title")
_INTEGER_INDEX_FIELDS = ("year",)


def get_client(url: str = DEFAULT_URL) -> QdrantClient:
    return QdrantClient(url=url)


def collection_for(model_name: str) -> str:
    slug = model_name.rsplit("/", 1)[-1].lower()
    return f"{DEFAULT_COLLECTION}__{slug}"


def ensure_collection(
    client: QdrantClient,
    collection: str,
    *,
    vector_size: int,
    with_sparse: bool = False,
) -> None:
    """Create the collection and its payload indexes if they don't already exist."""
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config={
                DENSE_VECTOR_NAME: qmodels.VectorParams(
                    size=vector_size, distance=qmodels.Distance.COSINE
                ),
            },
            sparse_vectors_config=(
                {SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()} if with_sparse else None
            ),
        )
        for field in _KEYWORD_INDEX_FIELDS:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.KEYWORD,
            )
        for field in _INTEGER_INDEX_FIELDS:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=qmodels.PayloadSchemaType.INTEGER,
            )


def chunk_point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be a uint64 or UUID; derive a stable UUID from the chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def chunk_payload(chunk: Chunk, *, embedding_model: str) -> dict[str, object]:
    return {
        **chunk.model_dump(),
        "embedding_model": embedding_model,
    }


def point_vectors(
    vectors: Sequence[Sequence[float]],
    sparse_vectors: Sequence[qmodels.SparseVector] | None = None,
) -> list[dict[str, qmodels.Vector]]:
    """Build the named-vector payload for each point: dense, plus sparse if given."""
    if sparse_vectors is None:
        return [{DENSE_VECTOR_NAME: list(vector)} for vector in vectors]
    return [
        {DENSE_VECTOR_NAME: list(vector), SPARSE_VECTOR_NAME: sparse_vector}
        for vector, sparse_vector in zip(vectors, sparse_vectors, strict=True)
    ]


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    *,
    embedding_model: str,
    sparse_vectors: Sequence[qmodels.SparseVector] | None = None,
) -> None:
    points = [
        qmodels.PointStruct(
            id=chunk_point_id(chunk.chunk_id),
            vector=vector,
            payload=chunk_payload(chunk, embedding_model=embedding_model),
        )
        for chunk, vector in zip(chunks, point_vectors(vectors, sparse_vectors), strict=True)
    ]
    client.upsert(collection_name=collection, points=points)
