"""Qdrant collection setup and chunk upserts."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from bridges_rag.chunk.models import Chunk

DEFAULT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "bridges_papers"

# Payload fields that support metadata filtering (author, title, year).
_KEYWORD_INDEX_FIELDS = ("authors", "title")
_INTEGER_INDEX_FIELDS = ("year",)


def get_client(url: str = DEFAULT_URL) -> QdrantClient:
    return QdrantClient(url=url)


def collection_for(model_name: str) -> str:
    """Per-model collection name, e.g. 'bge-large-en-v1.5' -> 'bridges_papers__bge-large-en-v1.5'.

    Lets the sweep runner build a separate collection per candidate embedding model.
    """
    slug = model_name.rsplit("/", 1)[-1].lower()
    return f"{DEFAULT_COLLECTION}__{slug}"


def ensure_collection(client: QdrantClient, collection: str, *, vector_size: int) -> None:
    """Create the collection and its payload indexes if they don't already exist."""
    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
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


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    *,
    embedding_model: str,
) -> None:
    points = [
        qmodels.PointStruct(
            id=chunk_point_id(chunk.chunk_id),
            vector=list(vector),
            payload=chunk_payload(chunk, embedding_model=embedding_model),
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]
    client.upsert(collection_name=collection, points=points)
