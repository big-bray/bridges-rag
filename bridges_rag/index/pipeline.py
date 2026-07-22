"""Orchestrates embedding and indexing a year's chunks into Qdrant."""

from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from bridges_rag.chunk.chunker import chunk_year
from bridges_rag.chunk.models import Chunk
from bridges_rag.chunk.store import read_chunks
from bridges_rag.embed.embedder import DEFAULT_MODEL_NAME, Embedder
from bridges_rag.index.qdrant import (
    DEFAULT_COLLECTION,
    DEFAULT_URL,
    ensure_collection,
    get_client,
    upsert_chunks,
)


def load_chunks(year: int, data_dir: Path) -> list[Chunk]:
    """Read a year's chunks.jsonl, chunking on demand if it doesn't exist yet."""
    chunks_path = data_dir / str(year) / "chunks.jsonl"
    if chunks_path.exists():
        return read_chunks(chunks_path)
    return chunk_year(year, data_dir)


def index_chunks(
    chunks: list[Chunk],
    embedder: Embedder,
    client: QdrantClient,
    *,
    collection: str = DEFAULT_COLLECTION,
    batch_size: int = 64,
) -> int:
    """Embed and upsert `chunks` into `collection`, sized for `embedder`."""
    if not chunks:
        return 0

    ensure_collection(client, collection, vector_size=embedder.dimension)

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed_passages([c.text for c in batch]).tolist()
        upsert_chunks(client, collection, batch, vectors, embedding_model=embedder.model_name)
        print(f"[{min(start + batch_size, total)}/{total}] embedded & indexed", flush=True)

    return total


def index_year(
    year: int,
    data_dir: Path,
    *,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
    batch_size: int = 64,
) -> int:
    chunks = load_chunks(year, data_dir)
    embedder = Embedder(model_name, device=device)
    client = get_client(qdrant_url)
    return index_chunks(chunks, embedder, client, collection=collection, batch_size=batch_size)
