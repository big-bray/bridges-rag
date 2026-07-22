"""Sweep candidate embedding models over the benchmark, reporting quality and cost."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from huggingface_hub import scan_cache_dir
from qdrant_client import QdrantClient

from bridges_rag.chunk.models import Chunk
from bridges_rag.embed.embedder import Embedder
from bridges_rag.eval.models import BenchmarkQuestion
from bridges_rag.eval.runner import evaluate
from bridges_rag.index.pipeline import index_chunks
from bridges_rag.index.qdrant import collection_for
from bridges_rag.search.retriever import DenseRetriever

CANDIDATE_MODELS = (
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "thenlper/gte-large",
    "intfloat/e5-base-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2",
)

BYTES_PER_FLOAT32 = 4


@dataclass
class SweepRow:
    model_name: str
    dimension: int
    recall_at_k: dict[int, float]
    mrr: float
    embed_seconds: float
    avg_query_ms: float
    model_disk_mb: float | None
    vectors_mb: float


def model_disk_usage_bytes(model_name: str) -> int | None:
    """Bytes on disk for `model_name` in the local Hugging Face cache."""
    for repo in scan_cache_dir().repos:
        if repo.repo_id == model_name:
            return repo.size_on_disk
    return None


def estimated_vector_bytes(n_vectors: int, dimension: int) -> int:
    """Raw float32 vector storage, ignoring Qdrant's HNSW index/payload overhead."""
    return n_vectors * dimension * BYTES_PER_FLOAT32


def run_sweep(
    chunks: Sequence[Chunk],
    questions: Sequence[BenchmarkQuestion],
    client: QdrantClient,
    *,
    models: Sequence[str] = CANDIDATE_MODELS,
    device: str | None = None,
    batch_size: int = 64,
) -> list[SweepRow]:
    rows = []
    for model_name in models:
        embedder = Embedder(model_name, device=device)
        collection = collection_for(model_name)

        embed_start = time.perf_counter()
        index_chunks(list(chunks), embedder, client, collection=collection, batch_size=batch_size)
        embed_seconds = time.perf_counter() - embed_start

        retriever = DenseRetriever(client, embedder, collection=collection)
        eval_start = time.perf_counter()
        results = evaluate(retriever, questions)
        eval_seconds = time.perf_counter() - eval_start

        rows.append(
            SweepRow(
                model_name=model_name,
                dimension=embedder.dimension,
                recall_at_k=results.recall_at_k,
                mrr=results.mrr,
                embed_seconds=embed_seconds,
                avg_query_ms=eval_seconds / len(questions) * 1000,
                model_disk_mb=_bytes_to_mb(model_disk_usage_bytes(model_name)),
                vectors_mb=estimated_vector_bytes(len(chunks), embedder.dimension) / (1024 * 1024),
            )
        )
    return rows


def _bytes_to_mb(n: int | None) -> float | None:
    return None if n is None else n / (1024 * 1024)


def format_sweep_table(rows: Sequence[SweepRow]) -> str:
    header = (
        "| Model | Dim | Recall@1 | Recall@5 | Recall@10 | MRR "
        "| Embed (s) | Query (ms) | Weights (MB) | Vectors (MB) |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, separator]
    for row in rows:
        weights = f"{row.model_disk_mb:.0f}" if row.model_disk_mb is not None else "?"
        lines.append(
            f"| {row.model_name} | {row.dimension} "
            f"| {row.recall_at_k.get(1, 0.0):.2f} "
            f"| {row.recall_at_k.get(5, 0.0):.2f} "
            f"| {row.recall_at_k.get(10, 0.0):.2f} "
            f"| {row.mrr:.3f} "
            f"| {row.embed_seconds:.1f} "
            f"| {row.avg_query_ms:.0f} "
            f"| {weights} "
            f"| {row.vectors_mb:.1f} |"
        )
    return "\n".join(lines)
