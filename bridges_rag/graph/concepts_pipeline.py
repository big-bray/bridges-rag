"""Orchestrates LLM concept extraction over a year's chunks, with on-disk caching."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from bridges_rag.chunk.models import Chunk
from bridges_rag.graph.concepts import ConceptExtractor, ExtractedConcepts
from bridges_rag.graph.concepts_store import ChunkConcepts, read_concepts, write_concepts
from bridges_rag.index.pipeline import load_chunks

logger = logging.getLogger(__name__)


def extract_chunk_concepts(chunk: Chunk, extractor: ConceptExtractor) -> ChunkConcepts:
    try:
        extracted: ExtractedConcepts = extractor.extract(chunk.text)
    except Exception as exc:
        logger.exception("failed to extract concepts for chunk %s", chunk.chunk_id)
        return ChunkConcepts(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            entities=[],
            triples=[],
            error=str(exc),
        )
    return ChunkConcepts(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        entities=extracted.entities,
        triples=extracted.triples,
    )


def extract_concepts_for_chunks(
    chunks: Sequence[Chunk], extractor: ConceptExtractor
) -> list[ChunkConcepts]:
    total = len(chunks)
    results = []
    for i, chunk in enumerate(chunks, start=1):
        start = time.monotonic()
        concepts = extract_chunk_concepts(chunk, extractor)
        elapsed = time.monotonic() - start
        status = "ok" if concepts.error is None else f"error: {concepts.error}"
        print(f"[{i}/{total}] {chunk.chunk_id} ({elapsed:.1f}s) {status}", flush=True)
        results.append(concepts)
    return results


def extract_concepts_year(
    year: int,
    data_dir: Path,
    extractor: ConceptExtractor,
    *,
    limit: int | None = None,
    force: bool = False,
) -> list[ChunkConcepts]:
    """Read a year's concepts.jsonl if present (chunking/extracting on demand otherwise).

    `--limit` only applies when (re-)extracting; a cached file is used as-is.
    """
    concepts_path = data_dir / str(year) / "concepts.jsonl"
    if concepts_path.exists() and not force:
        return read_concepts(concepts_path)

    chunks = load_chunks(year, data_dir)
    if limit is not None:
        chunks = chunks[:limit]

    concepts = extract_concepts_for_chunks(chunks, extractor)
    write_concepts(concepts, concepts_path)
    return concepts
