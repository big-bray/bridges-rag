"""Read/write the JSONL file of per-chunk concept extractions."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from bridges_rag.graph.concepts import ExtractedTriple


class ChunkConcepts(BaseModel):
    """Concept entities + relation triples extracted from one chunk, joined to its paper."""

    chunk_id: str
    paper_id: str
    entities: list[str]
    triples: list[ExtractedTriple]
    error: str | None = None


def write_concepts(concepts: Iterable[ChunkConcepts], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk_concepts in concepts:
            f.write(chunk_concepts.model_dump_json())
            f.write("\n")


def read_concepts(path: Path) -> list[ChunkConcepts]:
    concepts: list[ChunkConcepts] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                concepts.append(ChunkConcepts.model_validate_json(line))
    return concepts
