"""Read/write the JSONL file of chunks produced by the chunker."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from bridges_rag.chunk.models import Chunk


def write_chunks(chunks: Iterable[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json())
            f.write("\n")


def read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(Chunk.model_validate_json(line))
    return chunks
