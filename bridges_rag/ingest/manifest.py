"""Read/write the JSONL manifest of scraped paper metadata."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from bridges_rag.ingest.models import Paper


def write_manifest(papers: Iterable[Paper], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(paper.model_dump_json())
            f.write("\n")


def read_manifest(path: Path) -> list[Paper]:
    papers: list[Paper] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(Paper.model_validate_json(line))
    return papers
