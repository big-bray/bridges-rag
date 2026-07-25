"""Read/write the JSONL manifest of figures produced by figure extraction."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from bridges_rag.figures.models import Figure


def write_figures(figures: Iterable[Figure], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for figure in figures:
            f.write(figure.model_dump_json())
            f.write("\n")


def read_figures(path: Path) -> list[Figure]:
    figures: list[Figure] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                figures.append(Figure.model_validate_json(line))
    return figures
