"""Read the JSONL file of hand-labeled benchmark questions."""

from __future__ import annotations

from pathlib import Path

from bridges_rag.eval.models import BenchmarkQuestion


def load_benchmark(path: Path) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(BenchmarkQuestion.model_validate_json(line))
    return questions
