"""Pydantic model for a hand-labeled benchmark question."""

from __future__ import annotations

from pydantic import BaseModel


class BenchmarkQuestion(BaseModel):
    """A query paired with the paper-level gold labels that should be retrieved."""

    question: str
    paper_ids: list[str]
