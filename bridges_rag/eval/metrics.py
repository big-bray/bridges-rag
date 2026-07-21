"""Recall@k and reciprocal rank over a ranked list of paper IDs."""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(ranked_paper_ids: Sequence[str], gold_paper_ids: set[str], k: int) -> float:
    """Fraction of gold papers present in the top-k of a ranked, deduplicated paper list."""
    if not gold_paper_ids:
        return 0.0
    retrieved = set(ranked_paper_ids[:k])
    return len(retrieved & gold_paper_ids) / len(gold_paper_ids)


def reciprocal_rank(ranked_paper_ids: Sequence[str], gold_paper_ids: set[str]) -> float:
    """1/rank of the first gold paper in the ranked list, or 0 if none appear."""
    for rank, paper_id in enumerate(ranked_paper_ids, start=1):
        if paper_id in gold_paper_ids:
            return 1 / rank
    return 0.0
