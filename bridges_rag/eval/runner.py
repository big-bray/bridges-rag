"""Run the search pipeline over the benchmark and aggregate Recall@k and MRR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient

from bridges_rag.embed.embedder import Embedder
from bridges_rag.eval.metrics import recall_at_k, reciprocal_rank
from bridges_rag.eval.models import BenchmarkQuestion
from bridges_rag.index.qdrant import DEFAULT_COLLECTION
from bridges_rag.search.search import search

RECALL_KS = (1, 5, 10)
# Chunks fetched per query; deduplicated to papers before metrics are calculated
POOL_SIZE = 50


@dataclass
class EvalResults:
    n_questions: int
    recall_at_k: dict[int, float]
    mrr: float


def _ranked_paper_ids(
    client: QdrantClient,
    embedder: Embedder,
    question: BenchmarkQuestion,
    collection: str,
) -> list[str]:
    results = search(client, embedder, question.question, collection=collection, top_k=POOL_SIZE)
    ranked: list[str] = []
    seen: set[str] = set()
    for result in results:
        paper_id = result.chunk.paper_id
        if paper_id not in seen:
            seen.add(paper_id)
            ranked.append(paper_id)
    return ranked


def evaluate(
    client: QdrantClient,
    embedder: Embedder,
    questions: Sequence[BenchmarkQuestion],
    *,
    collection: str = DEFAULT_COLLECTION,
) -> EvalResults:
    recalls: dict[int, list[float]] = {k: [] for k in RECALL_KS}
    reciprocal_ranks: list[float] = []

    for question in questions:
        ranked = _ranked_paper_ids(client, embedder, question, collection)
        gold = set(question.paper_ids)
        for k in RECALL_KS:
            recalls[k].append(recall_at_k(ranked, gold, k))
        reciprocal_ranks.append(reciprocal_rank(ranked, gold))

    n = len(questions)
    return EvalResults(
        n_questions=n,
        recall_at_k={k: sum(vals) / n for k, vals in recalls.items()},
        mrr=sum(reciprocal_ranks) / n,
    )


def format_results_table(results: EvalResults) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    for k in sorted(results.recall_at_k):
        lines.append(f"| Recall@{k} | {results.recall_at_k[k]:.2f} |")
    lines.append(f"| MRR | {results.mrr:.3f} |")
    return "\n".join(lines)
