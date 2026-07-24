"""Run a Retriever over the benchmark and aggregate Recall@k, MRR, and latency."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

from bridges_rag.eval.metrics import recall_at_k, reciprocal_rank
from bridges_rag.eval.models import BenchmarkQuestion
from bridges_rag.search.retriever import Retriever

RECALL_KS = (1, 5, 10)
# Chunks fetched per query; deduplicated to papers before metrics are calculated
POOL_SIZE = 50


@dataclass
class EvalResults:
    n_questions: int
    recall_at_k: dict[int, float]
    mrr: float
    avg_query_ms: float


def _ranked_paper_ids(retriever: Retriever, question: BenchmarkQuestion) -> tuple[list[str], float]:
    start = time.perf_counter()
    results = retriever.retrieve(question.question, top_k=POOL_SIZE)
    elapsed_ms = (time.perf_counter() - start) * 1000

    ranked: list[str] = []
    seen: set[str] = set()
    for result in results:
        paper_id = result.chunk.paper_id
        if paper_id not in seen:
            seen.add(paper_id)
            ranked.append(paper_id)
    return ranked, elapsed_ms


def evaluate(retriever: Retriever, questions: Sequence[BenchmarkQuestion]) -> EvalResults:
    recalls: dict[int, list[float]] = {k: [] for k in RECALL_KS}
    reciprocal_ranks: list[float] = []
    query_ms: list[float] = []

    for question in questions:
        ranked, elapsed_ms = _ranked_paper_ids(retriever, question)
        query_ms.append(elapsed_ms)
        gold = set(question.paper_ids)
        for k in RECALL_KS:
            recalls[k].append(recall_at_k(ranked, gold, k))
        reciprocal_ranks.append(reciprocal_rank(ranked, gold))

    n = len(questions)
    return EvalResults(
        n_questions=n,
        recall_at_k={k: sum(vals) / n for k, vals in recalls.items()},
        mrr=sum(reciprocal_ranks) / n,
        avg_query_ms=sum(query_ms) / n,
    )


def format_results_table(results: EvalResults) -> str:
    lines = ["| Metric | Value |", "|---|---|"]
    for k in sorted(results.recall_at_k):
        lines.append(f"| Recall@{k} | {results.recall_at_k[k]:.2f} |")
    lines.append(f"| MRR | {results.mrr:.3f} |")
    lines.append(f"| Avg query (ms) | {results.avg_query_ms:.0f} |")
    return "\n".join(lines)
