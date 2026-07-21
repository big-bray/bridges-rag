from bridges_rag.chunk.models import Chunk
from bridges_rag.eval.models import BenchmarkQuestion
from bridges_rag.eval.runner import evaluate
from bridges_rag.search.models import SearchResult


def _chunk(paper_id: str, chunk_index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{paper_id}-{chunk_index}",
        paper_id=paper_id,
        chunk_index=chunk_index,
        heading=None,
        text="text",
        pdf_pages=[1],
        proceedings_pages=[1],
        title="Title",
        authors=["Author"],
        year=2025,
    )


class _StubRetriever:
    """Fixed retrieval order per question, keyed by question text."""

    def __init__(self, results_by_question: dict[str, list[str]]) -> None:
        self._results_by_question = results_by_question

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        paper_ids = self._results_by_question[question][:top_k]
        return [
            SearchResult(chunk=_chunk(paper_id, i), score=1.0 - i * 0.01)
            for i, paper_id in enumerate(paper_ids)
        ]


def test_evaluate_aggregates_recall_and_mrr_over_questions():
    retriever = _StubRetriever(
        {
            "q1": ["a", "b", "c"],
            "q2": ["z", "b"],
        }
    )
    questions = [
        BenchmarkQuestion(question="q1", paper_ids=["a"]),
        BenchmarkQuestion(question="q2", paper_ids=["b"]),
    ]

    results = evaluate(retriever, questions)

    assert results.n_questions == 2
    assert results.recall_at_k[1] == 0.5  # q1 hits at rank 1, q2 doesn't
    assert results.recall_at_k[5] == 1.0
    assert results.mrr == (1.0 + 0.5) / 2
    assert results.avg_query_ms >= 0.0


def test_evaluate_deduplicates_repeated_papers_across_chunks():
    retriever = _StubRetriever({"q": ["a", "a", "b"]})
    questions = [BenchmarkQuestion(question="q", paper_ids=["b"])]

    results = evaluate(retriever, questions)

    # "b" is the second distinct paper once "a" is deduplicated, so MRR is 1/2.
    assert results.mrr == 0.5
