from bridges_rag.chunk.models import Chunk
from bridges_rag.search.models import SearchResult
from bridges_rag.search.rerank import RerankRetriever


def _result(paper_id: str, score: float) -> SearchResult:
    chunk = Chunk(
        chunk_id=f"{paper_id}-0",
        paper_id=paper_id,
        chunk_index=0,
        heading=None,
        text=paper_id,
        pdf_pages=[1],
        proceedings_pages=[1],
        title="Title",
        authors=["Author"],
        year=2025,
    )
    return SearchResult(chunk=chunk, score=score)


class _StubBaseRetriever:
    """Records the top_k it was called with and returns a fixed pool."""

    def __init__(self, pool: list[SearchResult]) -> None:
        self.pool = pool
        self.last_top_k: int | None = None

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        self.last_top_k = top_k
        return self.pool[:top_k]


class _StubReranker:
    """Reverses the candidate order, simulating a reranker that disagrees with the base."""

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int) -> list[SearchResult]:
        return list(reversed(results))[:top_k]


def test_rerank_retriever_requests_at_least_pool_size_from_base():
    base = _StubBaseRetriever([_result(f"p{i}", 1.0 - i * 0.01) for i in range(5)])
    retriever = RerankRetriever(base, _StubReranker(), pool_size=5)

    retriever.retrieve("query", top_k=2)

    assert base.last_top_k == 5


def test_rerank_retriever_requests_top_k_when_larger_than_pool_size():
    base = _StubBaseRetriever([_result(f"p{i}", 1.0 - i * 0.01) for i in range(10)])
    retriever = RerankRetriever(base, _StubReranker(), pool_size=3)

    retriever.retrieve("query", top_k=8)

    assert base.last_top_k == 8


def test_rerank_retriever_applies_reranker_and_truncates():
    base = _StubBaseRetriever([_result("a", 0.9), _result("b", 0.8), _result("c", 0.7)])
    retriever = RerankRetriever(base, _StubReranker(), pool_size=3)

    results = retriever.retrieve("query", top_k=2)

    assert [r.chunk.paper_id for r in results] == ["c", "b"]
