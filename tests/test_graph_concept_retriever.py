from bridges_rag.chunk.models import Chunk
from bridges_rag.graph.concept_retriever import (
    ConceptGraphRetriever,
    fetch_concept_names,
    link_concepts,
)
from bridges_rag.search.models import SearchResult


def _chunk(paper_id: str) -> Chunk:
    return Chunk(
        chunk_id=f"{paper_id}-0",
        paper_id=paper_id,
        chunk_index=0,
        heading=None,
        text=paper_id,
        pdf_pages=[1],
        proceedings_pages=[1],
        title=f"Title of {paper_id}",
        authors=["Author"],
        year=2025,
    )


def _result(paper_id: str, score: float) -> SearchResult:
    return SearchResult(chunk=_chunk(paper_id), score=score)


class _StubBaseRetriever:
    def __init__(self, pool: list[SearchResult]) -> None:
        self.pool = pool
        self.last_top_k: int | None = None

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        self.last_top_k = top_k
        return self.pool[:top_k]


class _StubTraversal:
    def __init__(self, ranking: list[str]) -> None:
        self.ranking = ranking
        self.last_concepts: list[str] | None = None

    def __call__(self, concepts: list[str]) -> list[str]:
        self.last_concepts = concepts
        return self.ranking


class _FakeSession:
    def __init__(self, names: list[str | None]) -> None:
        self._names = names

    def run(self, query: str, **kwargs: object) -> "_FakeSession":
        return self

    def single(self) -> dict[str, list[str | None]]:
        return {"names": self._names}

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeDriver:
    def __init__(self, names: list[str | None]) -> None:
        self._names = names

    def session(self) -> _FakeSession:
        return _FakeSession(self._names)


def test_link_concepts_matches_case_insensitively():
    concepts = ["hyperbolic tiling", "modular origami"]
    assert link_concepts("tell me about Hyperbolic Tiling", concepts) == ["hyperbolic tiling"]


def test_link_concepts_no_match_returns_empty():
    assert link_concepts("papers about weaving", ["hyperbolic tiling"]) == []


def test_link_concepts_orders_longest_first():
    concepts = ["tiling", "hyperbolic tiling"]
    matches = link_concepts("hyperbolic tiling patterns", concepts)
    assert matches[0] == "hyperbolic tiling"


def test_fetch_concept_names_filters_blanks():
    driver = _FakeDriver(["weaving", None, "", "tiling"])
    assert fetch_concept_names(driver) == ["weaving", "tiling"]


def test_concept_graph_retriever_falls_back_to_base_when_nothing_links():
    base = _StubBaseRetriever([_result("p1", 0.9), _result("p2", 0.8)])
    traversal = _StubTraversal([])
    retriever = ConceptGraphRetriever(
        base=base,
        traverse=traversal,
        concept_names=["hyperbolic tiling"],
        chunk_by_paper={},
    )

    results = retriever.retrieve("query with no linked concepts", top_k=2)

    assert [r.chunk.paper_id for r in results] == ["p1", "p2"]


def test_concept_graph_retriever_fuses_concept_only_hit_using_representative_chunk():
    base = _StubBaseRetriever([_result("p1", 0.9)])
    traversal = _StubTraversal(["p2", "p1"])
    retriever = ConceptGraphRetriever(
        base=base,
        traverse=traversal,
        concept_names=["hyperbolic tiling"],
        chunk_by_paper={"p2": _chunk("p2")},
    )

    results = retriever.retrieve("hyperbolic tiling papers", top_k=5)

    paper_ids = [r.chunk.paper_id for r in results]
    assert "p1" in paper_ids
    assert "p2" in paper_ids


def test_concept_graph_retriever_passes_linked_concepts_to_traversal():
    base = _StubBaseRetriever([_result("p1", 0.9)])
    traversal = _StubTraversal([])
    retriever = ConceptGraphRetriever(
        base=base,
        traverse=traversal,
        concept_names=["hyperbolic tiling", "weaving"],
        chunk_by_paper={},
    )

    retriever.retrieve("a question about hyperbolic tiling", top_k=5)

    assert traversal.last_concepts == ["hyperbolic tiling"]


def test_concept_graph_retriever_requests_pool_size_from_base():
    base = _StubBaseRetriever([_result(f"p{i}", 1.0 - i * 0.01) for i in range(5)])
    traversal = _StubTraversal([])
    retriever = ConceptGraphRetriever(
        base=base,
        traverse=traversal,
        concept_names=[],
        chunk_by_paper={},
        pool_size=5,
    )

    retriever.retrieve("query", top_k=2)

    assert base.last_top_k == 5
