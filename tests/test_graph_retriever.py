from bridges_rag.chunk.models import Chunk
from bridges_rag.graph.retriever import (
    GraphRetriever,
    combine_rankings,
    dedup_paper_ids,
    link_authors,
    link_titles,
    rrf_fuse,
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
        self.last_args: tuple[list[str], list[str]] | None = None

    def __call__(self, authors: list[str], titles: list[str]) -> list[str]:
        self.last_args = (authors, titles)
        return self.ranking


def test_link_authors_matches_case_insensitively():
    assert link_authors("What did Ada Lovelace work on?", ["Ada Lovelace", "Alan Turing"]) == [
        "Ada Lovelace"
    ]


def test_link_authors_no_match_returns_empty():
    assert link_authors("papers about tilings", ["Ada Lovelace"]) == []


def test_link_authors_orders_longest_first():
    names = ["Ada Lovelace", "Ada Lovelace Jr"]
    matches = link_authors("ada lovelace jr wrote this", names)
    assert matches[0] == "Ada Lovelace Jr"


def test_link_titles_matches_case_insensitively():
    titles = ["Modular Mayhem", "Transforming 2D Materials"]
    assert link_titles("tell me about Modular Mayhem", titles) == ["Modular Mayhem"]


def test_combine_rankings_direct_and_title_outrank_indirect():
    combined = combine_rankings(["p1"], ["p2"], ["p1", "p3"])
    assert combined == ["p1", "p2", "p3"]


def test_rrf_fuse_prefers_paper_ranked_high_in_both_lists():
    fused = rrf_fuse([["p1", "p2", "p3"], ["p2", "p1", "p4"]])
    assert fused[0] in {"p1", "p2"}
    assert set(fused) == {"p1", "p2", "p3", "p4"}


def test_rrf_fuse_single_ranking_preserves_order():
    assert rrf_fuse([["p1", "p2", "p3"]]) == ["p1", "p2", "p3"]


def test_dedup_paper_ids_preserves_first_occurrence_order():
    results = [_result("p1", 0.9), _result("p2", 0.8), _result("p1", 0.7)]
    assert dedup_paper_ids(results) == ["p1", "p2"]


def test_graph_retriever_falls_back_to_base_when_nothing_links():
    base = _StubBaseRetriever([_result("p1", 0.9), _result("p2", 0.8)])
    traversal = _StubTraversal([])
    retriever = GraphRetriever(
        base=base,
        traverse=traversal,
        author_names=["Ada Lovelace"],
        titles=[],
        chunk_by_paper={},
    )

    results = retriever.retrieve("query with no linked entities", top_k=2)

    assert [r.chunk.paper_id for r in results] == ["p1", "p2"]


def test_graph_retriever_fuses_graph_only_hit_using_representative_chunk():
    base = _StubBaseRetriever([_result("p1", 0.9)])
    traversal = _StubTraversal(["p2", "p1"])
    retriever = GraphRetriever(
        base=base,
        traverse=traversal,
        author_names=["Ada Lovelace"],
        titles=[],
        chunk_by_paper={"p2": _chunk("p2")},
    )

    results = retriever.retrieve("Ada Lovelace collaborators", top_k=5)

    paper_ids = [r.chunk.paper_id for r in results]
    assert "p2" in paper_ids
    assert "p1" in paper_ids


def test_graph_retriever_drops_graph_only_hit_without_a_representative_chunk():
    base = _StubBaseRetriever([_result("p1", 0.9)])
    traversal = _StubTraversal(["p3", "p1"])
    retriever = GraphRetriever(
        base=base,
        traverse=traversal,
        author_names=["Ada Lovelace"],
        titles=[],
        chunk_by_paper={},
    )

    results = retriever.retrieve("Ada Lovelace collaborators", top_k=5)

    assert [r.chunk.paper_id for r in results] == ["p1"]


def test_graph_retriever_requests_pool_size_from_base():
    base = _StubBaseRetriever([_result(f"p{i}", 1.0 - i * 0.01) for i in range(5)])
    traversal = _StubTraversal([])
    retriever = GraphRetriever(
        base=base,
        traverse=traversal,
        author_names=[],
        titles=[],
        chunk_by_paper={},
        pool_size=5,
    )

    retriever.retrieve("query", top_k=2)

    assert base.last_top_k == 5
