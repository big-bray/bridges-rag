from bridges_rag.graph.build import build_graph, clear_graph, ensure_constraints
from bridges_rag.ingest.models import Paper


class _FakeSession:
    def __init__(self, calls: list[tuple[str, dict[str, object]]]) -> None:
        self._calls = calls

    def run(self, query: str, **kwargs: object) -> None:
        self._calls.append((query, kwargs))

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _FakeDriver:
    """Records every Cypher statement issued, without needing a real Neo4j instance."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def session(self) -> _FakeSession:
        return _FakeSession(self.calls)


def _paper(paper_id: str, authors: list[str], *, year: int = 2025, title: str = "") -> Paper:
    return Paper(
        paper_id=paper_id,
        year=year,
        title=title or paper_id,
        authors=authors,
        category="Papers",
        pdf_url=f"https://example.org/{paper_id}.pdf",
    )


def test_ensure_constraints_issues_three_statements():
    driver = _FakeDriver()
    ensure_constraints(driver)
    assert len(driver.calls) == 3
    assert all("CONSTRAINT" in query for query, _ in driver.calls)


def test_clear_graph_detaches_and_deletes():
    driver = _FakeDriver()
    clear_graph(driver)
    assert len(driver.calls) == 1
    assert "DETACH DELETE" in driver.calls[0][0]


def test_build_graph_counts_papers_authors_years():
    papers = [
        _paper("p1", ["Ada Lovelace"], year=2024),
        _paper("p2", ["Ada Lovelace", "Alan Turing"], year=2025),
        _paper("p3", ["Grace Hopper"], year=2025),
    ]
    driver = _FakeDriver()

    stats = build_graph(papers, driver)

    assert stats.papers == 3
    assert stats.authors == 3
    assert stats.years == 2


def test_build_graph_counts_co_authored_pairs_only_for_multi_author_papers():
    papers = [
        _paper("p1", ["Ada Lovelace"]),
        _paper("p2", ["Ada Lovelace", "Alan Turing", "Grace Hopper"]),
    ]
    driver = _FakeDriver()

    stats = build_graph(papers, driver)

    # combinations(3, 2) == 3 pairs for the one multi-author paper
    assert stats.co_authored_pairs == 3


def test_build_graph_skips_papers_with_no_authors():
    papers = [_paper("p1", [])]
    driver = _FakeDriver()

    build_graph(papers, driver)

    # only the constraint setup runs; no per-paper MERGE for an authorless paper
    assert len(driver.calls) == 3


def test_build_graph_calls_ensure_constraints_first():
    papers = [_paper("p1", ["Ada Lovelace"])]
    driver = _FakeDriver()

    build_graph(papers, driver)

    assert "CONSTRAINT" in driver.calls[0][0]
