from bridges_rag.graph.build_concepts import (
    build_concept_graph,
    ensure_concept_constraint,
    normalize_concept,
)
from bridges_rag.graph.concepts import ExtractedTriple
from bridges_rag.graph.concepts_store import ChunkConcepts


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
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def session(self) -> _FakeSession:
        return _FakeSession(self.calls)


def _cc(
    chunk_id: str,
    paper_id: str,
    entities: list[str],
    triples: list[ExtractedTriple] | None = None,
    error: str | None = None,
) -> ChunkConcepts:
    return ChunkConcepts(
        chunk_id=chunk_id, paper_id=paper_id, entities=entities, triples=triples or [], error=error
    )


def test_normalize_concept_collapses_case_and_whitespace():
    assert normalize_concept("Hyperbolic   Tiling") == "hyperbolic tiling"
    assert normalize_concept("  weaving ") == "weaving"


def test_ensure_concept_constraint_issues_one_statement():
    driver = _FakeDriver()
    ensure_concept_constraint(driver)
    assert len(driver.calls) == 1
    assert "CONSTRAINT" in driver.calls[0][0]
    assert "Concept" in driver.calls[0][0]


def test_build_concept_graph_skips_failed_chunks():
    chunk_concepts = [_cc("p1-0", "p1", ["weaving"], error="model unavailable")]
    driver = _FakeDriver()

    stats = build_concept_graph(chunk_concepts, driver)

    assert stats.chunks_failed == 1
    assert stats.chunks_processed == 0
    assert stats.concepts == 0
    assert stats.mentions_edges == 0


def test_build_concept_graph_counts_mentions_and_related_to():
    chunk_concepts = [
        _cc(
            "p1-0",
            "p1",
            ["Weaving", "Tiling"],
            [ExtractedTriple(subject="Weaving", relation="uses", object="strands")],
        ),
    ]
    driver = _FakeDriver()

    stats = build_concept_graph(chunk_concepts, driver)

    assert stats.chunks_processed == 1
    assert stats.mentions_edges == 2
    assert stats.related_to_edges == 1
    # weaving, tiling, strands -> 3 distinct concepts
    assert stats.concepts == 3


def test_build_concept_graph_dedupes_case_variants_within_a_chunk():
    chunk_concepts = [_cc("p1-0", "p1", ["Weaving", "weaving", " WEAVING "])]
    driver = _FakeDriver()

    stats = build_concept_graph(chunk_concepts, driver)

    assert stats.concepts == 1


def test_build_concept_graph_skips_blank_entities_and_triples():
    chunk_concepts = [
        _cc(
            "p1-0",
            "p1",
            ["  ", "weaving"],
            [ExtractedTriple(subject="  ", relation="uses", object="x")],
        )
    ]
    driver = _FakeDriver()

    stats = build_concept_graph(chunk_concepts, driver)

    assert stats.mentions_edges == 1
    assert stats.related_to_edges == 0


def test_build_concept_graph_calls_ensure_constraint_first():
    driver = _FakeDriver()
    build_concept_graph([_cc("p1-0", "p1", ["weaving"])], driver)
    assert "CONSTRAINT" in driver.calls[0][0]
