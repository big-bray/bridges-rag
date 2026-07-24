from pathlib import Path

from bridges_rag.chunk.models import Chunk
from bridges_rag.chunk.store import write_chunks
from bridges_rag.graph.concepts import ExtractedConcepts, ExtractedTriple
from bridges_rag.graph.concepts_pipeline import (
    extract_chunk_concepts,
    extract_concepts_for_chunks,
    extract_concepts_year,
)
from bridges_rag.graph.concepts_store import read_concepts


def _chunk(chunk_id: str, paper_id: str = "p1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        chunk_index=0,
        heading=None,
        text="some text about weaving",
        pdf_pages=[1],
        proceedings_pages=[1],
        title="Title",
        authors=["Author"],
        year=2025,
    )


class _StubExtractor:
    def __init__(self, result: ExtractedConcepts | Exception) -> None:
        self.result = result
        self.calls: list[str] = []

    def extract(self, text: str) -> ExtractedConcepts:
        self.calls.append(text)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_extract_chunk_concepts_success():
    extracted = ExtractedConcepts(
        entities=["weaving"],
        triples=[ExtractedTriple(subject="weaving", relation="uses", object="strands")],
    )
    extractor = _StubExtractor(extracted)

    result = extract_chunk_concepts(_chunk("p1-0"), extractor)

    assert result.chunk_id == "p1-0"
    assert result.paper_id == "p1"
    assert result.entities == ["weaving"]
    assert result.error is None


def test_extract_chunk_concepts_isolates_failure():
    extractor = _StubExtractor(RuntimeError("model unavailable"))

    result = extract_chunk_concepts(_chunk("p1-0"), extractor)

    assert result.entities == []
    assert result.triples == []
    assert result.error == "model unavailable"


def test_extract_concepts_for_chunks_continues_after_a_failure():
    class _FlakyExtractor:
        def __init__(self) -> None:
            self.calls = 0

        def extract(self, text: str) -> ExtractedConcepts:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return ExtractedConcepts(entities=["weaving"], triples=[])

    extractor = _FlakyExtractor()
    chunks = [_chunk("p1-0"), _chunk("p1-1")]

    results = extract_concepts_for_chunks(chunks, extractor)

    assert [r.error for r in results] == ["boom", None]
    assert results[1].entities == ["weaving"]
    assert extractor.calls == 2


def test_extract_concepts_year_writes_and_caches(tmp_path: Path):
    year = 2025
    write_chunks([_chunk("p1-0"), _chunk("p1-1")], tmp_path / str(year) / "chunks.jsonl")
    extractor = _StubExtractor(ExtractedConcepts(entities=["weaving"], triples=[]))

    first = extract_concepts_year(year, tmp_path, extractor, limit=None)
    assert len(first) == 2
    assert (tmp_path / str(year) / "concepts.jsonl").exists()
    assert extractor.calls == ["some text about weaving", "some text about weaving"]

    second = extract_concepts_year(year, tmp_path, extractor)
    assert second == first
    # cached: no additional extraction calls on the second run
    assert len(extractor.calls) == 2


def test_extract_concepts_year_respects_limit(tmp_path: Path):
    year = 2025
    write_chunks(
        [_chunk("p1-0"), _chunk("p1-1"), _chunk("p1-2")], tmp_path / str(year) / "chunks.jsonl"
    )
    extractor = _StubExtractor(ExtractedConcepts(entities=[], triples=[]))

    results = extract_concepts_year(year, tmp_path, extractor, limit=1)

    assert len(results) == 1
    assert read_concepts(tmp_path / str(year) / "concepts.jsonl") == results


def test_extract_concepts_year_force_re_extracts(tmp_path: Path):
    year = 2025
    write_chunks([_chunk("p1-0")], tmp_path / str(year) / "chunks.jsonl")
    extractor = _StubExtractor(ExtractedConcepts(entities=[], triples=[]))

    extract_concepts_year(year, tmp_path, extractor)
    assert len(extractor.calls) == 1

    extract_concepts_year(year, tmp_path, extractor, force=True)
    assert len(extractor.calls) == 2
