from pathlib import Path

from bridges_rag.graph.concepts import ExtractedTriple
from bridges_rag.graph.concepts_store import ChunkConcepts, read_concepts, write_concepts


def test_write_then_read_round_trips(tmp_path: Path):
    concepts = [
        ChunkConcepts(
            chunk_id="p1-0",
            paper_id="p1",
            entities=["weaving", "tiling"],
            triples=[ExtractedTriple(subject="weaving", relation="uses", object="strands")],
        ),
        ChunkConcepts(chunk_id="p2-0", paper_id="p2", entities=[], triples=[], error="failed"),
    ]
    path = tmp_path / "concepts.jsonl"

    write_concepts(concepts, path)
    result = read_concepts(path)

    assert result == concepts


def test_write_creates_parent_directory(tmp_path: Path):
    path = tmp_path / "nested" / "concepts.jsonl"
    write_concepts([], path)
    assert path.exists()


def test_read_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "concepts.jsonl"
    chunk = ChunkConcepts(chunk_id="p1-0", paper_id="p1", entities=[], triples=[])
    path.write_text(f"\n{chunk.model_dump_json()}\n\n", encoding="utf-8")

    assert read_concepts(path) == [chunk]
