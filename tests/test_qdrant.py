from bridges_rag.chunk.models import Chunk
from bridges_rag.index.qdrant import chunk_payload, chunk_point_id, collection_for


def _chunk() -> Chunk:
    return Chunk(
        chunk_id="bridges2025-1-0000",
        paper_id="bridges2025-1",
        chunk_index=0,
        heading="# Intro",
        text="# Intro\n\nSome text.",
        pdf_pages=[1],
        proceedings_pages=[29],
        title="A Paper Title",
        authors=["Ada Lovelace"],
        year=2025,
    )


def test_chunk_point_id_is_stable_and_deterministic():
    assert chunk_point_id("bridges2025-1-0000") == chunk_point_id("bridges2025-1-0000")
    assert chunk_point_id("bridges2025-1-0000") != chunk_point_id("bridges2025-1-0001")


def test_chunk_payload_includes_chunk_fields_and_embedding_model():
    payload = chunk_payload(_chunk(), embedding_model="BAAI/bge-base-en-v1.5")

    assert payload["chunk_id"] == "bridges2025-1-0000"
    assert payload["title"] == "A Paper Title"
    assert payload["authors"] == ["Ada Lovelace"]
    assert payload["year"] == 2025
    assert payload["embedding_model"] == "BAAI/bge-base-en-v1.5"


def test_collection_for_slugifies_the_model_name():
    assert collection_for("BAAI/bge-large-en-v1.5") == "bridges_papers__bge-large-en-v1.5"


def test_collection_for_lowercases_and_has_no_slash():
    slug = collection_for("Some/Weird-Casing")
    assert slug == "bridges_papers__weird-casing"
    assert "/" not in slug
