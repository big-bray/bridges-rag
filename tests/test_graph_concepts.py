import pytest
from pydantic import ValidationError

from bridges_rag.graph.concepts import (
    MAX_ENTITIES,
    MAX_NAME_LENGTH,
    MAX_TRIPLES,
    ExtractedConcepts,
    ExtractedTriple,
)


def test_extracted_concepts_accepts_empty_lists():
    concepts = ExtractedConcepts(entities=[], triples=[])
    assert concepts.entities == []
    assert concepts.triples == []


def test_extracted_concepts_parses_from_json():
    payload = (
        '{"entities": ["weaving", "tiling"], '
        '"triples": [{"subject": "weaving", "relation": "uses", "object": "strands"}]}'
    )
    concepts = ExtractedConcepts.model_validate_json(payload)
    assert concepts.entities == ["weaving", "tiling"]
    assert concepts.triples == [
        ExtractedTriple(subject="weaving", relation="uses", object="strands")
    ]


def test_extracted_concepts_rejects_too_many_entities():
    with pytest.raises(ValidationError):
        ExtractedConcepts(entities=[f"c{i}" for i in range(MAX_ENTITIES + 1)], triples=[])


def test_extracted_concepts_rejects_too_many_triples():
    triple = ExtractedTriple(subject="a", relation="r", object="b")
    with pytest.raises(ValidationError):
        ExtractedConcepts(entities=[], triples=[triple] * (MAX_TRIPLES + 1))


def test_extracted_concepts_rejects_an_overlong_entity():
    with pytest.raises(ValidationError):
        ExtractedConcepts(entities=["x" * (MAX_NAME_LENGTH + 1)], triples=[])
