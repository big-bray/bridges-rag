"""LLM-based concept/technique entity + relation extraction (Microsoft GraphRAG-style triples).

Runs against a local Ollama model. Extraction is per-chunk, bounded (at most a handful of
entities/triples per call) so a single call stays fast and the JSON schema stays easy for a
small local model to fill in reliably.
"""

from __future__ import annotations

from typing import Annotated, Protocol

from ollama import Client
from pydantic import BaseModel, Field

DEFAULT_MODEL_NAME = "llama3.2"
DEFAULT_HOST = "http://localhost:11434"

MAX_ENTITIES = 8
MAX_TRIPLES = 6
# A real concept/technique name or short relation phrase, not a sentence; caps runaway
# generations (e.g. a whole clause returned as an "entity") without another prompt round-trip.
MAX_NAME_LENGTH = 60
MAX_RELATION_LENGTH = 40

ConceptName = Annotated[str, Field(max_length=MAX_NAME_LENGTH)]

SYSTEM_PROMPT = """You extract mathematical-art concepts and techniques literally present in a \
short excerpt of an academic paper about mathematics and art (the Bridges conference).

A concept/technique is a named mathematical structure, geometric construction, artistic \
technique, material, or algorithm that the excerpt actually discusses (not just mentions in \
passing as an author's affiliation or address). Do NOT extract author names, institutions, \
addresses, or generic words.

If the excerpt is only an author byline, address, or table-of-contents line with no discussion \
of a technique or structure, return empty "entities" and "triples" lists.

Also extract (subject, relation, object) triples describing how two extracted concepts relate, \
using a short verb phrase for the relation. Only include a relation explicitly supported by the \
excerpt."""


class ExtractedTriple(BaseModel):
    """A (subject, relation, object) triple between two concepts."""

    subject: ConceptName
    relation: Annotated[str, Field(max_length=MAX_RELATION_LENGTH)]
    object: ConceptName


class ExtractedConcepts(BaseModel):
    """Raw structured output of one extraction call."""

    entities: list[ConceptName] = Field(max_length=MAX_ENTITIES)
    triples: list[ExtractedTriple] = Field(max_length=MAX_TRIPLES)


class ConceptExtractor(Protocol):
    def extract(self, text: str) -> ExtractedConcepts:
        """Extract concept entities and relation triples mentioned in `text`."""
        ...


class OllamaConceptExtractor:
    """Reference ConceptExtractor: a local Ollama model with JSON-schema-constrained output."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, host: str = DEFAULT_HOST) -> None:
        self.model_name = model_name
        self._client = Client(host=host)

    def extract(self, text: str) -> ExtractedConcepts:
        response = self._client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            format=ExtractedConcepts.model_json_schema(),
            options={"temperature": 0},
            think=False,
        )
        content = response.message.content
        if not content:
            return ExtractedConcepts(entities=[], triples=[])
        return ExtractedConcepts.model_validate_json(content)
