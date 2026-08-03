"""Write LLM-extracted concept entities + relation triples into Neo4j.

`MENTIONS` links a Paper to each Concept its chunks mention; `RELATED_TO` links two
Concepts per an extracted (subject, relation, object) triple, with the relation text
kept as an edge property rather than a fixed taxonomy of relationship types. Concept
names are merged case-insensitively (trimmed + lowercased) so "Hyperbolic Tiling" and
"hyperbolic tiling" collide into one node; the first-seen casing is kept for display.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from bridges_rag.graph.concepts_store import ChunkConcepts
from bridges_rag.graph.db import GraphDriver

_CONCEPT_CONSTRAINT = "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE"

_MERGE_MENTIONS = """
MATCH (p:Paper {paper_id: $paper_id})
UNWIND $entities AS entity
MERGE (c:Concept {name: entity.key})
ON CREATE SET c.display_name = entity.display
MERGE (p)-[:MENTIONS]->(c)
"""

_MERGE_RELATED_TO = """
UNWIND $triples AS triple
MERGE (a:Concept {name: triple.subject_key})
ON CREATE SET a.display_name = triple.subject_display
MERGE (b:Concept {name: triple.object_key})
ON CREATE SET b.display_name = triple.object_display
MERGE (a)-[:RELATED_TO {label: triple.relation}]-(b)
"""


def normalize_concept(name: str) -> str:
    """Merge key for a concept name: trimmed, whitespace-collapsed, lowercased."""
    return " ".join(name.strip().lower().split())


@dataclass
class ConceptGraphStats:
    chunks_processed: int
    chunks_failed: int
    concepts: int
    mentions_edges: int
    related_to_edges: int


def ensure_concept_constraint(driver: GraphDriver) -> None:
    with driver.session() as session:
        session.run(_CONCEPT_CONSTRAINT)


def build_concept_graph(
    chunk_concepts: Iterable[ChunkConcepts], driver: GraphDriver
) -> ConceptGraphStats:
    """Write MENTIONS (Paper->Concept) and RELATED_TO (Concept--Concept) edges from
    LLM-extracted per-chunk concepts. Idempotent (MERGE-based); assumes `build_graph`
    already created the Paper nodes these MENTIONS edges attach to."""
    chunk_concepts = list(chunk_concepts)
    ensure_concept_constraint(driver)

    concepts: set[str] = set()
    mentions_edges = 0
    related_to_edges = 0
    chunks_failed = 0

    with driver.session() as session:
        for cc in chunk_concepts:
            if cc.error is not None:
                chunks_failed += 1
                continue

            entities = [
                {"key": normalize_concept(e), "display": e}
                for e in dict.fromkeys(cc.entities)
                if e.strip()
            ]
            if entities:
                session.run(_MERGE_MENTIONS, paper_id=cc.paper_id, entities=entities)
                concepts.update(e["key"] for e in entities)
                mentions_edges += len(entities)

            triples = [
                {
                    "subject_key": normalize_concept(t.subject),
                    "subject_display": t.subject,
                    "object_key": normalize_concept(t.object),
                    "object_display": t.object,
                    "relation": t.relation,
                }
                for t in cc.triples
                if t.subject.strip() and t.object.strip()
            ]
            if triples:
                session.run(_MERGE_RELATED_TO, triples=triples)
                concepts.update(t["subject_key"] for t in triples)
                concepts.update(t["object_key"] for t in triples)
                related_to_edges += len(triples)

    return ConceptGraphStats(
        chunks_processed=len(chunk_concepts) - chunks_failed,
        chunks_failed=chunks_failed,
        concepts=len(concepts),
        mentions_edges=mentions_edges,
        related_to_edges=related_to_edges,
    )
