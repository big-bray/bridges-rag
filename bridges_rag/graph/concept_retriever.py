"""Graph retriever v2: entity-link query concepts, traverse the concept subgraph (shared
MENTIONS + adjacent RELATED_TO concepts), and fuse with a base vector retriever via RRF.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from bridges_rag.chunk.models import Chunk
from bridges_rag.graph.build_concepts import normalize_concept
from bridges_rag.graph.db import GraphDriver
from bridges_rag.graph.retriever import combine_rankings, fuse_graph_ranking
from bridges_rag.search.models import SearchResult
from bridges_rag.search.retriever import Retriever

_CONCEPT_NAMES_QUERY = "MATCH (c:Concept) RETURN collect(DISTINCT c.display_name) AS names"

_TRAVERSE_QUERY = """
UNWIND $concepts AS name
MATCH (c:Concept {name: name})
OPTIONAL MATCH (c)<-[:MENTIONS]-(direct:Paper)
OPTIONAL MATCH (c)-[:RELATED_TO]-(:Concept)<-[:MENTIONS]-(indirect:Paper)
RETURN collect(DISTINCT direct.paper_id) AS direct_ids,
       collect(DISTINCT indirect.paper_id) AS indirect_ids
"""


def link_concepts(question: str, concept_names: Iterable[str]) -> list[str]:
    """Substring-match known concept names against the question (case-insensitive,
    no fuzzy matching, no LLM). Longest names first so overlapping names don't collide."""
    q = question.lower()
    matches = {name for name in concept_names if name and name.lower() in q}
    return sorted(matches, key=len, reverse=True)


def fetch_concept_names(driver: GraphDriver) -> list[str]:
    """All known Concept display names, straight from the graph (the source of truth,
    post-build/post-dedup), to entity-link queries against."""
    with driver.session() as session:
        record = session.run(_CONCEPT_NAMES_QUERY).single()
        if record is None:
            return []
        return [name for name in record["names"] if name]


class ConceptTraversal(Protocol):
    def __call__(self, concepts: list[str]) -> list[str]:
        """Return a paper-id ranking for the linked concepts, best hits first."""
        ...


@dataclass
class Neo4jConceptTraversal:
    """Reference ConceptTraversal: MENTIONS/RELATED_TO traversal over Neo4j.

    Papers directly MENTIONS-ing a linked concept outrank papers reached by hopping to an
    adjacent concept via RELATED_TO first (a looser, "related work" signal)."""

    driver: GraphDriver

    def __call__(self, concepts: list[str]) -> list[str]:
        if not concepts:
            return []
        keys = [normalize_concept(name) for name in concepts]
        with self.driver.session() as session:
            record = session.run(_TRAVERSE_QUERY, concepts=keys).single()
            if record is None:
                return []
            direct_ids = [pid for pid in record["direct_ids"] if pid]
            indirect_ids = [pid for pid in record["indirect_ids"] if pid]
        return combine_rankings(direct_ids, [], indirect_ids)


@dataclass
class ConceptGraphRetriever:
    """Retriever: fuse a base vector retriever with concept-subgraph traversal.

    Entity-links concept names mentioned in the query (e.g. "hyperbolic tiling"), traverses
    MENTIONS/RELATED_TO for papers discussing that concept or one related to it, and fuses
    the resulting paper ranking with the base retriever's via RRF. Falls back to the base
    retriever untouched when nothing links.
    """

    base: Retriever
    traverse: ConceptTraversal
    concept_names: list[str]
    chunk_by_paper: dict[str, Chunk]
    pool_size: int = 50

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        base_results = self.base.retrieve(question, top_k=max(self.pool_size, top_k))

        linked_concepts = link_concepts(question, self.concept_names)
        graph_ranking = self.traverse(linked_concepts)
        return fuse_graph_ranking(base_results, graph_ranking, self.chunk_by_paper, top_k=top_k)
