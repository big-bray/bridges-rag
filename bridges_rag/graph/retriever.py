from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from bridges_rag.chunk.models import Chunk
from bridges_rag.graph.db import GraphDriver
from bridges_rag.search.models import SearchResult
from bridges_rag.search.retriever import Retriever

RRF_K = 60

_TRAVERSE_QUERY = """
UNWIND $authors AS name
MATCH (a:Author {name: name})
OPTIONAL MATCH (a)-[:AUTHORED]->(direct:Paper)
OPTIONAL MATCH (a)-[:CO_AUTHORED_WITH]-(:Author)-[:AUTHORED]->(indirect:Paper)
RETURN collect(DISTINCT direct.paper_id) AS direct_ids,
       collect(DISTINCT indirect.paper_id) AS indirect_ids
"""

_TITLE_QUERY = """
UNWIND $titles AS t
MATCH (p:Paper {title: t})
RETURN collect(DISTINCT p.paper_id) AS ids
"""


def link_authors(question: str, author_names: Iterable[str]) -> list[str]:
    """Substring-match known author full names against the question (case-insensitive,
    no fuzzy matching, no LLM). Longest names first so overlapping names don't collide."""
    q = question.lower()
    matches = {name for name in author_names if name and name.lower() in q}
    return sorted(matches, key=len, reverse=True)


def link_titles(question: str, titles: Iterable[str]) -> list[str]:
    """Substring-match known paper titles against the question."""
    q = question.lower()
    matches = {title for title in titles if title and title.lower() in q}
    return sorted(matches, key=len, reverse=True)


def combine_rankings(
    direct_ids: list[str], title_ids: list[str], indirect_ids: list[str]
) -> list[str]:
    """Direct authorship and title hits outrank co-authorship (indirect) hits."""
    return list(dict.fromkeys([*direct_ids, *title_ids, *indirect_ids]))


def rrf_fuse(rankings: list[list[str]], *, k: int = RRF_K) -> list[str]:
    """Reciprocal-rank-fuse multiple paper-id rankings into one, highest score first."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, paper_id in enumerate(ranking, start=1):
            scores[paper_id] += 1.0 / (k + rank)
    return sorted(scores, key=lambda paper_id: scores[paper_id], reverse=True)


def dedup_paper_ids(results: list[SearchResult]) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for result in results:
        paper_id = result.chunk.paper_id
        if paper_id not in seen:
            seen.add(paper_id)
            ranked.append(paper_id)
    return ranked


class GraphTraversal(Protocol):
    def __call__(self, authors: list[str], titles: list[str]) -> list[str]:
        """Return a paper-id ranking for the linked authors/titles, best hits first."""
        ...


@dataclass
class Neo4jGraphTraversal:
    """Reference GraphTraversal: AUTHORED/CO_AUTHORED_WITH traversal over Neo4j."""

    driver: GraphDriver

    def __call__(self, authors: list[str], titles: list[str]) -> list[str]:
        if not authors and not titles:
            return []
        with self.driver.session() as session:
            direct_ids: list[str] = []
            indirect_ids: list[str] = []
            if authors:
                record = session.run(_TRAVERSE_QUERY, authors=authors).single()
                if record is not None:
                    direct_ids = [pid for pid in record["direct_ids"] if pid]
                    indirect_ids = [pid for pid in record["indirect_ids"] if pid]

            title_ids: list[str] = []
            if titles:
                record = session.run(_TITLE_QUERY, titles=titles).single()
                if record is not None:
                    title_ids = [pid for pid in record["ids"] if pid]

        return combine_rankings(direct_ids, title_ids, indirect_ids)


@dataclass
class GraphRetriever:
    """Retriever: fuse a base vector retriever with metadata-graph traversal."""

    base: Retriever
    traverse: GraphTraversal
    author_names: list[str]
    titles: list[str]
    chunk_by_paper: dict[str, Chunk]
    pool_size: int = 50

    def retrieve(self, question: str, *, top_k: int) -> list[SearchResult]:
        base_results = self.base.retrieve(question, top_k=max(self.pool_size, top_k))

        linked_authors = link_authors(question, self.author_names)
        linked_titles = link_titles(question, self.titles)
        graph_ranking = self.traverse(linked_authors, linked_titles)
        if not graph_ranking:
            return base_results[:top_k]

        dense_ranking = dedup_paper_ids(base_results)
        fused_ranking = rrf_fuse([dense_ranking, graph_ranking])[:top_k]

        by_paper = {result.chunk.paper_id: result for result in base_results}
        fused_results: list[SearchResult] = []
        for rank, paper_id in enumerate(fused_ranking, start=1):
            if paper_id in by_paper:
                fused_results.append(by_paper[paper_id])
            elif paper_id in self.chunk_by_paper:
                score = 1.0 / (RRF_K + rank)
                fused_results.append(SearchResult(chunk=self.chunk_by_paper[paper_id], score=score))
        return fused_results
