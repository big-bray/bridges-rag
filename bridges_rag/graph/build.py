from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import combinations

from bridges_rag.graph.db import GraphDriver
from bridges_rag.ingest.models import Paper

_CONSTRAINTS = (
    "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (y:Year) REQUIRE y.year IS UNIQUE",
)

_MERGE_PAPER_AND_AUTHORS = """
MERGE (p:Paper {paper_id: $paper_id})
SET p.title = $title, p.year = $year
MERGE (y:Year {year: $year})
MERGE (p)-[:PUBLISHED_IN]->(y)
WITH p
UNWIND $authors AS name
MERGE (a:Author {name: name})
MERGE (a)-[:AUTHORED]->(p)
"""

_MERGE_CO_AUTHORS = """
UNWIND $pairs AS pair
MATCH (a:Author {name: pair[0]}), (b:Author {name: pair[1]})
MERGE (a)-[:CO_AUTHORED_WITH]-(b)
"""


@dataclass
class GraphStats:
    papers: int
    authors: int
    years: int
    co_authored_pairs: int


def ensure_constraints(driver: GraphDriver) -> None:
    with driver.session() as session:
        for statement in _CONSTRAINTS:
            session.run(statement)


def clear_graph(driver: GraphDriver) -> None:
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")


def build_graph(papers: Iterable[Paper], driver: GraphDriver) -> GraphStats:
    papers = list(papers)
    ensure_constraints(driver)

    authors: set[str] = set()
    years: set[int] = set()
    co_authored_pairs = 0

    with driver.session() as session:
        for paper in papers:
            if not paper.authors:
                continue
            session.run(
                _MERGE_PAPER_AND_AUTHORS,
                paper_id=paper.paper_id,
                title=paper.title,
                year=paper.year,
                authors=paper.authors,
            )
            authors.update(paper.authors)
            years.add(paper.year)

            unique_authors = sorted(set(paper.authors))
            pairs = list(combinations(unique_authors, 2))
            if pairs:
                session.run(_MERGE_CO_AUTHORS, pairs=pairs)
                co_authored_pairs += len(pairs)

    return GraphStats(
        papers=len(papers),
        authors=len(authors),
        years=len(years),
        co_authored_pairs=co_authored_pairs,
    )
