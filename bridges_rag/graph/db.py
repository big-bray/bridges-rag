"""Neo4j connection helper.

`GraphDriver`/`GraphSession` are structural (Protocol) stand-ins for neo4j's `Driver`/
`Session` — just the `.session()`/`.run()` surface `build_graph` and `GraphRetriever`
actually use. That lets tests pass a lightweight fake instead of a real Neo4j instance,
the same way `Retriever`/`Reranker` are Protocols elsewhere in this codebase.
"""

from __future__ import annotations

from typing import Any, Protocol

from neo4j import Driver, GraphDatabase

DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "bridgesrag"


class GraphSession(Protocol):
    def run(self, query: str, **kwargs: Any) -> Any: ...
    def __enter__(self) -> GraphSession: ...
    def __exit__(self, *args: Any) -> Any: ...


class GraphDriver(Protocol):
    def session(self) -> GraphSession: ...


def get_driver(
    uri: str = DEFAULT_URI, user: str = DEFAULT_USER, password: str = DEFAULT_PASSWORD
) -> Driver:
    return GraphDatabase.driver(uri, auth=(user, password))
