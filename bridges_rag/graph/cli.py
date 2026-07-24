"""Command-line entry point for building the metadata graph into Neo4j from the manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridges_rag.graph.build import build_graph, clear_graph
from bridges_rag.graph.db import DEFAULT_PASSWORD, DEFAULT_URI, DEFAULT_USER, get_driver
from bridges_rag.ingest.manifest import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="root directory holding the year's manifest.jsonl",
    )
    parser.add_argument("--neo4j-uri", default=DEFAULT_URI)
    parser.add_argument("--neo4j-user", default=DEFAULT_USER)
    parser.add_argument("--neo4j-password", default=DEFAULT_PASSWORD)
    parser.add_argument(
        "--clear",
        action="store_true",
        help="wipe the graph before rebuilding (otherwise MERGE-based rebuild is additive-safe)",
    )
    args = parser.parse_args()

    manifest_path = args.data_dir / str(args.year) / "manifest.jsonl"
    papers = read_manifest(manifest_path)

    driver = get_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        if args.clear:
            clear_graph(driver)
        stats = build_graph(papers, driver)
    finally:
        driver.close()

    print(
        f"Built graph for {args.year}: {stats.papers} papers, {stats.authors} authors, "
        f"{stats.years} years, {stats.co_authored_pairs} co-authorship pairs"
    )


if __name__ == "__main__":
    main()
