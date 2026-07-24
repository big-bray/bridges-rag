"""Command-line entry point for LLM concept extraction and writing the concept graph."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridges_rag.graph.build_concepts import build_concept_graph
from bridges_rag.graph.concepts import DEFAULT_HOST, DEFAULT_MODEL_NAME, OllamaConceptExtractor
from bridges_rag.graph.concepts_pipeline import extract_concepts_year
from bridges_rag.graph.db import DEFAULT_PASSWORD, DEFAULT_URI, DEFAULT_USER, get_driver


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="root directory holding the year's chunks.jsonl / concepts.jsonl",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ollama-host", default=DEFAULT_HOST)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only extract from the first N chunks (for trying the pipeline out cheaply)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run extraction even if data/<year>/concepts.jsonl already exists",
    )
    parser.add_argument("--neo4j-uri", default=DEFAULT_URI)
    parser.add_argument("--neo4j-user", default=DEFAULT_USER)
    parser.add_argument("--neo4j-password", default=DEFAULT_PASSWORD)
    args = parser.parse_args()

    extractor = OllamaConceptExtractor(args.model_name, args.ollama_host)
    chunk_concepts = extract_concepts_year(
        args.year, args.data_dir, extractor, limit=args.limit, force=args.force
    )

    driver = get_driver(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    try:
        stats = build_concept_graph(chunk_concepts, driver)
    finally:
        driver.close()

    print(
        f"Concept graph for {args.year}: {stats.chunks_processed} chunks processed "
        f"({stats.chunks_failed} failed), {stats.concepts} concepts, "
        f"{stats.mentions_edges} MENTIONS, {stats.related_to_edges} RELATED_TO"
    )


if __name__ == "__main__":
    main()
