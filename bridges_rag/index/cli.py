"""Command-line entry point for embedding and indexing a year's chunks into Qdrant."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridges_rag.embed.embedder import DEFAULT_MODEL_NAME
from bridges_rag.index.pipeline import index_year
from bridges_rag.index.qdrant import DEFAULT_COLLECTION, DEFAULT_URL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="root directory holding the year's chunks.jsonl (chunked on demand if missing)",
    )
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    total = index_year(
        args.year,
        args.data_dir,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    print(f"Indexed {total} chunks for {args.year} into collection '{args.collection}'")


if __name__ == "__main__":
    main()
