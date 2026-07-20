"""Command-line entry point for ingesting a Bridges proceedings year."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bridges_rag.ingest.pipeline import ingest_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="root directory for downloaded PDFs and the manifest (gitignored)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only process the first N papers (for trying the pipeline out)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="minimum seconds between HTTP requests",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    papers = ingest_year(args.year, args.data_dir, limit=args.limit, delay=args.delay)
    print(f"Ingested {len(papers)} papers for {args.year} into {args.data_dir / str(args.year)}")


if __name__ == "__main__":
    main()
