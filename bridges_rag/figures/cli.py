"""Command-line entry point for extracting figures from downloaded PDFs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bridges_rag.figures.pipeline import extract_figures_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="root directory holding the year's manifest.jsonl and downloaded PDFs "
        "(figures require PDFs persisted via `ingest --persist-pdf`)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only process the first N papers in the manifest",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    figures = extract_figures_year(args.year, args.data_dir, limit=args.limit)
    print(f"Extracted {len(figures)} figures for {args.year}")


if __name__ == "__main__":
    main()
