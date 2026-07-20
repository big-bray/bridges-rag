"""Command-line entry point for extracting markdown from downloaded PDFs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bridges_rag.extract.pipeline import extract_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="root directory holding the year's manifest.jsonl and downloaded PDFs",
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

    extracted = extract_year(args.year, args.data_dir, limit=args.limit)
    ok = sum(1 for paper in extracted if paper.ok)
    print(f"Extracted markdown for {ok}/{len(extracted)} papers for {args.year}")


if __name__ == "__main__":
    main()
