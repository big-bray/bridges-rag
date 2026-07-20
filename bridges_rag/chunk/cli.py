"""Command-line entry point for chunking a year's extracted markdown."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridges_rag.chunk.chunker import DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_TOKENS, chunk_year


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="root directory holding the year's manifest.jsonl and extracted markdown",
    )
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_OVERLAP_TOKENS)
    args = parser.parse_args()

    chunks = chunk_year(
        args.year,
        args.data_dir,
        target_tokens=args.target_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    out_path = args.data_dir / str(args.year) / "chunks.jsonl"
    print(f"Chunked {len(chunks)} chunks for {args.year} into {out_path}")


if __name__ == "__main__":
    main()
