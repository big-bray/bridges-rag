"""Command-line entry point for sweeping candidate embedding models over the benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridges_rag.eval.benchmark import load_benchmark
from bridges_rag.eval.sweep import CANDIDATE_MODELS, format_sweep_table, run_sweep
from bridges_rag.index.pipeline import load_chunks
from bridges_rag.index.qdrant import DEFAULT_URL, get_client


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--benchmark", type=Path, default=Path("eval/benchmark.jsonl"))
    parser.add_argument("--qdrant-url", default=DEFAULT_URL)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="override the candidate model list (default: the built-in shortlist)",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    chunks = load_chunks(args.year, args.data_dir)
    questions = load_benchmark(args.benchmark)
    client = get_client(args.qdrant_url)

    rows = run_sweep(
        chunks,
        questions,
        client,
        models=args.models or CANDIDATE_MODELS,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"n = {len(questions)} questions, {len(chunks)} chunks\n")
    print(format_sweep_table(rows))


if __name__ == "__main__":
    main()
