"""Command-line entry point for running the retrieval benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridges_rag.embed.embedder import DEFAULT_MODEL_NAME, Embedder
from bridges_rag.embed.sparse import DEFAULT_SPARSE_MODEL_NAME, SparseEmbedder
from bridges_rag.eval.benchmark import load_benchmark
from bridges_rag.eval.runner import evaluate, format_results_table
from bridges_rag.index.qdrant import DEFAULT_COLLECTION, DEFAULT_URL, get_client
from bridges_rag.search.retriever import DenseRetriever, HybridRetriever, Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("eval/benchmark.jsonl"))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--qdrant-url", default=DEFAULT_URL)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="fuse dense + BM25 sparse retrieval; the collection must be "
        "indexed with `index --hybrid` first",
    )
    parser.add_argument("--sparse-model-name", default=DEFAULT_SPARSE_MODEL_NAME)
    args = parser.parse_args()

    questions = load_benchmark(args.benchmark)
    client = get_client(args.qdrant_url)
    embedder = Embedder(args.model_name)

    retriever: Retriever
    if args.hybrid:
        sparse_embedder = SparseEmbedder(args.sparse_model_name)
        retriever = HybridRetriever(client, embedder, sparse_embedder, collection=args.collection)
    else:
        retriever = DenseRetriever(client, embedder, collection=args.collection)

    results = evaluate(retriever, questions)

    mode = "hybrid" if args.hybrid else "dense"
    print(f"n = {results.n_questions} questions, model = {args.model_name} ({mode})\n")
    print(format_results_table(results))


if __name__ == "__main__":
    main()
