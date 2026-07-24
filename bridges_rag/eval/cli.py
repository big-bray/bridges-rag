"""Command-line entry point for running the retrieval benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from bridges_rag.embed.embedder import DEFAULT_MODEL_NAME, Embedder
from bridges_rag.embed.sparse import DEFAULT_SPARSE_MODEL_NAME, SparseEmbedder
from bridges_rag.eval.benchmark import load_benchmark
from bridges_rag.eval.runner import evaluate, format_results_table
from bridges_rag.index.qdrant import DEFAULT_COLLECTION, DEFAULT_URL, get_client
from bridges_rag.search.rerank import (
    DEFAULT_RERANKER_MODEL_NAME,
    CrossEncoderReranker,
    RerankRetriever,
)
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
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="rerank the retrieved candidate pool with a cross-encoder before scoring",
    )
    parser.add_argument("--reranker-model-name", default=DEFAULT_RERANKER_MODEL_NAME)
    parser.add_argument(
        "--pool-size",
        type=int,
        default=100,
        help="candidate pool size fetched before reranking (only used with --rerank)",
    )
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

    mode = "hybrid" if args.hybrid else "dense"
    if args.rerank:
        reranker = CrossEncoderReranker(args.reranker_model_name)
        retriever = RerankRetriever(retriever, reranker, pool_size=args.pool_size)
        mode += "→rerank"

    results = evaluate(retriever, questions)

    print(f"n = {results.n_questions} questions, model = {args.model_name} ({mode})\n")
    print(format_results_table(results))


if __name__ == "__main__":
    main()
