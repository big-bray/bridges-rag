# BridgesRAG

Local semantic search over the [Bridges Mathematical Art Archive](https://archive.bridgesmathart.org/): ingests proceedings papers, builds a vector index, and returns relevant passages with citations.

## Architecture

```text
Bridges archive (HTML + PDFs)
        │  httpx + BeautifulSoup
        ▼
   ingest  ──▶ data/<year>/manifest.jsonl + *.pdf
        │  PyMuPDF4LLM
        ▼
  extract  ──▶ data/<year>/markdown/*.md   (per-page markdown, page numbers preserved)
        │  heading-aware chunking
        ▼
    chunk  ──▶ data/<year>/chunks.jsonl    (token-budgeted, metadata denormalized on each chunk)
        │  sentence-transformers (bge-base-en-v1.5)
        ▼
    index  ──▶ Qdrant collection           (vectors + payload, filterable by author/year)
        │
        ▼
   search  ──▶ Streamlit UI (app.py)       (query → embed → Qdrant search → cited passages)
```

Every stage writes JSON/JSONL that the next stage reads, so any stage can be rerun independently.

## Setup

```sh
uv sync
docker compose up -d   # starts Qdrant; dashboard at http://localhost:6333/dashboard
```

## Build the index

Run once per proceedings year (defaults to `--year 2025`):

```sh
uv run python -m bridges_rag.ingest.cli   # scrape listing, download PDFs, write the manifest
uv run python -m bridges_rag.extract.cli  # extract per-paper markdown
uv run python -m bridges_rag.index.cli    # chunk (on demand), embed, and upsert into Qdrant
```

## Usage

```sh
uv run streamlit run app.py
```

Enter a query, optionally filter by author or year, and get back ranked passages with paper/page citations.

## Evaluation

25 hand-written questions with paper-level gold labels (`eval/benchmark.jsonl`) measure retrieval quality at the paper level. Recall@k is the fraction of gold papers found in the top-k unique papers; MRR is the reciprocal rank of the first gold paper.

```sh
uv run python -m bridges_rag.eval.cli
```

Results for the MVP config (`bge-base-en-v1.5`, cosine similarity, n=25):

| Metric | Value |
|---|---|
| Recall@1 | 0.54 |
| Recall@5 | 0.80 |
| Recall@10 | 0.89 |
| MRR | 0.907 |

## Future Work

The MVP ships one retrieval technique (dense embeddings only) so it can serve as a baseline. Deferred experiments will be benchmarked against it using the same Recall@k/MRR eval:

- **Embedding model comparison** — swap `bge-base-en-v1.5` for alternatives; payloads already record the embedding model name to support side-by-side indexes.
- **Improved retrieval** — hybrid search (BM25 + embeddings), reranking, and query expansion, to see how far each pushes Recall@k/MRR past the dense-only baseline.
- **Knowledge graph (GraphRAG)** — extract entities and relationships into a Neo4j graph and combine graph traversal with vector retrieval, aimed at relational questions (e.g. "who has collaborated with X on tiling papers") that similarity search alone can't answer.

## Development

```sh
uv run ruff check .
uv run mypy .
uv run pytest
```
