# BridgesRAG

Local semantic search over the [Bridges Mathematical Art Archive](https://archive.bridgesmathart.org/): ingests proceedings papers, builds a vector index, and returns relevant passages with citations.

## Architecture

```text
Bridges archive (HTML + PDFs)
        │  httpx + BeautifulSoup + PyMuPDF4LLM
        ▼
   ingest  ──▶ data/<year>/manifest.jsonl          (PDFs streamed to markdown in memory,
        │      data/<year>/markdown/*.md            never written to disk, by default)
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
`ingest` extracts markdown straight from downloaded PDF bytes by default (`extract` then just
confirms the markdown is already there); pass `--persist-pdf` (or set `PERSIST_PDF=1`) to write
PDFs to disk instead, e.g. for offline re-extraction.

## Setup

```sh
make setup   # uv sync + start Qdrant (docker compose) + wait for it to be ready
```

## Build the index

Runs ingest → extract → index → eval for one proceedings year (defaults to `YEAR=2025`):

```sh
make build
```

Or run a stage at a time: `make ingest`, `make extract`, `make index`, `make eval`. Override
`YEAR` or `DATA_DIR` as needed, e.g. `make build YEAR=2025 DATA_DIR=data`.

## Usage

```sh
uv run streamlit run app.py
```

Enter a query, optionally filter by author or year, and get back ranked passages with paper/page citations.

## Evaluation

25 hand-written questions with paper-level gold labels (`eval/benchmark.jsonl`) measure retrieval quality at the paper level. Recall@k is the fraction of gold papers found in the top-k unique papers; MRR is the reciprocal rank of the first gold paper.

```sh
make eval
```

Results for the MVP config (`bge-base-en-v1.5`, cosine similarity, n=25):

| Metric | Value |
|---|---|
| Recall@1 | 0.54 |
| Recall@5 | 0.80 |
| Recall@10 | 0.89 |
| MRR | 0.907 |

### Embedding model sweep

`sweep_cli` re-embeds and re-indexes the corpus with each candidate model and reports the same Recall@k/MRR metrics alongside embedding cost (time, model size on disk, vector storage):

```sh
make sweep
```

Results for n=25, cosine similarity:

| Model | Dim | Recall@1 | Recall@5 | Recall@10 | MRR | Embed (s) | Query (ms) | Weights (MB) | Vectors (MB) |
|---|---|---|---|---|---|---|---|---|---|
| BAAI/bge-base-en-v1.5 (MVP) | 768 | 0.54 | 0.80 | 0.89 | 0.907 | 62.4 | 40 | 419 | 6.4 |
| BAAI/bge-large-en-v1.5 | 1024 | 0.55 | 0.88 | 0.93 | 0.923 | 202.3 | 55 | 1280 | 8.5 |
| thenlper/gte-large | 1024 | 0.58 | 0.89 | 0.93 | 0.980 | 186.2 | 50 | 640 | 8.5 |
| intfloat/e5-base-v2 | 768 | 0.57 | 0.83 | 0.89 | 0.913 | 63.9 | 39 | 419 | 6.4 |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 0.55 | 0.81 | 0.91 | 0.933 | 9.2 | 66 | 87 | 3.2 |
| nomic-ai/nomic-embed-text-v1.5 | 768 | 0.62 | 0.82 | 0.90 | 1.000 | 342.5 | 60 | 523 | 6.4 |

`nomic-embed-text-v1.5` tops both Recall@1 and MRR but it's the slowest to embed, likely due to its 8192-token context window and lack of a fast ONNX/PyTorch path in this setup. `gte-large` is the best all-around performer when considering embed cost. `all-MiniLM-L6-v2` trails slightly on quality but embeds significantly faster at a fraction of the size, which is a reasonable tradeoff if embedding throughput or storage matters more than the last few points of recall.

### Hybrid search (dense + BM25)

`index --hybrid` additionally computes and stores BM25 sparse vectors alongside the dense ones. `eval --hybrid` fuses dense + sparse retrieval via RRF:

```sh
make index HYBRID=1
make eval HYBRID=1
```

Results for n=25, using `bge-base-en-v1.5` dense model and FastEmbed's `Qdrant/bm25`:

| Metric | Dense | Hybrid | Δ |
|---|---|---|---|
| Recall@1 | 0.54 | 0.61 | +0.07 |
| Recall@5 | 0.80 | 0.82 | +0.02 |
| Recall@10 | 0.89 | 0.89 | — |
| MRR | 0.907 | 0.973 | +0.066 |

Hybrid improves retrieval across the board. RRF fusion pulls exact-term matches (titles, author names, jargon) above where dense-only search ranked them.

### Cross-encoder reranking

`eval --rerank` fetches a wide candidate pool from the base retriever and rescores it with a cross-encoder that scores each (query, passage) pair jointly instead of comparing independently-embedded vectors:

```sh
make eval RERANK=1
make eval HYBRID=1 RERANK=1
```

Results for n=25, `bge-base-en-v1.5` dense model, `bge-reranker-base` cross-encoder:

| Config | Recall@1 | Recall@5 | Recall@10 | MRR | Avg query (ms) |
|---|---|---|---|---|---|
| Dense | 0.54 | 0.80 | 0.89 | 0.907 | 50 |
| Dense → Rerank | 0.41 | 0.81 | 0.92 | 0.806 | 3,004 |
| Hybrid | 0.61 | 0.82 | 0.89 | 0.973 | 46 |
| Hybrid → Rerank | 0.39 | 0.82 | 0.89 | 0.788 | 2,425 |

Reranking is a net loss on this benchmark: it drops recall and MRR in both the dense and hybrid case, while making queries >50x slower.

## Future Work
- **Query expansion** — rewrite or expand the query before retrieval, to see whether it pushes Recall@k/MRR further than hybrid search alone.
- **Knowledge graph (GraphRAG)** — extract entities and relationships into a Neo4j graph and combine graph traversal with vector retrieval, aimed at relational questions (e.g. "who has collaborated with X on tiling papers") that similarity search alone can't answer.
- **Mathematical symbols and images** — `extract` currently flattens PDFs to plain markdown text; PyMuPDF4LLM's handling of formulas is inconsistent (equations often come through as mangled Unicode or drop out entirely), and embedded figures — tiling diagrams, sculpture photos, geometric constructions, all central to this corpus — are ignored altogether, so no chunk, embedding, or citation ever represents them. A question that hinges on a specific formula or references "the spiral pattern in Figure 3" is currently unanswerable no matter how good retrieval gets. Worth evaluating: a formula-aware extractor (e.g. Nougat, Mathpix) to preserve LaTeX in chunk text, and a multimodal embedding model (e.g. CLIP-style) to index figures so image-referencing questions become retrievable and citable alongside text.

## Development

```sh
make lint
make test
```
