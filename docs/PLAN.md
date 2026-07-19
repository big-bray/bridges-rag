# BridgesRAG

**BridgesRAG** is a local, open-source semantic search system for the Bridges Mathematical Art Archive. It ingests papers, extracts text and metadata, builds a searchable vector index, and returns relevant passages with citations. The project is designed as both a useful research tool and a hands-on exploration of modern retrieval systems.

## Goals

* Learn the complete RAG pipeline from ingestion to retrieval.
* Learn how vector databases work in practice.
* Create a genuinely useful search tool for the Bridges archive.

## Guiding Principles

* **Build one retrieval system well before adding more retrieval techniques.** A strong semantic-search baseline makes every later enhancement — hybrid search, GraphRAG, multimodal retrieval, equation search — easy to evaluate quantitatively.
* **The MVP is search, not Q&A.** Ranked passages with citations, served by a thin Streamlit UI. Answer generation is deferred.
* **One tool per job.** One parser, one embedding model, one chunking strategy. Every alternative is a deferred, benchmarkable experiment — not MVP scope.
* **Timebox each milestone.** Ship ugly-but-working; polish at the end.

## MVP Scope (2025 Proceedings)

Target corpus: the most recent *published* proceedings (2025 — the 2026 proceedings don't exist yet).

### 1. Ingestion

* Scrape the 2025 archive listing with httpx + Beautiful Soup.
* Download PDFs with rate limiting and local caching (idempotent re-runs).
* Collect metadata (title, authors, abstract, proceedings pages, page count, citation, supplements) into Pydantic models.
* Commit a reproducible JSONL manifest (URLs + checksums); gitignore `data/` — the PDFs are copyrighted and must not be committed.

### 2. Document Processing

* Extract markdown text with PyMuPDF4LLM, preserving page numbers.
* Math promise: equations may be garbled but must **never break** parsing, chunking, or embedding. Fidelity is deferred.
* Chunk on markdown headings with a token-size target and small overlap; carry paper metadata + page numbers on every chunk.

### 3. Embedding & Indexing

* Embed locally with sentence-transformers (`bge-base-en-v1.5`).
* Run Qdrant in Docker (one compose file, dashboard at localhost:6333).
* Store chunks with payloads: paper ID, title, authors, year, page numbers, **embedding model name** (enables future model comparisons).
* Payload indexes for metadata filtering (author, title, year).

### 4. Search & UI

* Semantic search returning ranked passages with paper/page citations.
* Metadata filters (author, year).
* Thin Streamlit page importing the retrieval code directly — no API layer.

### 5. Evaluation

* Hand-write ~25 questions with **paper-level** gold labels (1–3 relevant papers each) — fast to label, robust to future re-chunking.
* Measure Recall@k and MRR on the single MVP config.
* Publish the results table in the README. This benchmark is the instrument all deferred experiments will reuse.

## Engineering Bar

* Python 3.12, uv, type hints throughout, ruff.
* pytest on deterministic logic only (chunking, metadata parsing, citation formatting) with small fixture files — no mocked-network test theater, no coverage targets.
* GitHub Actions running lint + tests.
* README with architecture diagram and eval results table.

## MVP Stack

```text
Language          Python 3.12
Environment       uv
HTTP              httpx
HTML parsing      Beautiful Soup
Schemas           Pydantic
PDF parsing       PyMuPDF4LLM
Embeddings        sentence-transformers (bge-base-en-v1.5)
Vector database   Qdrant (Docker)
Frontend          Streamlit
Testing           pytest
Data formats      JSONL manifest + original PDFs (gitignored)
```

---

# Milestones

Four weekends, one milestone each. Each milestone ends with something demonstrable.

## Milestone 1 — Ingestion & Parsing

**Done when:** one command downloads the 2025 corpus and produces clean per-paper markdown + metadata.

* [x] Scaffold repo: uv project, `bridges_rag/` package, ruff config, pytest, CI workflow.
* [ ] Scrape the 2025 archive listing; parse paper entries into Pydantic metadata models.
* [ ] Download PDFs with rate limiting; skip already-downloaded files (idempotent).
* [ ] Write the JSONL manifest (metadata + URLs + checksums); gitignore `data/`.
* [ ] Extract markdown per paper with PyMuPDF4LLM, preserving page numbers.
* [ ] Verify math-heavy papers don't crash extraction; log parse failures instead of dying.
* [ ] Tests: metadata parsing from fixture HTML, manifest round-trip.

## Milestone 2 — Chunking, Embedding & Indexing

**Done when:** the full corpus is chunked, embedded, and queryable in the Qdrant dashboard.

* [ ] Heading-aware chunker with token-size target and overlap; page numbers + paper metadata on every chunk.
* [ ] Tests: chunker on fixture markdown (boundaries, overlap, metadata propagation, empty/garbled input).
* [ ] docker-compose.yml for Qdrant; document one-command startup.
* [ ] Embedding module wrapping sentence-transformers; batch embedding with progress reporting.
* [ ] Index chunks into Qdrant with full payloads (including embedding model name); create payload indexes for author/title/year.
* [ ] Smoke check: hand-run a few queries in the Qdrant dashboard and eyeball results.

## Milestone 3 — Search & UI

**Done when:** a Streamlit page answers a real query with cited passages and filters.

* [ ] Search module: query → embed → Qdrant search → ranked results with citations (paper title, authors, pages).
* [ ] Metadata filtering (author, year) plumbed through to Qdrant payload filters.
* [ ] Citation formatting helper + tests.
* [ ] Streamlit page: query box, filter controls, results with passage text + citations.
* [ ] Use it on 5–10 real questions from your own math-art work; fix whatever is obviously broken.

## Milestone 4 — Evaluation & Polish

**Done when:** the README shows an architecture diagram and an eval table a stranger can reproduce.

* [ ] Write ~25 benchmark questions with paper-level gold labels (JSONL).
* [ ] Eval runner: Recall@k (k=1,5,10) and MRR over the benchmark.
* [ ] Record results for the MVP config; commit the results table.
* [ ] README: what it is, architecture diagram, setup (uv + docker compose + one ingest command), usage, eval results.
* [ ] Code pass: type hints complete, ruff clean, module docstrings, dead code removed.

---

# Deferred

Intentionally postponed to keep the MVP focused. Each is a benchmarkable experiment against the MVP baseline.

### Parser & Chunking Experiments

* Docling as the structured parser — **first deferred experiment**.
* Alternative chunking strategies (semantic, fixed-window, layout-aware).
* Compare on the existing benchmark.

### Embedding Model Comparison

* Swap models (payloads already record the model name).
* Compare Recall@k / MRR on the existing benchmark.

### Answer Generation

* Grounded answers with a local LLM (Ollama or llama.cpp).
* Passage-level gold labels for a benchmark subset; answer-quality eval.

### Larger Corpus

* Ingest additional Bridges proceedings.
* Incremental updates.

### Improved Retrieval

* Hybrid search (BM25 + embeddings).
* Reranking.
* Query expansion.

### Knowledge Graph (GraphRAG)

* Extract entities and relationships.
* Build a Neo4j knowledge graph.
* Combine graph traversal with vector retrieval.

### Images

* Extract figures and captions.
* Image embeddings (e.g., CLIP/SigLIP).
* Image and multimodal search.

### Mathematical Retrieval

* Better equation extraction.
* Equation-aware search.
* Symbolic mathematical similarity.

### Production Features

* FastAPI service layer.
* User accounts.
* Cloud deployment.
* Continuous ingestion pipeline.

### Deferred Stack

```text
PDF structured parser  Docling
Generator              Ollama or llama.cpp
API                    FastAPI
Knowledge graph        Neo4j
```
