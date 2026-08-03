.PHONY: setup wait-for-qdrant ingest extract index graph concepts eval sweep build all clean lint fmt test

YEAR ?= 2025
DATA_DIR ?= data
QDRANT_URL ?= http://localhost:6333
HYBRID ?=
RERANK ?=
GRAPH ?=
BENCHMARK ?= eval/benchmark.jsonl

setup:
	uv sync
	docker compose up -d
	$(MAKE) wait-for-qdrant

wait-for-qdrant:
	@echo "Waiting for Qdrant at $(QDRANT_URL)..."
	@until curl -sf $(QDRANT_URL)/readyz > /dev/null 2>&1; do sleep 1; done
	@echo "Qdrant is ready."

ingest:
	uv run python -m bridges_rag.ingest.cli --year $(YEAR) --data-dir $(DATA_DIR)

extract:
	uv run python -m bridges_rag.extract.cli --year $(YEAR) --data-dir $(DATA_DIR)

index: wait-for-qdrant
	uv run python -m bridges_rag.index.cli --year $(YEAR) --data-dir $(DATA_DIR) $(if $(HYBRID),--hybrid)

graph:
	uv run python -m bridges_rag.graph.cli --year $(YEAR) --data-dir $(DATA_DIR)

concepts:
	uv run python -m bridges_rag.graph.concepts_cli --year $(YEAR) --data-dir $(DATA_DIR)

eval:
	uv run python -m bridges_rag.eval.cli --benchmark $(BENCHMARK) $(if $(HYBRID),--hybrid) $(if $(RERANK),--rerank) $(if $(GRAPH),--graph --year $(YEAR) --data-dir $(DATA_DIR))

sweep: wait-for-qdrant
	uv run python -m bridges_rag.eval.sweep_cli --year $(YEAR) --data-dir $(DATA_DIR)

build: ingest extract index eval

all: build

clean:
	rm -rf $(DATA_DIR)

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy .

fmt:
	uv run ruff format .

test:
	uv run pytest
