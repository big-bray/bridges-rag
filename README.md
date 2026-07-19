# BridgesRAG

Local semantic search over the [Bridges Mathematical Art Archive](https://archive.bridgesmathart.org/): ingests proceedings papers, builds a vector index, and returns relevant passages with citations.

See [PLAN.md](PLAN.md) for scope and milestones.

## Setup

```sh
uv sync
```

## Development

```sh
uv run ruff check .
uv run pytest
```
