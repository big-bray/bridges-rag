from pathlib import Path

from bridges_rag.eval.benchmark import load_benchmark
from bridges_rag.eval.models import BenchmarkQuestion


def test_load_benchmark_round_trip(tmp_path: Path):
    questions = [
        BenchmarkQuestion(question="What is a Penrose tiling?", paper_ids=["bridges2025-263"]),
        BenchmarkQuestion(
            question="Papers about crochet?", paper_ids=["bridges2025-77", "bridges2025-381"]
        ),
    ]
    path = tmp_path / "benchmark.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.model_dump_json())
            f.write("\n")

    loaded = load_benchmark(path)

    assert loaded == questions


def test_committed_benchmark_has_at_least_25_questions_with_gold_labels():
    path = Path(__file__).parent.parent / "eval" / "benchmark.jsonl"
    questions = load_benchmark(path)

    assert len(questions) >= 25
    for q in questions:
        assert 1 <= len(q.paper_ids) <= 3
