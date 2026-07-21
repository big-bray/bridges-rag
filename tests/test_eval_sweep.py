from bridges_rag.eval.sweep import (
    SweepRow,
    estimated_vector_bytes,
    format_sweep_table,
    model_disk_usage_bytes,
)


def test_estimated_vector_bytes_is_four_bytes_per_dimension():
    assert estimated_vector_bytes(n_vectors=100, dimension=768) == 100 * 768 * 4


def test_model_disk_usage_bytes_unknown_model_is_none():
    assert model_disk_usage_bytes("nonexistent/not-a-real-model") is None


def test_format_sweep_table_includes_all_columns():
    rows = [
        SweepRow(
            model_name="BAAI/bge-base-en-v1.5",
            dimension=768,
            recall_at_k={1: 0.5, 5: 0.8, 10: 0.9},
            mrr=0.65,
            embed_seconds=12.3,
            avg_query_ms=45.0,
            model_disk_mb=419.0,
            vectors_mb=6.4,
        ),
        SweepRow(
            model_name="unmeasured/model",
            dimension=384,
            recall_at_k={1: 0.4, 5: 0.7, 10: 0.85},
            mrr=0.55,
            embed_seconds=5.0,
            avg_query_ms=20.0,
            model_disk_mb=None,
            vectors_mb=3.3,
        ),
    ]

    table = format_sweep_table(rows)

    assert "BAAI/bge-base-en-v1.5" in table
    assert "0.65" in table  # MRR
    assert "419" in table  # model weights, MB
    assert "?" in table  # unmeasured weights fall back to a placeholder
