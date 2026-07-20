from bridges_rag.embed.embedder import DEFAULT_MODEL_NAME, query_prefix


def test_query_prefix_known_model():
    assert query_prefix(DEFAULT_MODEL_NAME) == (
        "Represent this sentence for searching relevant passages: "
    )


def test_query_prefix_unknown_model_is_empty():
    assert query_prefix("sentence-transformers/all-MiniLM-L6-v2") == ""
