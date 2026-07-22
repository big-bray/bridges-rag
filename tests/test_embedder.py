from bridges_rag.embed.embedder import DEFAULT_MODEL_NAME, passage_prefix, query_prefix


def test_query_prefix_known_model():
    assert query_prefix(DEFAULT_MODEL_NAME) == (
        "Represent this sentence for searching relevant passages: "
    )


def test_query_prefix_unknown_model_is_empty():
    assert query_prefix("sentence-transformers/all-MiniLM-L6-v2") == ""


def test_query_prefix_asymmetric_model():
    assert query_prefix("intfloat/e5-base-v2") == "query: "


def test_passage_prefix_asymmetric_model():
    assert passage_prefix("intfloat/e5-base-v2") == "passage: "


def test_passage_prefix_symmetric_model_is_empty():
    assert passage_prefix(DEFAULT_MODEL_NAME) == ""
