from bridges_rag.eval.metrics import recall_at_k, reciprocal_rank


def test_recall_at_k_all_gold_within_k():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a", "c"}, k=4) == 1.0


def test_recall_at_k_partial_match():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"a", "z"}, k=4) == 0.5


def test_recall_at_k_respects_k_cutoff():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"d"}, k=2) == 0.0
    assert recall_at_k(ranked, {"d"}, k=4) == 1.0


def test_recall_at_k_no_gold_papers_is_zero():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank(["a", "b", "c"], {"a"}) == 1.0


def test_reciprocal_rank_later_hit():
    assert reciprocal_rank(["a", "b", "c"], {"c"}) == 1 / 3


def test_reciprocal_rank_no_hit_is_zero():
    assert reciprocal_rank(["a", "b", "c"], {"z"}) == 0.0
