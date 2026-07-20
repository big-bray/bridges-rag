from qdrant_client.http import models as qmodels

from bridges_rag.search.search import build_filter


def _field_condition(condition: object) -> qmodels.FieldCondition:
    assert isinstance(condition, qmodels.FieldCondition)
    return condition


def test_build_filter_returns_none_without_constraints():
    assert build_filter() is None


def test_build_filter_author_only():
    filt = build_filter(author="Ada Lovelace")
    assert filt is not None and isinstance(filt.must, list)
    condition = _field_condition(filt.must[0])
    assert condition.key == "authors"
    assert isinstance(condition.match, qmodels.MatchValue)
    assert condition.match.value == "Ada Lovelace"


def test_build_filter_year_only():
    filt = build_filter(year=2025)
    assert filt is not None and isinstance(filt.must, list)
    condition = _field_condition(filt.must[0])
    assert condition.key == "year"
    assert isinstance(condition.match, qmodels.MatchValue)
    assert condition.match.value == 2025


def test_build_filter_author_and_year():
    filt = build_filter(author="Ada Lovelace", year=2025)
    assert filt is not None and isinstance(filt.must, list)
    keys = {_field_condition(c).key for c in filt.must}
    assert keys == {"authors", "year"}
