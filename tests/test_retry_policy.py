import pytest

from rag.graph_state import RAGState
from rag.retry_policy import should_retry


def test_should_retry_when_context_is_insufficient() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        sufficient_context=False,
        retry_count=0,
    )

    assert should_retry(state, max_retries=2) is True


def test_should_not_retry_when_context_is_sufficient() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        sufficient_context=True,
        retry_count=0,
    )

    assert should_retry(state, max_retries=2) is False


def test_should_not_retry_after_max_retries() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        sufficient_context=False,
        retry_count=2,
    )

    assert should_retry(state, max_retries=2) is False


def test_should_not_retry_when_retry_count_exceeds_limit() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        sufficient_context=False,
        retry_count=3,
    )

    assert should_retry(state, max_retries=2) is False


def test_zero_max_retries_disables_retry() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        sufficient_context=False,
        retry_count=0,
    )

    assert should_retry(state, max_retries=0) is False


@pytest.mark.parametrize("max_retries", [-1, -2])
def test_negative_max_retries_is_invalid(max_retries: int) -> None:
    state = RAGState(query="PostgreSQL indexes")

    with pytest.raises(
        ValueError,
        match="max_retries cannot be negative",
    ):
        should_retry(state, max_retries=max_retries)
