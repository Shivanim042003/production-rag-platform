import pytest
from pydantic import ValidationError

from rag.api_models import RAGQueryRequest


def test_rag_query_request_accepts_valid_query() -> None:
    request = RAGQueryRequest(
        query="What are PostgreSQL indexes used for?"
    )

    assert request.query == "What are PostgreSQL indexes used for?"


def test_rag_query_request_rejects_empty_query() -> None:
    with pytest.raises(ValidationError):
        RAGQueryRequest(query="")


def test_rag_query_request_rejects_query_over_2000_characters() -> None:
    long_query = "a" * 2001

    with pytest.raises(ValidationError):
        RAGQueryRequest(query=long_query)


def test_rag_query_request_accepts_query_at_2000_characters() -> None:
    query = "a" * 2000

    request = RAGQueryRequest(query=query)

    assert len(request.query) == 2000