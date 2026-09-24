from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("rag.main.rag_service")
def test_query_endpoint_returns_rag_response(
    mock_rag_service: MagicMock,
) -> None:
    mock_rag_service.query.return_value = {
        "query": "What are PostgreSQL indexes used for?",
        "answer": "PostgreSQL indexes improve query performance.",
        "grounded": True,
        "sources": [
            {
                "chunk_id": "chunk-1",
                "score": 0.95,
            }
        ],
    }

    response = client.post(
        "/api/v1/rag/query",
        json={
            "query": "What are PostgreSQL indexes used for?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["query"] == "What are PostgreSQL indexes used for?"
    assert data["answer"] == "PostgreSQL indexes improve query performance."
    assert data["grounded"] is True
    assert data["sources"] == [
        {
            "chunk_id": "chunk-1",
            "score": 0.95,
        }
    ]

    mock_rag_service.query.assert_called_once_with(
        "What are PostgreSQL indexes used for?"
    )


def test_query_endpoint_rejects_empty_query() -> None:
    response = client.post(
        "/api/v1/rag/query",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_query_endpoint_rejects_oversized_query() -> None:
    response = client.post(
        "/api/v1/rag/query",
        json={"query": "a" * 2001},
    )

    assert response.status_code == 422


@patch("rag.main.rag_service")
def test_query_endpoint_returns_400_for_value_error(
    mock_rag_service: MagicMock,
) -> None:
    mock_rag_service.query.side_effect = ValueError(
        "Question cannot be empty."
    )

    response = client.post(
        "/api/v1/rag/query",
        json={"query": "valid query"},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Question cannot be empty."
    }


@patch("rag.main.rag_service")
def test_query_endpoint_returns_500_for_unexpected_error(
    mock_rag_service: MagicMock,
) -> None:
    mock_rag_service.query.side_effect = RuntimeError(
        "Unexpected failure."
    )

    response = client.post(
        "/api/v1/rag/query",
        json={"query": "valid query"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "RAG query failed."
    }


@patch("rag.main.rag_service")
def test_stream_endpoint_returns_sse_events(
    mock_rag_service: MagicMock,
) -> None:
    mock_rag_service.query.return_value = {
        "query": "What are PostgreSQL indexes used for?",
        "answer": "PostgreSQL indexes improve query performance.",
        "grounded": True,
        "sources": [
            {
                "chunk_id": "chunk-1",
                "score": 0.95,
            }
        ],
    }

    response = client.post(
        "/api/v1/rag/query/stream",
        json={
            "query": "What are PostgreSQL indexes used for?"
        },
    )

    assert response.status_code == 200

    assert response.headers["content-type"].startswith(
        "text/event-stream"
    )

    body = response.text

    assert '"type": "status"' in body
    assert '"type": "answer"' in body
    assert '"type": "grounding"' in body
    assert '"type": "sources"' in body
    assert '"type": "complete"' in body

    assert "PostgreSQL indexes improve query performance." in body

    mock_rag_service.query.assert_called_once_with(
        "What are PostgreSQL indexes used for?"
    )
