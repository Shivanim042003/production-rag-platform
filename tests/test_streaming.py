import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from rag.main import app

client = TestClient(app)

QUESTION = "What are PostgreSQL indexes used for?"


def parse_events(body: str) -> list[dict]:
    return [
        json.loads(line[len("data: "):])
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


@patch("rag.main.rag_service")
def test_stream_emits_tokens_before_final_answer(
    mock_rag_service: MagicMock,
) -> None:
    def fake_query(question, on_token=None):
        on_token("PostgreSQL")
        on_token(" indexes")

        return {
            "query": question,
            "answer": "PostgreSQL indexes improve query performance.",
            "grounded": True,
            "sources": [],
        }

    mock_rag_service.query.side_effect = fake_query

    response = client.post(
        "/api/v1/rag/query/stream",
        json={"query": QUESTION},
    )

    events = parse_events(response.text)

    assert [event["type"] for event in events] == [
        "status",
        "token",
        "token",
        "answer",
        "grounding",
        "sources",
        "complete",
    ]
    assert [
        event["text"] for event in events if event["type"] == "token"
    ] == ["PostgreSQL", " indexes"]


@patch("rag.main.rag_service")
def test_stream_reports_value_errors(
    mock_rag_service: MagicMock,
) -> None:
    mock_rag_service.query.side_effect = ValueError("bad question")

    response = client.post(
        "/api/v1/rag/query/stream",
        json={"query": QUESTION},
    )

    events = parse_events(response.text)

    assert [event["type"] for event in events] == ["status", "error"]
    assert events[-1]["message"] == "bad question"


@patch("rag.main.rag_service", None)
def test_stream_reports_service_not_ready() -> None:
    response = client.post(
        "/api/v1/rag/query/stream",
        json={"query": QUESTION},
    )

    events = parse_events(response.text)

    assert [event["type"] for event in events] == ["status", "error"]
    assert events[-1]["message"] == "RAG service is still initializing."
