from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from rag.observability import RAGTrace
from rag.rag_service import RAGService


@dataclass
class FakeChunk:
    chunk_id: str
    text: str = "test"


@dataclass
class FakeResult:
    chunk: FakeChunk
    score: float


@dataclass
class FakeGrade:
    relevant: bool


def test_rag_trace_records_request_metadata() -> None:
    trace = RAGTrace(
        query="What are indexes used for?"
    )

    trace.metadata.update(
        {
            "candidate_count": 10,
            "reranked_count": 5,
            "relevant_context_count": 2,
            "retry_count": 0,
            "grounded": True,
        }
    )

    assert trace.query == "What are indexes used for?"
    assert trace.metadata["candidate_count"] == 10
    assert trace.metadata["reranked_count"] == 5
    assert trace.metadata["relevant_context_count"] == 2
    assert trace.metadata["retry_count"] == 0
    assert trace.metadata["grounded"] is True


def test_rag_trace_records_stage_timing() -> None:
    trace = RAGTrace(query="test query")

    trace.add_stage(
        name="retrieval",
        duration_ms=10.5,
    )

    trace.add_stage(
        name="generation",
        duration_ms=25.0,
    )

    trace.finish(
        total_duration_ms=40.0,
    )

    assert len(trace.stages) == 2

    assert trace.stages[0].name == "retrieval"
    assert trace.stages[0].duration_ms == 10.5

    assert trace.stages[1].name == "generation"
    assert trace.stages[1].duration_ms == 25.0

    assert trace.total_duration_ms == 40.0


def test_rag_service_query_returns_result() -> None:
    service = object.__new__(RAGService)

    service._build_graph = MagicMock()

    fake_graph = MagicMock()
    fake_graph.invoke.return_value = {
        "answer": "PostgreSQL indexes improve query performance.",
        "grounded": True,
        "retry_count": 0,
        "candidates": [
            FakeResult(
                chunk=FakeChunk("chunk-1"),
                score=0.80,
            ),
            FakeResult(
                chunk=FakeChunk("chunk-2"),
                score=0.70,
            ),
        ],
        "reranked_results": [
            FakeResult(
                chunk=FakeChunk("chunk-1"),
                score=0.95,
            ),
        ],
        "graded_results": [
            (
                FakeResult(
                    chunk=FakeChunk("chunk-1"),
                    score=0.95,
                ),
                FakeGrade(relevant=True),
            ),
        ],
    }

    service._build_graph.return_value = fake_graph

    with patch("rag.rag_service.logger") as mock_logger:
        result = service.query(
            "What are PostgreSQL indexes used for?"
        )

    assert result["query"] == (
        "What are PostgreSQL indexes used for?"
    )

    assert result["answer"] == (
        "PostgreSQL indexes improve query performance."
    )

    assert result["grounded"] is True

    assert result["sources"] == [
        {
            "chunk_id": "chunk-1",
            "score": 0.95,
        }
    ]

    service._build_graph.assert_called_once()

    fake_graph.invoke.assert_called_once()

    mock_logger.info.assert_called_once()

    log_call = mock_logger.info.call_args

    assert log_call.args[0] == "RAG request completed"

    extra = log_call.kwargs["extra"]
    trace = extra["rag_trace"]

    assert trace["query"] == (
        "What are PostgreSQL indexes used for?"
    )

    assert trace["total_duration_ms"] >= 0

    assert trace["metadata"]["candidate_count"] == 2
    assert trace["metadata"]["reranked_count"] == 1
    assert trace["metadata"]["relevant_context_count"] == 1
    assert trace["metadata"]["retry_count"] == 0
    assert trace["metadata"]["grounded"] is True