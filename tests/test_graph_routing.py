from rag.graph_routing import route_after_grading
from rag.graph_state import RAGState


def test_route_to_generate_when_context_is_sufficient() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        sufficient_context=True,
    )

    assert route_after_grading(state) == "generate"


def test_route_to_retry_when_context_is_insufficient() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        sufficient_context=False,
    )

    assert route_after_grading(state) == "retry"


def test_route_defaults_to_retry_for_initial_state() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
    )

    assert route_after_grading(state) == "retry"
