from rag.graph_state import RAGState
from rag.grounding_routing import route_after_grounding


def test_route_to_end_when_answer_is_grounded() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        answer="Indexes improve query performance.",
        grounded=True,
    )

    assert route_after_grounding(state) == "end"


def test_route_to_fallback_when_answer_is_not_grounded() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        answer="Indexes reduce memory usage by 90%.",
        grounded=False,
    )

    assert route_after_grounding(state) == "fallback"


def test_route_to_fallback_when_grounding_is_unknown() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
    )

    assert route_after_grounding(state) == "fallback"
