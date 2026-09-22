from rag.graph_state import RAGState


def route_after_grading(state: RAGState) -> str:
    """Choose the next graph node after relevance grading."""

    if state.sufficient_context:
        return "generate"

    return "retry"
