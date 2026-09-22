from rag.graph_state import RAGState


def route_after_grounding(state: RAGState) -> str:
    """Choose the next node after grounding validation."""

    if state.grounded:
        return "end"

    return "fallback"
