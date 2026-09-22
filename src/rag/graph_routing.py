from rag.graph_state import RAGState
from rag.retry_policy import should_retry


def route_after_grading(
    state: RAGState,
    max_retries: int = 2,
) -> str:
    """Choose the next graph node after relevance grading."""

    if state.sufficient_context:
        return "generate"

    if should_retry(
        state,
        max_retries=max_retries,
    ):
        return "retry"

    return "fallback"
