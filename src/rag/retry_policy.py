from rag.graph_state import RAGState


def should_retry(
    state: RAGState,
    max_retries: int = 2,
) -> bool:
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative.")

    return (
        not state.sufficient_context
        and state.retry_count < max_retries
    )
