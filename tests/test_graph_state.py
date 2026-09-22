from rag.graph_state import RAGState


def test_rag_state_initializes_with_query() -> None:
    state = RAGState(query="How does Redis caching work?")

    assert state.query == "How does Redis caching work?"
    assert state.transformed_queries == []
    assert state.candidates == []
    assert state.reranked_results == []
    assert state.graded_results == []
    assert state.context == []
    assert state.answer is None
    assert state.grounded is None
    assert state.retry_count == 0
    assert state.sufficient_context is False


def test_rag_state_can_store_pipeline_results() -> None:
    state = RAGState(
        query="PostgreSQL indexes",
        transformed_queries=["PostgreSQL index usage"],
        context=["Indexes improve query performance."],
        answer="PostgreSQL indexes can improve query performance.",
        grounded=True,
        retry_count=1,
        sufficient_context=True,
    )

    assert state.transformed_queries == [
        "PostgreSQL index usage"
    ]
    assert state.context == [
        "Indexes improve query performance."
    ]
    assert state.answer == (
        "PostgreSQL indexes can improve query performance."
    )
    assert state.grounded is True
    assert state.retry_count == 1
    assert state.sufficient_context is True
