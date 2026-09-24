import pytest

from rag.rag_evaluation import (
    RAGEvaluationCase,
    RAGEvaluationResult,
    RAGEvaluationSample,
)


def test_evaluation_case_stores_ground_truth() -> None:
    case = RAGEvaluationCase(
        query="What are PostgreSQL indexes used for?",
        reference_answer=(
            "PostgreSQL indexes improve query performance."
        ),
        relevant_chunks=[
            "postgresql_chunk_001",
            "postgresql_chunk_002",
        ],
    )

    assert case.query == (
        "What are PostgreSQL indexes used for?"
    )
    assert case.reference_answer.startswith(
        "PostgreSQL indexes"
    )
    assert case.relevant_chunks == [
        "postgresql_chunk_001",
        "postgresql_chunk_002",
    ]


def test_evaluation_case_rejects_empty_query() -> None:
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        RAGEvaluationCase(
            query="",
            reference_answer="reference answer",
            relevant_chunks=["chunk_001"],
        )


def test_evaluation_case_rejects_empty_reference_answer() -> None:
    with pytest.raises(
        ValueError,
        match="Reference answer cannot be empty",
    ):
        RAGEvaluationCase(
            query="question",
            reference_answer=" ",
            relevant_chunks=["chunk_001"],
        )


def test_evaluation_case_rejects_empty_relevant_chunks() -> None:
    with pytest.raises(
        ValueError,
        match="Relevant chunks cannot be empty",
    ):
        RAGEvaluationCase(
            query="question",
            reference_answer="reference answer",
            relevant_chunks=[],
        )


def test_evaluation_case_rejects_empty_chunk_id() -> None:
    with pytest.raises(
        ValueError,
        match="Relevant chunks cannot contain empty strings",
    ):
        RAGEvaluationCase(
            query="question",
            reference_answer="reference answer",
            relevant_chunks=["chunk_001", " "],
        )


def test_evaluation_sample_stores_inputs() -> None:
    sample = RAGEvaluationSample(
        query="What are PostgreSQL indexes used for?",
        contexts=[
            "PostgreSQL indexes improve query performance."
        ],
        answer="Indexes improve query performance.",
        reference_answer=(
            "PostgreSQL indexes improve query performance "
            "by allowing efficient data access."
        ),
    )

    assert sample.query == (
        "What are PostgreSQL indexes used for?"
    )
    assert len(sample.contexts) == 1
    assert sample.answer.startswith("Indexes")
    assert sample.reference_answer is not None


def test_evaluation_sample_allows_missing_reference_answer() -> None:
    sample = RAGEvaluationSample(
        query="What is Redis?",
        contexts=[
            "Redis is an in-memory data store."
        ],
        answer="Redis is an in-memory data store.",
    )

    assert sample.reference_answer is None


def test_evaluation_sample_rejects_empty_query() -> None:
    with pytest.raises(
        ValueError,
        match="Query cannot be empty",
    ):
        RAGEvaluationSample(
            query="",
            contexts=["context"],
            answer="answer",
        )


def test_evaluation_sample_rejects_empty_contexts() -> None:
    with pytest.raises(
        ValueError,
        match="Contexts cannot be empty",
    ):
        RAGEvaluationSample(
            query="question",
            contexts=[],
            answer="answer",
        )


def test_evaluation_sample_rejects_empty_context_string() -> None:
    with pytest.raises(
        ValueError,
        match="Contexts cannot contain empty strings",
    ):
        RAGEvaluationSample(
            query="question",
            contexts=["context", "  "],
            answer="answer",
        )


def test_evaluation_sample_rejects_empty_answer() -> None:
    with pytest.raises(
        ValueError,
        match="Answer cannot be empty",
    ):
        RAGEvaluationSample(
            query="question",
            contexts=["context"],
            answer="",
        )


def test_evaluation_sample_rejects_empty_reference_answer() -> None:
    with pytest.raises(
        ValueError,
        match="Reference answer cannot be empty",
    ):
        RAGEvaluationSample(
            query="question",
            contexts=["context"],
            answer="answer",
            reference_answer=" ",
        )


def test_evaluation_result_defaults_to_uncomputed_metrics() -> None:
    result = RAGEvaluationResult()

    assert result.faithfulness is None
    assert result.answer_relevancy is None
    assert result.context_precision is None
    assert result.context_recall is None


def test_evaluation_result_accepts_valid_metrics() -> None:
    result = RAGEvaluationResult(
        faithfulness=0.9,
        answer_relevancy=0.8,
        context_precision=0.7,
        context_recall=1.0,
    )

    assert result.faithfulness == 0.9
    assert result.answer_relevancy == 0.8
    assert result.context_precision == 0.7
    assert result.context_recall == 1.0


@pytest.mark.parametrize(
    "field_name",
    [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ],
)
def test_evaluation_result_rejects_metric_above_one(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0.0 and 1.0",
    ):
        RAGEvaluationResult(**{field_name: 1.1})


@pytest.mark.parametrize(
    "field_name",
    [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ],
)
def test_evaluation_result_rejects_metric_below_zero(
    field_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"{field_name} must be between 0.0 and 1.0",
    ):
        RAGEvaluationResult(**{field_name: -0.1})