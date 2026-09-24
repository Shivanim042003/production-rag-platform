import pytest

from rag.answer_relevancy import AnswerRelevancyResult
from rag.rag_evaluation import RAGEvaluationSample
from rag.rag_evaluator import BasicRAGEvaluator


def test_basic_evaluator_returns_evaluation_result() -> None:
    sample = RAGEvaluationSample(
        query="What are indexes used for?",
        contexts=["Indexes improve query performance."],
        answer="They improve query performance.",
    )

    evaluator = BasicRAGEvaluator()

    result = evaluator.evaluate(sample)

    assert result is not None


def test_basic_evaluator_records_query_metadata() -> None:
    sample = RAGEvaluationSample(
        query="What are indexes used for?",
        contexts=["Indexes improve query performance."],
        answer="They improve query performance.",
    )

    evaluator = BasicRAGEvaluator()

    result = evaluator.evaluate(sample)

    assert result.metadata["query"] == (
        "What are indexes used for?"
    )


def test_basic_evaluator_records_context_count() -> None:
    sample = RAGEvaluationSample(
        query="What are indexes used for?",
        contexts=[
            "Indexes improve query performance.",
            "Indexes help locate rows efficiently.",
        ],
        answer="They improve query performance.",
    )

    evaluator = BasicRAGEvaluator()

    result = evaluator.evaluate(sample)

    assert result.metadata["num_contexts"] == 2


def test_basic_evaluator_records_reference_answer_presence() -> None:
    sample = RAGEvaluationSample(
        query="What are indexes used for?",
        contexts=["Indexes improve query performance."],
        answer="They improve query performance.",
        reference_answer=(
            "Indexes improve query performance by "
            "helping locate rows efficiently."
        ),
    )

    evaluator = BasicRAGEvaluator()

    result = evaluator.evaluate(sample)

    assert result.metadata["has_reference_answer"] is True


def test_basic_evaluator_uses_faithfulness_scorer() -> None:
    class FakeFaithfulnessScorer:
        def score(
            self,
            answer: str,
            context: list[str],
        ) -> float:
            assert answer == "They improve query performance."
            assert context == [
                "Indexes improve query performance."
            ]

            return 0.8

    sample = RAGEvaluationSample(
        query="What are indexes used for?",
        contexts=["Indexes improve query performance."],
        answer="They improve query performance.",
    )

    evaluator = BasicRAGEvaluator(
        faithfulness_scorer=FakeFaithfulnessScorer(),
    )

    result = evaluator.evaluate(sample)

    assert result.faithfulness == 0.8


def test_basic_evaluator_uses_answer_relevancy_scorer() -> None:
    class FakeAnswerRelevancyScorer:
        def score(
            self,
            query: str,
            answer: str,
        ) -> AnswerRelevancyResult:
            assert query == "What are indexes used for?"
            assert answer == "They improve query performance."

            return AnswerRelevancyResult(
                relevant=True,
                score=0.5,
                reason="Partially relevant.",
            )

    sample = RAGEvaluationSample(
        query="What are indexes used for?",
        contexts=["Indexes improve query performance."],
        answer="They improve query performance.",
    )

    evaluator = BasicRAGEvaluator(
        answer_relevancy_scorer=FakeAnswerRelevancyScorer(),
    )

    result = evaluator.evaluate(sample)

    assert result.answer_relevancy == 0.5